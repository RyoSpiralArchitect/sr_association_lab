#!/usr/bin/env python3
"""SR Association Lab: a one-file, local causal-LM observation harness.

Run `python sr_association_lab.py --help`. Requires numpy/matplotlib for analysis,
and PyTorch/Transformers OR MLX/mlx-lm for inference. No service or judge is used.

Alignment: row r contains the state of prefix prompt + generated[:r], and the
raw logits predicting generated[r]. Row 0 is the last prompt position. A final
consume-only forward writes row N, so generated token t has its post-token state
at row t+1. Only meta.json.valid_rows are data; memmap tails are allocation.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import unicodedata
from typing import Any

import numpy as np

SCHEMA_VERSION = "association-lab/0.3"


def atomic_json(path: Path, data: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def probabilities(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or np.isnan(x).any() or np.isposinf(x).any() or not np.isfinite(x).any():
        raise ValueError("Logits must be a 1-D vector with at least one finite value and no NaN/+inf.")
    shifted = x - np.max(x)
    z = np.exp(shifted).sum()
    logp = shifted - math.log(float(z))
    return np.exp(logp), logp


def entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))


def sampling_distribution(raw: np.ndarray, temperature: float, top_k: int,
                          top_p: float) -> tuple[np.ndarray, np.ndarray]:
    """CPU float64 policy, stable ties by token ID; never mutate raw logits."""
    if not math.isfinite(temperature) or not math.isfinite(top_p) or temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Finite temperature >= 0, top_k >= 0, and finite 0 < top_p <= 1 are required.")
    probabilities(raw)  # validate the raw distribution even for greedy decoding
    if temperature == 0:
        scores = np.full(raw.shape, -np.inf, dtype=np.float64)
        scores[int(np.argmax(raw))] = 0.0
        return scores, np.exp(scores)
    scores = np.asarray(raw, dtype=np.float64).copy() / temperature
    order = np.argsort(-scores, kind="stable")
    if top_k and top_k < len(scores):
        scores[order[top_k:]] = -np.inf
    p, _ = probabilities(scores)
    if top_p < 1:
        # Keep the first token crossing the threshold, including when it is top 1.
        count = min(len(scores), int(np.searchsorted(np.cumsum(p[order]), top_p, side="left")) + 1)
        scores[order[count:]] = -np.inf
        p, _ = probabilities(scores)
    return scores, p


@dataclasses.dataclass
class GenerationOptions:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 43
    save_logits: bool = True
    save_policy_logits: bool = False
    hidden_dtype: str = "float32"
    stop_on_newline: bool = False
    stop_on_sentence: bool = False
    max_context: int = 4096
    top_candidates: int = 12
    use_cache: bool = True

    def validate(self) -> None:
        if self.max_new_tokens < 1 or self.max_context < 2 or self.top_candidates < 1:
            raise ValueError("Token limits and top_candidates must be positive.")
        if self.hidden_dtype not in ("float16", "float32"):
            raise ValueError("hidden_dtype must be float16 or float32.")
        sampling_distribution(np.array([0.0, 1.0]), self.temperature, self.top_k, self.top_p)


@dataclasses.dataclass
class LiveSession:
    """Exact consumed token ledger and its live KV cache; never a text round trip.

    A failed forward may mutate modern Cache objects in place. Such a session
    is invalid until reset(), even when its last committed ledger is intact.
    """
    token_ids: list[int] = dataclasses.field(default_factory=list)
    past_key_values: Any = None
    valid: bool = True
    reset_count: int = 0
    model_identity: int | None = None

    def reset(self):
        self.past_key_values = None
        self.token_ids = []
        self.valid = True
        self.model_identity = None
        self.reset_count += 1


def save_session(path, session):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {"token_ids": session.token_ids,
        "logical_cache_tokens": len(session.token_ids), "valid": session.valid,
        "reset_count": session.reset_count, "cache_tensors_serialized": False})


class HookRecorder:
    """Read-only last-position output hooks. No activation edits or global hooks."""

    def __init__(self, model, layers: str = "sample", patterns=None, enabled: bool = True):
        import torch
        self.torch = torch
        self.sites: list[dict] = []
        self.handles = []
        self.snapshots: dict = {}
        modules = dict(model.named_modules())
        if not enabled:
            return
        selected = []
        suffixes = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers",
                    "transformer.blocks", "backbone.layers", "model.language_model.layers")
        block_path = next((name for suffix in suffixes for name, module in modules.items()
                           if (name == suffix or name.endswith("." + suffix))
                           and isinstance(module, torch.nn.ModuleList)), None)
        if layers != "none":
            if block_path is None:
                raise ValueError("Decoder block list not recognized. Use --layers none --hook 'EXACT.PATH' or --no-hooks.")
            count = len(modules[block_path])
            if layers == "all":
                indices = list(range(count))
            elif layers == "sample":
                indices = sorted({0, count // 4, count // 2, 3 * count // 4, count - 1})
            else:
                indices = sorted({int(x) % count if -count <= int(x) < count else count
                                  for x in layers.split(",")})
                if count in indices:
                    raise ValueError(f"Layer index out of range for {count} decoder layers.")
            selected += [f"{block_path}.{i}" for i in indices]
            norm_suffixes = ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm",
                             "model.decoder.final_layer_norm", "model.language_model.norm")
            norms = [n for n in modules if any(n == s or n.endswith("." + s) for s in norm_suffixes)]
            if norms:
                selected.append(norms[0])
        for pattern in patterns or []:
            matches = [n for n in modules if n and fnmatch.fnmatchcase(n, pattern)]
            if not matches:
                raise ValueError(f"Hook pattern matched no modules: {pattern!r}. Use the modules command.")
            selected.extend(matches)
        try:
            for name in dict.fromkeys(selected):
                self.sites.append({"name": name, "path": name, "site": "first_rank3_output_last_position"})
                self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))
        except BaseException:
            self.close()
            raise

    def _make_hook(self, name):
        def first_tensor(value):
            if isinstance(value, self.torch.Tensor) and value.ndim == 3:
                return value
            if isinstance(value, (list, tuple)):
                for item in value:
                    found = first_tensor(item)
                    if found is not None:
                        return found
            if isinstance(value, dict):
                for item in value.values():
                    found = first_tensor(item)
                    if found is not None:
                        return found
            return None

        def hook(module, args, output):
            value = first_tensor(output)
            if value is None or value.shape[0] != 1:
                raise ValueError(f"{name}: expected an output tensor [1, sequence, width].")
            if name in self.snapshots:
                raise ValueError(f"{name}: called repeatedly in one forward; this hook site is unsupported.")
            # Clone only this position; a view would retain the entire prefill tensor.
            self.snapshots[name] = value[0, -1, :].detach().clone()
            return None
        return hook

    def clear(self):
        self.snapshots.clear()

    def take(self) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        missing = [x["name"] for x in self.sites if x["name"] not in self.snapshots]
        if missing:
            raise ValueError(f"Selected hooks did not run: {missing}")
        dtypes = {name: str(t.dtype) for name, t in self.snapshots.items()}
        arrays = {name: t.float().cpu().numpy() for name, t in self.snapshots.items()}
        self.clear()
        return arrays, dtypes

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def model_devices(model) -> list:
    import torch
    devices = {p.device for p in model.parameters() if p.device.type != "meta"}
    for value in getattr(model, "hf_device_map", {}).values():
        if value != "disk":
            devices.add(torch.device(f"cuda:{value}" if isinstance(value, int) else value))
    return sorted(devices, key=str)


def synchronize(devices) -> None:
    import torch
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()


def input_device(model):
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        hook = getattr(model.get_input_embeddings(), "_hf_hook", None)
        if hook is None or getattr(hook, "execution_device", None) is None:
            raise ValueError("Cannot resolve the embedding execution device for this offloaded model.")
        device = hook.execution_device
    return device


# ---- Native MLX backend (kept in this monolithic file; imported lazily) ----

class MLXModelHandle:
    """Small facade for the common recorder; native MLX modules stay untouched."""
    _association_backend = "mlx"

    def __init__(self, model, config, *, model_path=None, eos_ids=None, resolved_commit=None):
        from types import SimpleNamespace
        self.model = model
        self.config_dict = copy.deepcopy(config)
        if resolved_commit is not None:
            self.config_dict["_commit_hash"] = resolved_commit
        self.config = SimpleNamespace(**self.config_dict)
        self.generation_config = SimpleNamespace(eos_token_id=list(eos_ids or []))
        self.model_path = str(model_path) if model_path is not None else None
        self._active_recorder = None

    def eval(self):
        self.model.eval()
        return self


class MLXTokenizerAdapter:
    """mlx-lm's wrapper delegates attributes, but Python does not delegate __call__."""
    def __init__(self, wrapped):
        self._wrapped = wrapped
        # An explicit --tokenizer is already a callable HF tokenizer. Its own
        # _tokenizer may instead be the non-callable Rust tokenizer engine.
        self._hf = wrapped if callable(wrapped) else wrapped._tokenizer

    def __call__(self, *args, **kwargs):
        return self._hf(*args, **kwargs)

    def __len__(self):
        return len(self._hf)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def load_mlx_model(args):
    """Load an MLX checkpoint; no PyTorch, opaque generate(), or implicit KV trimming."""
    try:
        import mlx.core as mx
        from mlx_lm import load as native_load
        from mlx_lm import utils as mlx_utils
    except ImportError as exc:
        raise ImportError("MLX inference needs mlx and mlx-lm. On Apple silicon: "
                          "pip install 'mlx==0.32.4' 'mlx-lm==0.32.2'. "
                          "On Linux CPU: pip install 'mlx[cpu]==0.32.4' 'mlx-lm==0.32.2'.") from exc
    if getattr(args, "device_map", None):
        raise ValueError("--device-map is a PyTorch/Accelerate option; MLX uses one selected device.")
    if getattr(args, "load_in_4bit", False):
        raise ValueError("--load-in-4bit is bitsandbytes-only. For MLX, load a pre-quantized MLX checkpoint.")
    if getattr(args, "attn_implementation", "eager") != "eager":
        raise ValueError("--attn-implementation selects a Transformers implementation. "
                         "Omit it for MLX, which uses its model's native attention kernels.")
    device = getattr(args, "device", "auto")
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("MLX --device must be auto, cpu, or gpu (Metal on Apple silicon).")
    if device != "auto":
        selected = mx.cpu if device == "cpu" else mx.gpu
        if not mx.is_available(selected):
            raise ValueError(f"MLX device {device!r} is not available in this runtime.")
        mx.set_default_device(selected)
    mx.random.seed(int(getattr(args, "seed", 43)) % (2**32))
    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        # mlx-lm 0.31 keeps these patterns inside its downloader; 0.32 exports
        # the constant. Local checkpoint loading must work with either API.
        allow_patterns = getattr(mlx_utils, "DEFAULT_ALLOW_PATTERNS", [
            "*.json", "model*.safetensors", "*.py", "tokenizer.model",
            "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja",
        ])
        model_path = Path(snapshot_download(args.model, revision=getattr(args, "revision", None),
            local_files_only=getattr(args, "local_files_only", False), allow_patterns=allow_patterns))
    if not model_path.is_dir():
        raise ValueError("MLX --model must resolve to a checkpoint directory.")
    trust = getattr(args, "trust_remote_code", False)
    # Older loaders execute model_file without accepting a trust flag. Preserve
    # the CLI's explicit trust contract before invoking either loader version.
    checkpoint_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if checkpoint_config.get("model_file") is not None and not trust:
        raise ValueError("This MLX checkpoint contains custom model code; --trust-remote-code is required.")
    load_kwargs = {"return_config": True,
        "tokenizer_config": {"trust_remote_code": trust, "local_files_only": True},
        "adapter_path": getattr(args, "adapter", None)}
    if "trust_remote_code" in inspect.signature(native_load).parameters:
        load_kwargs["trust_remote_code"] = trust
    native, wrapped, config = native_load(str(model_path), **load_kwargs)
    if getattr(args, "tokenizer", None):
        from transformers import AutoTokenizer
        wrapped = AutoTokenizer.from_pretrained(args.tokenizer,
            revision=getattr(args, "revision", None), trust_remote_code=trust,
            local_files_only=getattr(args, "local_files_only", False))
    tokenizer = MLXTokenizerAdapter(wrapped)
    dtype = getattr(args, "dtype", "auto")
    if dtype != "auto":
        # Module.set_dtype's default predicate casts floating types only; packed
        # integer quantization weights retain their representation.
        native.set_dtype(getattr(mx, dtype))
    native.eval()
    mx.eval(native.parameters())
    mx.synchronize()
    resolved_commit = config.get("_commit_hash")
    if model_path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40,64}", model_path.name):
        resolved_commit = model_path.name
    eos = config.get("eos_token_id")
    eos_ids = {int(eos)} if isinstance(eos, int) else set(eos or [])
    tokenizer_eos = getattr(wrapped, "eos_token_ids", None)
    eos_ids.update([int(tokenizer_eos)] if isinstance(tokenizer_eos, (int, np.integer)) else tokenizer_eos or [])
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    return MLXModelHandle(native, config, model_path=model_path,
                          eos_ids=sorted(eos_ids), resolved_commit=resolved_commit), tokenizer


def mlx_model_modules(handle):
    return dict(handle.model.named_modules())


def _mlx_norm_path(modules):
    suffixes = ("model.norm", "model.ln_f", "transformer.ln_f", "model.final_layernorm", "model.final_layer_norm",
                "model.decoder.final_layer_norm", "model.language_model.norm", "backbone.norm_f",
                "language_model.model.norm", "norm")
    return next((name for suffix in suffixes for name in modules
                 if name and (name == suffix or (suffix != "norm" and name.endswith("." + suffix)))), None)


class MLXHookRecorder:
    """Read-only output observation via local, reversible MLX child-module wrappers.

    Wrappers call the original module without altering arguments or outputs and
    delegate architectural attributes. They are installed after weight loading.
    No class/global monkeypatches are used. Only one active recorder per handle.
    """
    def __init__(self, model, layers="sample", patterns=None, enabled=True, *, pooling="last"):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_unflatten
        self.mx = mx
        self.model = model
        self.sites, self.restorations, self.snapshots = [], [], {}
        self.pooling = pooling
        self._tree_unflatten = tree_unflatten
        self.closed = False
        if not enabled:
            return
        if model._active_recorder is not None:
            raise ValueError("Close the current MLX hook recorder before attaching another.")
        if pooling not in ("last", "mean"):
            raise ValueError("MLX capture pooling must be last or mean.")
        modules = mlx_model_modules(model)
        selected = []
        if layers != "none":
            suffixes = ("model.layers", "model.h", "transformer.h", "gpt_neox.layers", "model.decoder.layers",
                        "transformer.blocks", "backbone.layers", "model.language_model.layers", "layers")
            prefixes = sorted({name.rsplit(".", 1)[0] for name in modules
                               if "." in name and name.rsplit(".", 1)[1].isdigit()})
            block_path = next((name for suffix in suffixes for name in prefixes
                               if name == suffix or name.endswith("." + suffix)), None)
            if block_path is None:
                raise ValueError("MLX decoder block paths not recognized. Use --layers none "
                                 "--hook 'EXACT.PATH' (see modules), or --no-hooks.")
            indices = sorted(int(name[len(block_path)+1:]) for name in modules
                             if name.startswith(block_path + ".") and name[len(block_path)+1:].isdigit())
            count = len(indices)
            if indices != list(range(count)) or not count:
                raise ValueError("MLX decoder block list must have contiguous zero-based indices.")
            if layers == "all":
                selected_indices = indices
            elif layers == "sample":
                selected_indices = sorted({0, count // 4, count // 2, 3 * count // 4, count - 1})
            else:
                requested = [int(x) for x in layers.split(",")]
                if any(x < -count or x >= count for x in requested):
                    raise ValueError(f"Layer index out of range for {count} MLX decoder layers.")
                selected_indices = sorted({x % count for x in requested})
            selected.extend(f"{block_path}.{i}" for i in selected_indices)
            norm_path = _mlx_norm_path(modules)
            if norm_path:
                selected.append(norm_path)
        for pattern in patterns or []:
            matches = [name for name in modules if name and fnmatch.fnmatchcase(name, pattern)]
            if not matches:
                raise ValueError(f"MLX hook pattern matched no modules: {pattern!r}. Use the modules command.")
            selected.extend(matches)
        selected = list(dict.fromkeys(selected))
        self.sites = [{"name": name, "path": name,
            "site": "first_rank3_output_last_position" if pooling == "last" else "first_rank3_output_mean_all_positions",
            "implementation": "mlx-local-output-wrapper"} for name in selected]
        recorder = self

        class OutputObserver(nn.Module):
            def __init__(self, wrapped, name):
                super().__init__()
                # Direct mapping assignment avoids delegation during initialization.
                self["wrapped"] = wrapped
                self.observer_name = name

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    wrapped = self.get("wrapped")
                    if wrapped is None:
                        raise
                    return getattr(wrapped, name)

            def __call__(self, *args, **kwargs):
                output = self.wrapped(*args, **kwargs)
                recorder._capture(self.observer_name, output)
                return output

        try:
            # Nested sites are supported: children are installed before their
            # parents; close restores parents before their original children.
            for name in sorted(selected, key=lambda n: n.count("."), reverse=True):
                original = modules[name]
                model.model.update_modules(tree_unflatten([(name, OutputObserver(original, name))]))
                self.restorations.append((name, original))
            if self.sites:
                model._active_recorder = self
        except BaseException:
            self.close()
            raise

    def _capture(self, name, output):
        def first(value):
            if isinstance(value, self.mx.array) and value.ndim == 3:
                return value
            if isinstance(value, (tuple, list)):
                return next((t for v in value if (t := first(v)) is not None), None)
            if isinstance(value, dict):
                return next((t for v in value.values() if (t := first(v)) is not None), None)
            return None
        value = first(output)
        if value is None or value.shape[0] != 1 or value.shape[1] == 0:
            raise ValueError(f"{name}: expected an MLX output [1, sequence, width].")
        if name in self.snapshots:
            raise ValueError(f"{name}: called repeatedly in one forward; this hook site is unsupported.")
        self.snapshots[name] = (value[0, -1, :] if self.pooling == "last"
                                else value[0].astype(self.mx.float32).mean(axis=0))

    def clear(self):
        self.snapshots.clear()

    def take(self):
        missing = [site["name"] for site in self.sites if site["name"] not in self.snapshots]
        if missing:
            raise ValueError(f"Selected MLX hooks did not run: {missing}")
        # Site order is stable even though child wrappers execute before parents.
        dtypes = {site["name"]: str(self.snapshots[site["name"]].dtype) for site in self.sites}
        arrays = {site["name"]: np.array(self.snapshots[site["name"]].astype(self.mx.float32), copy=True)
                  for site in self.sites}
        self.clear()
        return arrays, dtypes

    def close(self):
        for name, original in reversed(self.restorations):
            self.model.model.update_modules(self._tree_unflatten([(name, original)]))
        self.restorations.clear()
        self.clear()
        if self.model._active_recorder is self:
            self.model._active_recorder = None
        self.closed = True


def mlx_cache_description(cache):
    return [{"class": type(c).__name__,
             **({"max_size": int(c.max_size)} if isinstance(getattr(c, "max_size", None), int) else {})}
            for c in cache]


def mlx_record_forward(handle, ids, prefix_len, past, use_cache, hooks):
    """Same step contract as PyTorch; evaluate lazy graph + cache before stopping timer."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    cache = past
    if use_cache and cache is None:
        cache = make_prompt_cache(handle.model)
    if not use_cache:
        cache = None
    if cache is not None:
        expected_offset = prefix_len - len(ids)
        for c in cache:
            offset = getattr(c, "offset", None)
            if isinstance(offset, int) and offset != expected_offset:
                raise ValueError(f"MLX cache offset {offset} does not match token ledger {expected_offset}.")
    hooks.clear()
    inputs = mx.array([ids], dtype=mx.int32)
    mx.eval(inputs)
    mx.synchronize()
    started = time.perf_counter()
    output = handle.model(inputs, cache=cache)
    if not isinstance(output, mx.array) or output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("MLX model must return logits [1, sequence, vocabulary].")
    raw_tensor = output[0, -1, :]
    # Cache tensors can retain deferred updates even when logits have evaluated.
    # Include snapshots and all native cache state in the same synchronous eval.
    mx.eval(raw_tensor, list(hooks.snapshots.values()), [c.state for c in cache] if cache is not None else [])
    mx.synchronize()
    forward_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    raw_dtype = str(raw_tensor.dtype)
    raw = np.array(raw_tensor.astype(mx.float32), copy=True)
    states, state_dtypes = hooks.take()
    capture_ms = (time.perf_counter() - started) * 1000
    if cache is not None:
        for c in cache:
            offset = getattr(c, "offset", None)
            if isinstance(offset, int) and offset != prefix_len:
                raise ValueError(f"MLX cache did not consume the full prefix: {offset} != {prefix_len}.")
    return raw, states, state_dtypes, raw_dtype, forward_ms, capture_ms, cache


def mlx_encode_isolated(handle, tokenizer, texts, max_context=4096):
    """Unnormalized means of final normalized hidden states, with a fresh no-KV forward per text."""
    import mlx.core as mx
    if handle._active_recorder is not None:
        raise ValueError("Close generation hooks before isolated MLX encoding.")
    norm_path = _mlx_norm_path(mlx_model_modules(handle))
    if norm_path is None:
        raise ValueError("No recognized final normalization module for MLX --similarity self. "
                         "Use --similarity sentence-transformer or none for this architecture.")
    model_limit = (getattr(handle.config, "max_position_embeddings", None)
                   or getattr(handle.config, "n_positions", max_context))
    limit = min(max_context, int(model_limit or max_context))
    arrays, token_ids = [], []
    observer = MLXHookRecorder(handle, layers="none", patterns=[norm_path], pooling="mean")
    try:
        for text in texts:
            ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            if not ids or len(ids) > limit:
                raise ValueError("Similarity text is empty after tokenization or exceeds context; no silent truncation.")
            observer.clear()
            handle.model(mx.array([ids], dtype=mx.int32), cache=None)
            mx.eval(list(observer.snapshots.values()))
            mx.synchronize()
            states, _ = observer.take()
            arrays.append(states[norm_path])
            token_ids.append(ids)
    finally:
        observer.close()
    if not arrays:
        raise ValueError("MLX isolated encoding requires at least one text.")
    return np.stack(arrays), token_ids, {"backend": "mlx", "hidden_site": norm_path,
        "pooling": "mean of final normalization output over raw-text tokens; no special tokens or chat template",
        "native_cache_used": False, "generation_history_used": False,
        "vectors_returned_normalized": False}


def mlx_runtime_metadata(model, tokenizer, args, hooks, synthetic=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import get_total_parameters
    packages = {}
    for name in ("mlx", "mlx-metal", "mlx-cpu", "mlx-cuda-12", "mlx-cuda-13", "mlx-lm",
                 "transformers", "numpy", "matplotlib", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = copy.deepcopy(model.config_dict)
    params = tree_flatten(model.model.parameters())
    return {"schema_version": SCHEMA_VERSION, "harness": "sr_association_lab", "created_at": now(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running", "config": vars(args), "backend": "mlx", "model": {
            "identifier": "tiny-random-mlx-llama" if synthetic else args.model,
            "synthetic": synthetic, "untrained": synthetic, "class": type(model.model).__name__,
            "config": config, "resolved_commit": getattr(model.config, "_commit_hash", None),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
            "adapter": getattr(args, "adapter", None), "resolved_model_path": model.model_path,
            "parameter_count": int(get_total_parameters(model.model)),
            "stored_parameter_elements": sum(int(p.size) for _, p in params),
            "parameter_dtypes": sorted({str(p.dtype) for _, p in params}),
            "devices": [str(mx.default_device())], "hf_device_map": {},
            "native_cache": mlx_cache_description(make_prompt_cache(model.model)),
            "cache_policy": "mlx_lm.make_prompt_cache(model); model-native sliding/recurrent policies retained; harness imposes no rotation or trimming"},
        "tokenizer": {"identifier": getattr(tokenizer, "name_or_path", None),
            "class": type(getattr(tokenizer, "_wrapped", tokenizer)).__name__, "vocab_size": len(tokenizer),
            "special_ids": tokenizer.all_special_ids, "chat_template": tokenizer.chat_template,
            "eos_ids": model.generation_config.eos_token_id,
            "resolved_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")},
        "runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "device": str(mx.default_device()), "device_info": mx.device_info(),
            "gpu_available": bool(mx.is_available(mx.gpu)), "lazy_evaluation": True},
        "hook_sites": hooks.sites,
        "sampling": "CPU NumPy float64; temperature -> exact top-k -> nucleus top-p; stable ties by token ID; no other processors",
        "alignment": "row r = final prefix position of prompt+generated[:r]; token t post-state is row t+1; last row consumes final token without sampling",
        "timing": "instrumented_forward_ms includes native forward, captured hook arrays, cache-state mx.eval and mx.synchronize; host conversion, sampling/statistics and I/O separate; final consume separate",
        "context": "live retains consumed token IDs and native MLX cache across turns; carry re-renders text and prefills; reset starts fresh. No harness truncation.",
        "plot_semantics": "Per-event units are prompted responses; token IDs are not linguistic words; pooled states use non-special non-whitespace generated positions."}


class EventWriter:
    def __init__(self, directory: Path, options: GenerationOptions, meta: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.options = options
        self.arrays = {}
        self.meta = {**meta, "status": "running", "valid_rows": 0, "n_tokens": 0,
                     "arrays": {}, "hidden": {}, "generated_ids": [], "started_at": now()}
        self.token_file = (self.directory / "tokens.jsonl").open("w", encoding="utf-8")
        self.checkpoint()

    def allocate(self, key, filename, width, dtype):
        array = np.lib.format.open_memmap(self.directory / filename, mode="w+", dtype=dtype,
                                         shape=(self.options.max_new_tokens + 1, width))
        self.arrays[key] = array
        return array

    def write_row(self, row, raw, policy, states, source_dtypes, logit_dtype):
        started = time.perf_counter()
        if row == 0:
            self.meta["logits_source_dtype"] = logit_dtype
            self.meta["logits_stored_dtype"] = "float32"
            self.meta["vocab_size"] = len(raw)
            if self.options.save_logits:
                self.allocate("raw", "raw_logits.npy", len(raw), "float32")
                self.meta["arrays"]["raw_logits"] = "raw_logits.npy"
            if self.options.save_policy_logits:
                self.allocate("policy", "policy_logits.npy", len(raw), "float32")
                self.meta["arrays"]["policy_logits"] = "policy_logits.npy"
            for i, (name, state) in enumerate(states.items()):
                filename = f"hook_{i:03d}.npy"
                self.allocate("hook:" + name, filename, len(state), self.options.hidden_dtype)
                self.meta["hidden"][name] = {"file": filename, "width": len(state),
                    "source_dtype": source_dtypes[name], "stored_dtype": self.options.hidden_dtype}
        if "raw" in self.arrays:
            self.arrays["raw"][row] = raw
        if "policy" in self.arrays:
            self.arrays["policy"][row] = policy
        for name, state in states.items():
            self.arrays["hook:" + name][row] = state
        # A completed row is flushed before its valid_rows commit marker is updated.
        for array in self.arrays.values():
            array.flush()
        self.meta["valid_rows"] = row + 1
        return (time.perf_counter() - started) * 1000

    def token(self, record):
        self.token_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.token_file.flush()
        os.fsync(self.token_file.fileno())

    def checkpoint(self):
        atomic_json(self.directory / "meta.json", self.meta)

    def close(self):
        self.token_file.close()
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()


def record_generation(model, tokenizer, prompt_ids, event_dir, *, options: GenerationOptions,
                      hooks: HookRecorder, eos_ids=None, metadata=None,
                      session: LiveSession | None = None,
                      rng: np.random.Generator | None = None) -> dict:
    """Batch=1 autoregression. With a live session, prompt_ids are NEW IDs only."""
    mlx_backend = getattr(model, "_association_backend", None) == "mlx"
    if not mlx_backend:
        import torch
    options.validate()
    appended_ids = [int(t) for t in prompt_ids]
    if not appended_ids:
        raise ValueError("Prompt token sequence is empty.")
    if session is not None:
        if not options.use_cache:
            raise ValueError("Live sessions require KV caching; --no-cache is incompatible.")
        if not session.valid:
            raise ValueError("Live session is invalid after a failed forward; reset it before reuse.")
        if session.model_identity not in (None, id(model)):
            raise ValueError("Live session belongs to another model; reset it before reuse.")
        if bool(session.token_ids) != (session.past_key_values is not None):
            raise ValueError("Live session ledger/cache mismatch; reset it before reuse.")
    previous_ids = list(session.token_ids) if session is not None else []
    prompt_ids = previous_ids + appended_ids
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if model_limit is None:
        model_limit = getattr(model.config, "n_positions", options.max_context)
    context_limit = min(options.max_context, int(model_limit or options.max_context))
    if len(prompt_ids) + options.max_new_tokens > context_limit:
        raise ValueError(f"Context budget exceeded: {len(prompt_ids)} prompt + {options.max_new_tokens} new > {context_limit}. No silent truncation.")
    if eos_ids is None:
        values = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos_ids = [values] if isinstance(values, int) else list(values or [])
        if tokenizer.eos_token_id is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        tokenizer_eos = getattr(tokenizer, "eos_token_ids", None)
        eos_ids.extend([int(tokenizer_eos)] if isinstance(tokenizer_eos, (int, np.integer))
                       else [int(x) for x in tokenizer_eos or []])
    eos_set = set(int(x) for x in eos_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", [])) | eos_set
    rng = np.random.default_rng(options.seed) if rng is None else rng
    writer = EventWriter(Path(event_dir), options, {
        **(metadata or {}), "schema_version": SCHEMA_VERSION, "prompt_ids": prompt_ids,
        "backend": "mlx" if mlx_backend else "torch",
        "appended_prompt_ids": appended_ids, "past_tokens": len(previous_ids),
        "new_prompt_tokens": len(appended_ids), "cache_before_tokens": len(previous_ids),
        "cache_reused": bool(previous_ids), "live_session": session is not None,
        "rng_state_before": copy.deepcopy(rng.bit_generator.state),
        "prompt_tokens": len(prompt_ids), "generation": dataclasses.asdict(options),
        "hook_sites": hooks.sites, "eos_ids": sorted(eos_set), "effective_context_limit": context_limit,
        "alignment": "row r: prefix=prompt+generated[:r]; sampled token t is consumed at row t+1",
    })
    generated = []
    records = []
    devices = [] if mlx_backend else model_devices(model)
    device = None if mlx_backend else input_device(model)
    params = {} if mlx_backend else inspect.signature(model.forward).parameters
    past = session.past_key_values if session is not None else None
    forward_attempted = False
    previous_p = None
    event_start = time.perf_counter()
    model.eval()

    def forward(row):
        nonlocal past, forward_attempted
        prefix = prompt_ids + generated
        if options.use_cache and row > 0:
            ids = prefix[-1:]
        elif options.use_cache and past is not None:
            ids = appended_ids
        else:
            ids = prefix
        if mlx_backend:
            forward_attempted = True
            raw, states, state_dtypes, raw_dtype, fms, cms, past = mlx_record_forward(
                model, ids, len(prefix), past, options.use_cache, hooks)
            return raw, states, state_dtypes, raw_dtype, fms, cms
        kwargs = {"input_ids": torch.tensor([ids], dtype=torch.long, device=device),
                  "attention_mask": torch.ones((1, len(prefix)), dtype=torch.long, device=device),
                  "use_cache": options.use_cache, "return_dict": True}
        if options.use_cache and past is not None:
            kwargs["past_key_values"] = past
        if "position_ids" in params:
            kwargs["position_ids"] = torch.arange(len(prefix) - len(ids), len(prefix), device=device).unsqueeze(0)
        if "cache_position" in params:
            kwargs["cache_position"] = torch.arange(len(prefix) - len(ids), len(prefix), device=device)
        if "logits_to_keep" in params:
            kwargs["logits_to_keep"] = 1
        elif "num_logits_to_keep" in params:
            kwargs["num_logits_to_keep"] = 1
        hooks.clear()
        synchronize(devices)
        started = time.perf_counter()
        forward_attempted = True
        output = model(**kwargs)
        synchronize(devices)
        forward_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        raw_tensor = output.logits[0, -1, :].detach()
        raw_dtype = str(raw_tensor.dtype)
        raw = raw_tensor.float().cpu().numpy().copy()
        states, state_dtypes = hooks.take()
        past = getattr(output, "past_key_values", None) if options.use_cache else None
        if options.use_cache and past is None:
            raise ValueError("Model did not return a KV cache; retry with --no-cache.")
        capture_ms = (time.perf_counter() - started) * 1000
        return raw, states, state_dtypes, raw_dtype, forward_ms, capture_ms

    try:
        with contextlib.nullcontext() if mlx_backend else torch.inference_mode():
            for t in range(options.max_new_tokens):
                raw, states, state_dtypes, raw_dtype, fms, cms = forward(t)
                started = time.perf_counter()
                raw_p, raw_logp = probabilities(raw)
                policy, policy_p = sampling_distribution(raw, options.temperature, options.top_k, options.top_p)
                token_id = int(np.argmax(policy_p)) if options.temperature == 0 else int(rng.choice(len(policy_p), p=policy_p))
                order = np.argsort(-raw, kind="stable")
                top = order[:min(options.top_candidates, len(raw))]
                gap = float(raw[order[0]]) - float(raw[order[1]]) if len(raw) > 1 else None
                if gap is not None and not math.isfinite(gap):
                    gap = None
                js = None
                if previous_p is not None:
                    midpoint = (raw_p + previous_p) / 2
                    a, b = raw_p > 0, previous_p > 0
                    js = float(0.5 * (np.sum(raw_p[a] * np.log(raw_p[a] / midpoint[a])) +
                                     np.sum(previous_p[b] * np.log(previous_p[b] / midpoint[b]))))
                    js = max(0.0, js)
                previous_p = raw_p
                sampling_ms = (time.perf_counter() - started) * 1000
                sample_ready_ms = (time.perf_counter() - event_start) * 1000
                wms = writer.write_row(t, raw, policy, states, state_dtypes, raw_dtype)
                generated.append(token_id)
                text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                record = {"t": t, "phase": ("append_prefill" if previous_ids else "prefill") if t == 0 else "decode", "token_id": token_id,
                    "token_piece": str(tokenizer.convert_ids_to_tokens(token_id)),
                    "decoded_piece": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    "is_special": token_id in special_ids, "text_so_far": text,
                    "prefix_len": len(prompt_ids) + t, "readout_position": len(prompt_ids) + t - 1,
                    "consumed_token_id": prompt_ids[-1] if t == 0 else generated[t-1],
                    "predictor_row": t, "consumed_state_row": t + 1,
                    "raw_entropy_nats": entropy(raw_p), "policy_entropy_nats": entropy(policy_p),
                    "raw_surprisal_nats": float(-raw_logp[token_id]),
                    "policy_surprisal_nats": float(-math.log(policy_p[token_id])),
                    "raw_rank": int(np.count_nonzero(raw > raw[token_id])) + 1,
                    "raw_top1_p": float(raw_p[order[0]]), "raw_top1_gap": gap,
                    "raw_eos_p": float(sum(raw_p[i] for i in eos_set if 0 <= i < len(raw_p))),
                    "raw_js_prev_nats": js, "instrumented_forward_ms": fms,
                    "capture_to_cpu_ms": cms, "sampling_stats_ms": sampling_ms,
                    "sample_ready_ms": sample_ready_ms,
                    "array_write_ms": wms, "token_ready_ms": (time.perf_counter() - event_start) * 1000,
                    "top_raw": [{"id": int(i), "piece": str(tokenizer.convert_ids_to_tokens(int(i))),
                                 "p": float(raw_p[i]), "logit": float(raw[i]) if math.isfinite(float(raw[i])) else None}
                                for i in top]}
                records.append(record)
                writer.token(record)
                writer.meta.update(n_tokens=len(generated), generated_ids=generated.copy(), response=text)
                writer.checkpoint()
                if token_id in eos_set:
                    reason = "eos"
                    break
                if options.stop_on_sentence:
                    boundary = sentence_boundary(text)
                    if boundary is not None and boundary["complete"]:
                        reason = "sentence"
                        break
                if options.stop_on_newline and "\n" in text:
                    reason = "newline"
                    break
            else:
                reason = "length"
            # Consume the final sampled token exactly once, without sampling another.
            raw, states, state_dtypes, raw_dtype, fms, cms = forward(len(generated))
            policy, _ = sampling_distribution(raw, options.temperature, options.top_k, options.top_p)
            wms = writer.write_row(len(generated), raw, policy, states, state_dtypes, raw_dtype)
            content_indices = [i for i, token_id in enumerate(generated) if token_id not in special_ids
                               and tokenizer.decode([token_id], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False).strip()]
            response = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            writer.meta.update(status="complete", finished_at=now(), response=response,
                cache_after_tokens=len(prompt_ids) + len(generated) if options.use_cache else 0,
                rng_state_after=copy.deepcopy(rng.bit_generator.state),
                response_with_special_tokens=tokenizer.decode(generated, skip_special_tokens=False,
                                                              clean_up_tokenization_spaces=False),
                unit_text=response.split("\n", 1)[0].strip() if options.stop_on_newline else response.strip(),
                stop_reason=reason, censored=(reason == "length"), content_token_indices=content_indices,
                terminal={"row": len(generated), "phase": "final_consume", "sampled": False,
                          "prefix_len": len(prompt_ids) + len(generated),
                          "readout_position": len(prompt_ids) + len(generated) - 1,
                          "consumed_token_id": generated[-1], "instrumented_forward_ms": fms,
                          "capture_to_cpu_ms": cms, "array_write_ms": wms},
                event_wall_ms=(time.perf_counter() - event_start) * 1000,
                first_token_model_ms=records[0]["instrumented_forward_ms"],
                first_token_latency_ms=records[0]["sample_ready_ms"],
                response_model_ms=float(sum(r["instrumented_forward_ms"] for r in records)),
                response_wall_ms=records[-1]["sample_ready_ms"], terminal_consume_ms=fms,
                prefill_forward_ms=records[0]["instrumented_forward_ms"],
                decode_forward_ms_mean=float(np.mean([r["instrumented_forward_ms"] for r in records[1:]])) if len(records) > 1 else None)
            if options.stop_on_sentence:
                writer.meta["sentence_check"] = check_one_sentence(response, reason)
            writer.checkpoint()
            if session is not None:
                session.token_ids = prompt_ids + generated
                session.past_key_values = past
                session.model_identity = id(model)
            return writer.meta
    except BaseException as exc:
        if session is not None and forward_attempted:
            session.valid = False
            # No partial or possibly mutated cache can be reused accidentally.
            session.past_key_values = None
        writer.meta.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                           error_type=type(exc).__name__, error=str(exc), finished_at=now(),
                           session_invalidated=session is not None and forward_attempted,
                           rng_state_after=copy.deepcopy(rng.bit_generator.state),
                           event_wall_ms=(time.perf_counter() - event_start) * 1000)
        writer.checkpoint()
        raise
    finally:
        writer.close()
        hooks.clear()


PROTOCOLS = {
    "association": "これから刺激語を一語ずつ提示します。それぞれから最初に連想する語をただ一語だけ返してください。説明、文、番号、引用符は付けず、一語だけを一行で答えてください。\n刺激語：{cue}",
    "association_en": "I will present one stimulus word at a time. Respond with exactly ONE associated word. Give only that word on one line, without explanations, sentences, numbering, or quotes.\nStimulus: {cue}",
    "imagination": "提示した情景から連想される次の情景を、具体的な一文だけで描写してください。毎ターン、そこまでの情景を受けて続きを一文だけ返してください。物、動き、位置関係、音などを書き、句点（。）で終えてください。説明、分析、見出し、箇条書き、複数の文は不要です。\n起点の情景：{cue}",
    "imagination_en": "Describe the next associated scene in exactly ONE concrete sentence, continuing from the scene so far on every turn. Mention objects, movement, spatial relations, or sounds. End with sentence punctuation. Give only the sentence, without explanations, analysis, headings, lists, or multiple sentences.\nStarting scene: {cue}",
    "free": "{cue}",
}
CONTINUE_CUE = "そこから連想される次の情景を、一文だけ続けてください。"
CONTINUE_CUE_EN = "Continue with exactly one sentence describing the next associated scene."


def sentence_boundary(text: str, *, final: bool = False) -> dict | None:
    """A deterministic punctuation heuristic, not a grammatical sentence parser.

    Streaming cannot always disambiguate a dot at a token boundary. Numeric
    trailing dots wait for more text (or EOS); common abbreviations/initials
    are skipped. All sampled text, including boundary overshoot, is retained.
    """
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
                     "etc", "e.g", "i.e", "fig", "no", "inc", "dept"}
    pairs = {"「": "」", "『": "』", "（": "）", "(": ")", "[": "]", "【": "】", "“": "”", "{": "}"}
    closers = set(pairs.values()) | {'"', "'", "’"}
    for index, char in enumerate(text):
        if char not in ".。!?！？":
            continue
        if char == ".":
            before = text[index - 1] if index else ""
            after = text[index + 1] if index + 1 < len(text) else ""
            # Decimal points, domains and abbreviations are not assumed endings.
            if before.isalnum() and after.isalnum():
                continue
            match = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:index])
            word = match.group(1) if match else ""
            if word.lower() in abbreviations or (len(word) == 1 and word.isupper()):
                continue
            if before.isdigit() and not after and not final:
                continue
        end = index + 1
        while end < len(text) and text[end] in ".。!?！？":
            end += 1
        terminal = text[index:end]
        while end < len(text) and text[end] in closers:
            end += 1
        stack = []
        for pos, value in enumerate(text[:end]):
            if value == '"' and (pos == 0 or text[pos - 1] != "\\"):
                if stack and stack[-1] == '"':
                    stack.pop()
                else:
                    stack.append('"')
            elif value in pairs:
                stack.append(pairs[value])
            elif stack and value == stack[-1]:
                stack.pop()
        return {"start": index, "end": end, "terminal": terminal,
                "complete": not stack, "pending_closers": stack,
                "overshoot": text[end:], "policy": "punctuation-surface-v1"}
    return None


def check_one_sentence(response: str, stop_reason: str) -> dict:
    text = str(response).strip()
    boundary = sentence_boundary(text, final=True)
    reasons = []
    if not text or not any(unicodedata.category(c)[0] in "LN" for c in text):
        reasons.append("empty_or_no_content")
    if "\n" in text or "\r" in text:
        reasons.append("multiple_lines")
    if boundary is None:
        reasons.append("missing_terminal")
    else:
        if not boundary["complete"]:
            reasons.append("unclosed_delimiter")
        if boundary["overshoot"].strip():
            reasons.append("text_after_first_sentence")
    surface_ok = not reasons
    if stop_reason == "length":
        reasons.append("length_censored")
    return {"text": text, "surface_ok": surface_ok,
            "eligible": surface_ok and stop_reason != "length",
            "reasons": reasons, "boundary": boundary,
            "policy": "punctuation-surface-v1; no trimming or grammatical guarantee"}


def build_messages(protocol, cue, history):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    return [dict(x) for x in history] + [{"role": "user", "content": PROTOCOLS[protocol].format(cue=cue)}]


def render_prompt(tokenizer, messages, template="auto"):
    use_chat = template == "chat" or (template == "auto" and bool(getattr(tokenizer, "chat_template", None)))
    if use_chat:
        # Render once, then explicitly avoid a second special-token insertion.
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return text, list(ids), "chat"
    text = "\n\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"]
                         for m in messages) + "\n\nAssistant:"
    return text, list(tokenizer(text, add_special_tokens=True)["input_ids"]), "plain"


def build_live_prompt(tokenizer, protocol, cue, session, template="auto") -> dict:
    """Append native turn boundaries while preserving every earlier sampled ID.

    Later suffixes are tokenized independently. This is a defined wire format,
    not a claim of equality with re-tokenizing a whole rendered conversation.
    """
    if not session.valid:
        raise ValueError("Live session is invalid; reset it before building a prompt.")
    if not session.token_ids:
        text, ids, used = render_prompt(tokenizer, build_messages(protocol, cue, []), template)
        return {"ids": ids, "text": text, "wire_format": used + "-initial",
                "boundary_dedup_ids": []}
    use_chat = template == "chat" or (template == "auto" and bool(getattr(tokenizer, "chat_template", None)))
    if use_chat:
        # The marker is content, not a token inserted into the actual session.
        marker = "ASSOCIATION_LAB_ASSISTANT_BOUNDARY_7e61f8c2"
        if marker in str(cue):
            raise ValueError("Cue collides with the internal chat-boundary marker.")
        dummy = [{"role": "user", "content": "ASSOCIATION_LAB_FIRST_USER"},
                 {"role": "assistant", "content": marker},
                 {"role": "user", "content": str(cue)}]
        rendered = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
        if rendered.count(marker) != 1:
            raise ValueError("Chat template does not preserve the assistant boundary marker; use --template plain.")
        text = rendered.split(marker, 1)[1]
        wire = "chat-append-suffix"
    else:
        text = "\n\nUser: " + str(cue) + "\n\nAssistant:"
        wire = "plain-append-suffix"
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    # Deduplicate only a whole leading run of special closing tokens already
    # consumed by generation, never overlapping ordinary word/whitespace IDs.
    specials = set(getattr(tokenizer, "all_special_ids", []))
    special_prefix = 0
    while special_prefix < len(ids) and ids[special_prefix] in specials:
        special_prefix += 1
    removed = []
    for count in range(min(special_prefix, len(session.token_ids)), 0, -1):
        if session.token_ids[-count:] == ids[:count]:
            removed, ids = ids[:count], ids[count:]
            break
    if not ids:
        raise ValueError("Empty append-only turn suffix.")
    return {"ids": ids, "text": text, "wire_format": wire, "boundary_dedup_ids": removed}


def check_one_word(response: str, stop_reason: str) -> dict:
    """Conservative surface check; NOT a Japanese linguistic word segmenter."""
    text = str(response).strip()
    reasons = []
    if not text:
        reasons.append("empty")
    if any(c.isspace() for c in text):
        reasons.append("multiple_spans_or_lines")
    # Permit letters/numbers/marks plus internal hyphens/apostrophes/middle dots.
    allowed_joiners = "-'’・·"
    if text and (not any(unicodedata.category(c)[0] in "LN" for c in text) or
                 any(unicodedata.category(c)[0] not in "LNM" and c not in allowed_joiners for c in text) or
                 text[0] in allowed_joiners or text[-1] in allowed_joiners):
        reasons.append("non_word_surface")
    surface_ok = not reasons
    if stop_reason == "length":
        reasons.append("length_censored")
    return {"surface_ok": surface_ok, "eligible": surface_ok and stop_reason != "length",
            "reasons": reasons, "text": text}


def normalize_word_bank(words) -> list[str]:
    if not isinstance(words, (list, tuple)) or not words:
        raise ValueError("Word bank must be a nonempty list of strings.")
    normalized, seen = [], set()
    for item in words:
        if not isinstance(item, str):
            raise ValueError("Every word bank item must be a string.")
        word = unicodedata.normalize("NFC", item).strip()
        if not check_one_word(word, "eos")["surface_ok"]:
            raise ValueError(f"Bank item must be a single unquoted word: {item!r}")
        key = word.casefold()
        if key in seen:
            raise ValueError(f"Duplicate bank item: {word!r}")
        seen.add(key)
        normalized.append(word)
    return normalized


def word_bank_hash(words) -> str:
    canonical = json.dumps(normalize_word_bank(words), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_word_bank(text: str, size: int) -> dict:
    if size < 1:
        raise ValueError("Bank size must be positive.")
    source = text.strip()
    # Accept one fenced JSON array, but do not guess words from prose sentences.
    if source.startswith("```") and source.endswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source, count=1)
        source = re.sub(r"\s*```$", "", source, count=1)
    try:
        candidates = json.loads(source)
        if isinstance(candidates, dict) and isinstance(candidates.get("words"), list):
            candidates = candidates["words"]
        if not isinstance(candidates, list):
            raise ValueError
        format_name = "json"
    except (ValueError, TypeError):
        candidates = [re.sub(r"^\s*(?:[-*•]|\d+[.)、:])\s*", "", line).strip()
                      for line in source.splitlines() if line.strip()]
        format_name = "lines"
    words, rejected, duplicates, seen = [], [], [], set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, str):
            rejected.append({"index": index, "value": candidate, "reason": "not_string"})
            continue
        word = unicodedata.normalize("NFC", candidate).strip()
        check = check_one_word(word, "eos")
        if not check["surface_ok"]:
            rejected.append({"index": index, "value": candidate, "reason": check["reasons"]})
        elif word.casefold() in seen:
            duplicates.append(word)
        else:
            seen.add(word.casefold())
            words.append(word)
    return {"words": words[:size], "accepted": len(words) >= size,
            "rejected": rejected, "duplicates": duplicates, "extras": words[size:],
            "format": format_name, "requested_size": size, "valid_count": len(words)}


def build_trial_plan(words, seeds, repeats=1, order="fixed", order_seeds=(101,),
                     conditions=("live",)) -> list[dict]:
    words = normalize_word_bank(words)
    seeds = [int(s) for s in seeds]
    order_seeds = [int(s) for s in order_seeds]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or repeats < 1:
        raise ValueError("Use unique nonnegative response seeds and repeats >= 1.")
    if order not in ("fixed", "shuffle") or not order_seeds or min(order_seeds) < 0:
        raise ValueError("Use fixed/shuffle order and nonnegative order seeds.")
    if len(set(order_seeds)) != len(order_seeds):
        raise ValueError("Order seeds must be unique.")
    if not conditions or any(c not in ("live", "reset") for c in conditions) or len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be unique live/reset entries.")
    schedule = []
    for order_seed in (order_seeds if order == "shuffle" else [None]):
        permutation = (np.random.default_rng(order_seed).permutation(len(words)).tolist()
                       if order == "shuffle" else list(range(len(words))))
        for seed in seeds:
            for repeat in range(repeats):
                for condition in conditions:
                    schedule.append({"trial_id": f"t{len(schedule):04d}", "sample_seed": seed,
                        "repeat_index": repeat, "order_seed": order_seed,
                        "condition": condition, "bank_order": list(permutation)})
    return schedule


def load_word_bank(path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        record = content if isinstance(content, dict) else {"words": content}
    else:
        record = {"words": [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]}
    words = normalize_word_bank(record.get("words"))
    digest = word_bank_hash(words)
    if record.get("sha256") not in (None, digest):
        raise ValueError("Word bank checksum mismatch; the frozen bank has changed.")
    return {**record, "words": words, "sha256": digest, "source": record.get("source", "imported"),
            "loaded_from": str(path.resolve())}


def make_tiny_model(architecture="llama", seed=0):
    """Offline RANDOM test fixture. It has no learned language/meaning behavior."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    torch.manual_seed(seed)
    symbols = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({token: i for i, token in enumerate(symbols)}, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
                                       unk_token="<unk>", pad_token="<pad>", model_max_length=2048)
    if architecture == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=48, intermediate_size=96,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
    elif architecture == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=len(tokenizer), n_embd=48, n_layer=3, n_head=4,
                            n_positions=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config)
    else:
        raise ValueError("architecture must be llama or gpt2")
    model.eval()
    return model, tokenizer


def load_model(args):
    if getattr(args, "backend", "torch") == "mlx":
        return load_mlx_model(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "float16" if device.startswith("cuda") or device == "mps" else "float32"
    dtype = getattr(torch, dtype_name)
    common = {"trust_remote_code": args.trust_remote_code, "local_files_only": args.local_files_only}
    if args.revision:
        common["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, **common)
    kwargs = {**common, "dtype": dtype, "attn_implementation": args.attn_implementation}
    if args.load_in_4bit:
        if not device.startswith("cuda"):
            raise ValueError("--load-in-4bit in this harness requires CUDA and bitsandbytes/accelerate.")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = "auto" if args.device_map == "auto" else {"": device}
    elif args.device_map:
        kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if not args.load_in_4bit and not args.device_map:
        model = model.to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=args.local_files_only)
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("This harness supports decoder-only causal language models.")
    model.eval()
    return model, tokenizer


def prepare_runtime(args, seed):
    if int(seed) < 0:
        raise ValueError("Seeds must be nonnegative.")
    if getattr(args, "backend", "torch") == "mlx":
        if getattr(args, "cpu_threads", None) is not None:
            raise ValueError("--cpu-threads controls PyTorch; omit it for --backend mlx.")
        import mlx.core as mx
        mx.random.seed(int(seed) % (2**32))
    else:
        import torch
        torch.manual_seed(int(seed))
        if getattr(args, "cpu_threads", None) is not None:
            if args.cpu_threads < 1:
                raise ValueError("--cpu-threads must be positive.")
            torch.set_num_threads(args.cpu_threads)


def make_hooks(model, layers="sample", patterns=None, enabled=True):
    recorder = MLXHookRecorder if getattr(model, "_association_backend", None) == "mlx" else HookRecorder
    return recorder(model, layers=layers, patterns=patterns, enabled=enabled)


def model_vocab_size(model):
    if getattr(model, "_association_backend", None) == "mlx":
        return int(model.config.vocab_size)
    return int(model.get_output_embeddings().weight.shape[0])


def runtime_metadata(model, tokenizer, args, hooks, synthetic=False):
    if getattr(model, "_association_backend", None) == "mlx":
        return mlx_runtime_metadata(model, tokenizer, args, hooks, synthetic=synthetic)
    import torch
    packages = {}
    for name in ("torch", "transformers", "numpy", "matplotlib", "tokenizers", "accelerate", "peft", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = model.config.to_dict()
    return {"schema_version": SCHEMA_VERSION, "harness": "sr_association_lab", "backend": "torch", "created_at": now(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running", "config": vars(args), "model": {
            "identifier": "tiny-random-" + args.architecture if synthetic else args.model,
            "synthetic": synthetic, "untrained": synthetic,
            "class": type(model).__name__, "config": config,
            "resolved_commit": getattr(model.config, "_commit_hash", None),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
            "adapter": getattr(args, "adapter", None),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "devices": [str(x) for x in model_devices(model)],
            "hf_device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}},
        "tokenizer": {"identifier": getattr(tokenizer, "name_or_path", None),
            "class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
            "special_ids": tokenizer.all_special_ids, "chat_template": tokenizer.chat_template,
            "resolved_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")},
        "runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda": torch.version.cuda, "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
        "hook_sites": hooks.sites,
        "sampling": "CPU NumPy float64; temperature -> exact top-k -> nucleus top-p; stable ties by token ID; no other processors",
        "alignment": "row r = final prefix position of prompt+generated[:r]; token t post-state is row t+1; last row consumes final token without sampling",
        "timing": "instrumented_forward_ms synchronizes CUDA/MPS and includes hook cloning; host transfer, sampling/statistics, array I/O separate; first row prefill; final consume separate",
        "context": "live retains consumed token IDs and KV across turns; carry re-renders text and prefills; reset starts fresh. No hidden truncation.",
        "plot_semantics": "Per-event units are prompted responses; token IDs are not linguistic words; pooled states use non-special non-whitespace generated positions."}


def run_session(args, *, synthetic=False):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}. Choose a new directory; existing runs are never overwritten.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.chain and not args.protocol.startswith("association"):
        raise ValueError("--chain is for the association protocol.")
    imagination = args.protocol.startswith("imagination")
    context = args.context or ("live" if imagination else "reset")
    if context == "live" and args.no_cache:
        raise ValueError("--context live requires KV caching.")
    if args.prompt_file and context != "reset":
        raise ValueError("--prompt-file requires --context reset.")
    prepare_runtime(args, args.seed)
    model, tokenizer = make_tiny_model(args.architecture, args.seed) if synthetic else load_model(args)
    hooks = make_hooks(model, args.layers, args.hook, enabled=not args.no_hooks)
    cues = list(args.cue or [])
    if args.cues_file:
        cues += [x.strip() for x in Path(args.cues_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not cues:
        cues = ["鏡" if args.protocol == "association" else "誰もいない駅に、ひとつだけ灯りがついている。"]
    max_tokens = args.max_new_tokens if args.max_new_tokens is not None else (24 if args.protocol.startswith("association") else 128)
    options = GenerationOptions(max_new_tokens=max_tokens, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, seed=args.seed, save_logits=not args.no_logits, save_policy_logits=args.save_policy_logits,
        hidden_dtype=args.hidden_dtype, stop_on_newline=args.protocol.startswith("association") and not args.no_stop_newline,
        stop_on_sentence=imagination and not args.no_stop_sentence,
        max_context=args.max_context, top_candidates=args.top_candidates, use_cache=not args.no_cache)
    options.validate()
    manifest = runtime_metadata(model, tokenizer, args, hooks, synthetic)
    manifest["effective_context_policy"] = context
    manifest["generation"] = dataclasses.asdict(options)
    out.mkdir(parents=True)
    atomic_json(out / "manifest.json", manifest)
    vocab_size = model_vocab_size(model)
    atomic_json(out / "vocab.json", [str(tokenizer.convert_ids_to_tokens(i)) for i in range(vocab_size)])
    config_text = json.dumps(model.config_dict if getattr(model, "_association_backend", None) == "mlx"
                             else model.config.to_dict(), ensure_ascii=False, default=str)
    (out / "model_config.json").write_text(config_text + "\n", encoding="utf-8")
    if synthetic:
        print("DEMO: randomly initialized model. Output and plots validate the recorder; they are not language findings.")
    logit_mib = (max_tokens + 1) * vocab_size * 4 / 2**20
    print(f"Run: {out.resolve()} | context={context} | hooks={len(hooks.sites)}")
    print(f"Raw logits allocation/event: {logit_mib:.1f} MiB" if options.save_logits else "Raw logits storage disabled; token statistics remain enabled.")
    print("Timing includes instrumentation. " + ("Live KV persists between turns." if context == "live" else "Each event starts with a fresh prefill."))
    event_id = 0
    quit_requested = False
    completed_events = []
    session = None
    group_id = None
    try:
        for group_id, first_cue in enumerate(cues):
            history = []
            session = LiveSession() if context == "live" else None
            if imagination:
                origins = out / "origins"
                origins.mkdir(exist_ok=True)
                atomic_json(origins / f"{group_id:04d}.json", {"scene": normalize_scene(first_cue),
                    "sha256": scene_hash(first_cue), "source": "provided", "created_at": now(),
                    "language": "en" if args.protocol.endswith("_en") else "ja",
                    "provenance": {"generation_cache_used": False, "fresh_group_session": True}})
            session_rng = np.random.default_rng(args.seed + group_id) if context == "live" else None
            cue = first_cue
            parent_event_id = None
            for step in range(args.steps):
                framing = {}
                if context == "live":
                    messages = build_messages(args.protocol, cue, []) if not session.token_ids else [{"role": "user", "content": cue}]
                    framing = build_live_prompt(tokenizer, args.protocol, cue, session, args.template)
                    prompt_text, prompt_ids, used_template = framing["text"], framing["ids"], framing["wire_format"]
                else:
                    messages = build_messages(args.protocol, cue, history if context == "carry" else [])
                    prompt_text, prompt_ids, used_template = render_prompt(tokenizer, messages, args.template)
                if args.prompt_file:
                    # Exact continuation form for a base model; append cue verbatim, no role wrappers.
                    prefix = Path(args.prompt_file).read_text(encoding="utf-8")
                    prompt_text = prefix + cue
                    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=True)["input_ids"])
                    used_template = "raw-prefix-file"
                    if context == "carry":
                        raise ValueError("--prompt-file requires --context reset; it contains an exact prefix, not chat history.")
                event_options = dataclasses.replace(options, seed=args.seed + (group_id if context == "live" else event_id))
                relative = f"events/{event_id:06d}"
                metadata = {"event_id": event_id, "group_id": group_id, "step": step, "cue": cue,
                    "protocol": args.protocol, "parent_event_id": parent_event_id, "messages": messages,
                    "prompt_text": prompt_text, "template": used_template,
                    "history_context": context, "history_messages": len(history) if context != "reset" else 0,
                    "boundary_dedup_ids": framing.get("boundary_dedup_ids", []),
                    "session_reset_count": session.reset_count if session is not None else None}
                print(f"\n[{event_id}] {cue}", flush=True)
                result = record_generation(model, tokenizer, prompt_ids, out / relative,
                    options=event_options, hooks=hooks, eos_ids=[] if synthetic else None, metadata=metadata,
                    session=session, rng=session_rng)
                event = {k: result[k] for k in ("event_id", "group_id", "step", "cue", "protocol", "parent_event_id",
                    "response", "stop_reason", "censored", "prompt_tokens", "n_tokens", "content_token_indices",
                    "history_context", "event_wall_ms", "prefill_forward_ms", "decode_forward_ms_mean")}
                event.update(directory=relative, generated_tokens=result["n_tokens"], status="complete",
                             past_tokens=result["past_tokens"], new_prompt_tokens=result["new_prompt_tokens"])
                if imagination:
                    event["sentence_check"] = check_one_sentence(result["response"], result["stop_reason"])
                    event["origin_sha256"] = scene_hash(first_cue)
                completed_events.append(event)
                append_jsonl(out / "events.jsonl", event)
                print(result["response"])
                print(f"  {result['n_tokens']} tokens | stop={result['stop_reason']} | {result['event_wall_ms']:.0f} ms")
                history = (history if context == "live" else []) + messages + [{"role": "assistant", "content": result["response"]}]
                parent_event_id = event_id if context in ("carry", "live") or args.chain else None
                if session is not None:
                    save_session(out / "sessions" / f"{group_id:04d}.json", session)
                event_id += 1
                continuation = CONTINUE_CUE_EN if args.protocol.endswith("_en") else CONTINUE_CUE
                next_cue = result["unit_text"] if args.chain else continuation if imagination and context in ("carry", "live") else first_cue
                if args.chain and not next_cue:
                    print("Empty association: ending this chain.")
                    break
                if args.interactive and step + 1 < args.steps:
                    while True:
                        try:
                            answer = input("Next cue [Enter=continue, /quit, /reset, /note TEXT, /plot]: ")
                        except EOFError:
                            quit_requested = True
                            break
                        if answer == "/quit":
                            quit_requested = True
                            break
                        if answer == "/reset":
                            history = []
                            if session is not None:
                                session.reset()
                                save_session(out / "sessions" / f"{group_id:04d}.json", session)
                            parent_event_id = None
                            next_cue = first_cue
                            print("History and live KV reset. Enter the next cue.")
                            continue
                        if answer.startswith("/note "):
                            with (out / "notes.jsonl").open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps({"after_event_id": event_id-1, "time": now(), "text": answer[6:]}, ensure_ascii=False) + "\n")
                            continue
                        if answer == "/plot":
                            for path in plot_run(out):
                                print(path)
                            continue
                        next_cue = answer if answer else next_cue
                        break
                if quit_requested:
                    break
                cue = next_cue
            if quit_requested:
                break
        manifest.update(status="complete", finished_at=now(), completed_events=event_id,
                        stopped_by_user=quit_requested)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                        error_type=type(exc).__name__, error=str(exc), completed_events=event_id, finished_at=now())
        raise
    finally:
        hooks.close()
        if session is not None and group_id is not None:
            save_session(out / "sessions" / f"{group_id:04d}.json", session)
        atomic_jsonl(out / "events.jsonl", completed_events)
        atomic_json(out / "manifest.json", manifest)
        write_transcript(out)
    if not args.no_plot:
        for path in plot_run(out):
            print(f"Plot: {path}")
    return out


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_jsonl(path, records):
    """Checkpoint committed in-memory rows without relying on append visibility."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def generation_options(args, *, seed=None) -> GenerationOptions:
    return GenerationOptions(max_new_tokens=24 if args.max_new_tokens is None else args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        seed=args.seed if seed is None else seed, save_logits=not args.no_logits,
        save_policy_logits=args.save_policy_logits, hidden_dtype=args.hidden_dtype,
        stop_on_newline=not args.no_stop_newline, max_context=args.max_context,
        top_candidates=args.top_candidates, use_cache=not args.no_cache)


def generate_word_bank(model, tokenizer, args, out) -> dict:
    """Generate once (bounded logged retries), then discard every bank cache."""
    out = Path(out)
    if args.bank_size < 1 or args.bank_attempts < 1 or args.bank_seed < 0:
        raise ValueError("bank-size/attempts must be positive and bank-seed nonnegative.")
    if args.bank_language == "ja":
        prompt = (f"連想実験のための刺激語を{args.bank_size}語作ってください。"
                  "互いに異なる日本語の単語にし、各項目は一語にしてください。"
                  "自然、物体、場所、生き物、感覚などから自由に選んでください。"
                  "単語を文字列にしたJSON配列だけを出力してください。説明やコードフェンスは不要です。")
    else:
        prompt = (f"Generate exactly {args.bank_size} different English stimulus words for a word-association experiment. "
                  "Each item must be a single word. Choose freely from nature, objects, places, animals, and sensations. "
                  "Return ONLY a JSON array of strings. No explanation or code fence.")
    if args.bank_prompt:
        prompt = Path(args.bank_prompt).read_text(encoding="utf-8")
    prompt_text, prompt_ids, used_template = render_prompt(tokenizer, [{"role": "user", "content": prompt}], args.template)
    attempts = []
    hooks = make_hooks(model, enabled=False)
    try:
        for attempt in range(args.bank_attempts):
            seed = args.bank_seed + attempt
            options = GenerationOptions(max_new_tokens=args.bank_max_new_tokens,
                temperature=args.bank_temperature, top_p=args.bank_top_p, seed=seed,
                save_logits=False, stop_on_newline=False, max_context=args.max_context)
            session = LiveSession()
            relative = f"bank_generation/attempt_{attempt:02d}"
            print(f"Generating bank: attempt {attempt + 1}/{args.bank_attempts}, seed={seed}", flush=True)
            try:
                meta = record_generation(model, tokenizer, prompt_ids, out / relative,
                    options=options, hooks=hooks, session=session,
                    metadata={"role": "bank_generation", "attempt": attempt, "bank_seed": seed,
                              "prompt_text": prompt_text, "template": used_template})
            finally:
                # Release logical state. torch.cuda.empty_cache() is unnecessary:
                # allocator memory is not a model's conversation state.
                session.reset()
            parsed = parse_word_bank(meta["response"], args.bank_size)
            entry = {"attempt": attempt, "seed": seed, "directory": relative,
                     "stop_reason": meta["stop_reason"], "censored": meta["censored"],
                     "cache_discarded": True, "parse": parsed}
            attempts.append(entry)
            atomic_json(out / "bank_generation" / "attempts.json", attempts)
            if parsed["accepted"]:
                bank = {"schema_version": SCHEMA_VERSION, "words": parsed["words"],
                    "sha256": word_bank_hash(parsed["words"]), "source": "model-generated",
                    "language": args.bank_language, "created_at": now(),
                    "provenance": {"model": args.model, "revision": args.revision,
                        "resolved_commit": getattr(model.config, "_commit_hash", None),
                        "prompt_text": prompt_text, "template": used_template,
                        "bank_seed": args.bank_seed, "accepted_attempt": attempt,
                        "attempts": attempts, "cache_discarded_before_trials": True}}
                atomic_json(out / "bank.json", bank)
                return bank
        raise ValueError(f"Could not obtain {args.bank_size} distinct single-word items after {args.bank_attempts} attempts. Inspect bank_generation/attempts.json; no filler was invented.")
    finally:
        hooks.close()


def run_bank(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    prepare_runtime(args, args.bank_seed)
    model, tokenizer = load_model(args)
    hooks = make_hooks(model, enabled=False)
    manifest = runtime_metadata(model, tokenizer, args, hooks)
    manifest["kind"] = "word_bank_generation"
    out.mkdir(parents=True)
    atomic_json(out / "manifest.json", manifest)
    try:
        bank = generate_word_bank(model, tokenizer, args, out)
        manifest.update(status="complete", finished_at=now(), bank_sha256=bank["sha256"])
        print(f"Bank: {out / 'bank.json'} ({len(bank['words'])} words)")
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                        error_type=type(exc).__name__, error=str(exc), finished_at=now())
        raise
    finally:
        hooks.close()
        atomic_json(out / "manifest.json", manifest)
    return out


def parse_seeds(text, *, name="seeds") -> list[int]:
    try:
        values = [int(s.strip()) for s in text.split(",")]
    except (TypeError, ValueError):
        raise ValueError(f"--{name} must be comma-separated integers.") from None
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise ValueError(f"--{name} must contain unique nonnegative integers.")
    return values


def run_experiment(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}. Choose a new experiment directory.")
    if args.no_cache:
        raise ValueError("Bank experiments use KV sessions; --no-cache is available only in run reset/carry.")
    if args.n_seeds is not None and args.seeds is not None:
        raise ValueError("Choose either --seeds or --n-seeds.")
    if args.n_seeds is not None and args.n_seeds < 1:
        raise ValueError("--n-seeds must be positive.")
    if args.similarity == "sentence-transformer" and not args.embedding_model:
        raise ValueError("--similarity sentence-transformer requires --embedding-model.")
    options = generation_options(args)
    options.validate()
    seeds = parse_seeds(args.seeds) if args.seeds is not None else list(range(args.seed, args.seed + (args.n_seeds or 3)))
    order_seeds = parse_seeds(args.order_seeds, name="order-seeds")
    conditions = ("live", "reset") if args.condition == "both" else (args.condition,)
    # Validate randomization before model load, even when the bank is generated.
    build_trial_plan(["validation"], seeds, args.repeats, args.order, order_seeds, conditions)
    bank = load_word_bank(args.bank) if args.bank else None
    requested_language = args.bank_language
    args.bank_language = args.bank_language or (bank.get("language") if bank else None) or "ja"
    if args.bank_language not in ("ja", "en"):
        raise ValueError("Saved bank language is unsupported; choose --bank-language ja or en explicitly.")
    prepare_runtime(args, args.seed)
    model, tokenizer = load_model(args)
    no_hooks = make_hooks(model, enabled=False)
    manifest = runtime_metadata(model, tokenizer, args, no_hooks)
    manifest.update(kind="bank_experiment", completed_trials=0, completed_responses=0,
        requested_bank_language=requested_language, effective_association_language=args.bank_language,
        randomness={"bank": "separate NumPy RNG; bank_seed + explicit retry index",
            "order": "separate NumPy RNG per order_seed; same permutation across response seeds/conditions/repeats",
            "response": "one continuous NumPy RNG per trial, initialized from sample_seed",
            "repeats": "same seed repeats measure repeatability/timing; they are not independent random seeds"},
        timing_definitions={"first_token_model_ms": "first instrumented predictor forward, including appended prompt",
            "first_token_latency_ms": "event start to first sample, before that row's array write",
            "response_model_ms": "sum of predictor forwards; excludes terminal consume and host I/O",
            "response_wall_ms": "event start to final sample; includes earlier instrumentation/I/O",
            "terminal_consume_ms": "extra forward consuming the final sampled token",
            "event_wall_ms": "instrumented generation through terminal row write; excludes outer event-index I/O"},
        stimulus_visibility="Only the current cue enters each new turn. Bank generation caches are discarded; future bank words are not provided as a list.",
        warmup="No dedicated warm-up. Model-generated bank attempts precede trials; imported-bank runs may include cold-start effects in the first response.",
        one_word_check="Conservative single-span Unicode surface heuristic; not a Japanese morphological segmenter. Length-capped responses are ineligible.")
    out.mkdir(parents=True)
    (out / "responses.jsonl").touch()
    atomic_json(out / "manifest.json", manifest)
    hooks = None
    session = None
    completed_responses = []
    try:
        if bank is None:
            bank = generate_word_bank(model, tokenizer, args, out)
        else:
            atomic_json(out / "bank.json", bank)
        manifest["bank_sha256"] = bank["sha256"]
        schedule = build_trial_plan(bank["words"], seeds, args.repeats, args.order, order_seeds, conditions)
        atomic_json(out / "schedule.json", schedule)
        hooks = make_hooks(model, args.layers, args.hook, enabled=not args.no_hooks)
        manifest["hook_sites"] = hooks.sites
        vocab_size = model_vocab_size(model)
        atomic_json(out / "vocab.json", [str(tokenizer.convert_ids_to_tokens(i)) for i in range(vocab_size)])
        atomic_json(out / "manifest.json", manifest)
        print(f"Experiment: {len(bank['words'])} words x {len(schedule)} trials | bank={bank['sha256'][:12]}")
        protocol = "association_en" if args.bank_language == "en" else "association"
        for trial in schedule:
            trial_dir = out / "trials" / trial["trial_id"]
            trial_dir.mkdir(parents=True)
            trial_manifest = {**manifest, "kind": "bank_trial", "trial": trial,
                              "status": "running", "completed_events": 0}
            atomic_json(trial_dir / "manifest.json", trial_manifest)
            session = LiveSession()
            rng = np.random.default_rng(trial["sample_seed"])
            event_options = dataclasses.replace(options, seed=trial["sample_seed"])
            previous_event_id = None
            trial_events = []
            try:
                for position, bank_index in enumerate(trial["bank_order"]):
                    if trial["condition"] == "reset":
                        session.reset()
                    cue = bank["words"][bank_index]
                    framing = build_live_prompt(tokenizer, protocol, cue, session, args.template)
                    directory = f"events/{position:06d}"
                    metadata = {**{k: v for k, v in trial.items() if k != "bank_order"},
                        "event_id": position, "group_id": 0, "step": position, "position": position,
                        "bank_index": bank_index, "bank_sha256": bank["sha256"], "cue": cue,
                        "protocol": "association", "history_context": trial["condition"],
                        "parent_event_id": previous_event_id if trial["condition"] == "live" else None,
                        "prompt_text": framing["text"], "template": framing["wire_format"],
                        "boundary_dedup_ids": framing["boundary_dedup_ids"],
                        "session_reset_count": session.reset_count}
                    result = record_generation(model, tokenizer, framing["ids"], trial_dir / directory,
                        options=event_options, hooks=hooks, session=session, rng=rng, metadata=metadata)
                    tokens = read_jsonl(trial_dir / directory / "tokens.jsonl")
                    check = check_one_word(result["response"], result["stop_reason"])
                    summary_keys = ("trial_id", "event_id", "sample_seed", "repeat_index", "order_seed", "condition",
                        "position", "bank_index", "cue", "response", "stop_reason", "censored", "past_tokens",
                        "new_prompt_tokens", "first_token_model_ms", "first_token_latency_ms", "response_model_ms",
                        "response_wall_ms", "terminal_consume_ms", "event_wall_ms")
                    summary = {k: result[k] for k in summary_keys}
                    summary.update(generated_tokens=result["n_tokens"],
                        eligible=check["eligible"], surface_ok=check["surface_ok"], reasons=check["reasons"],
                        raw_entropy_mean=float(np.mean([t["raw_entropy_nats"] for t in tokens])),
                        raw_surprisal_mean=float(np.mean([t["raw_surprisal_nats"] for t in tokens])),
                        event_directory=str((trial_dir / directory).relative_to(out)))
                    event = {**summary, "directory": directory, "status": "complete", "protocol": "association",
                        "history_context": trial["condition"], "parent_event_id": metadata["parent_event_id"],
                        "prompt_tokens": result["prompt_tokens"], "n_tokens": result["n_tokens"],
                        "content_token_indices": result["content_token_indices"]}
                    trial_events.append(event)
                    completed_responses.append(summary)
                    append_jsonl(trial_dir / "events.jsonl", event)
                    append_jsonl(out / "responses.jsonl", summary)
                    previous_event_id = position
                    trial_manifest["completed_events"] = position + 1
                    manifest["completed_responses"] += 1
                    print(f"[{trial['trial_id']} {position:02d} B{bank_index} seed={trial['sample_seed']} {trial['condition']}] "
                          f"{cue} -> {result['response'].strip()!r} | {result['first_token_model_ms']:.1f} ms first | "
                          f"{'eligible' if check['eligible'] else ','.join(check['reasons'])}", flush=True)
                trial_manifest.update(status="complete", finished_at=now())
                manifest["completed_trials"] += 1
            except BaseException as exc:
                trial_manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                    error_type=type(exc).__name__, error=str(exc), finished_at=now())
                raise
            finally:
                atomic_jsonl(trial_dir / "events.jsonl", trial_events)
                trial_manifest["completed_events"] = len(trial_events)
                manifest["completed_responses"] = len(completed_responses)
                save_session(trial_dir / "session.json", session)
                session.reset()
                atomic_json(trial_dir / "manifest.json", trial_manifest)
                write_transcript(trial_dir)
                atomic_json(out / "manifest.json", manifest)
        manifest.update(status="generation_complete", generation_finished_at=now())
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                        error_type=type(exc).__name__, error=str(exc), finished_at=now())
        raise
    finally:
        if hooks is not None:
            hooks.close()
        no_hooks.close()
        if session is not None:
            session.reset()
        atomic_jsonl(out / "responses.jsonl", completed_responses)
        atomic_json(out / "manifest.json", manifest)
        write_experiment_transcript(out)
    try:
        analyze_experiment(out, args, model=model, tokenizer=tokenizer)
        manifest.update(status="complete", finished_at=now())
        atomic_json(out / "manifest.json", manifest)
        if not args.no_plot:
            for path in plot_experiment(out):
                print(f"Plot: {path}")
        manifest.update(status="complete", finished_at=now())
    except BaseException as exc:
        manifest.update(status="analysis_interrupted" if isinstance(exc, KeyboardInterrupt) else "analysis_error",
                        error_type=type(exc).__name__, error=str(exc), finished_at=now())
        raise
    finally:
        atomic_json(out / "manifest.json", manifest)
    return out


def normalized_vectors(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.full_like(values, np.nan), where=np.isfinite(norms) & (norms > 0))


def normalize_scene(scene):
    if not isinstance(scene, str):
        raise ValueError("An origin scene must be a string.")
    scene = unicodedata.normalize("NFC", scene).strip()
    if not scene or not any(unicodedata.category(c)[0] in "LN" for c in scene):
        raise ValueError("An origin scene must contain nonempty text.")
    return scene


def scene_hash(scene):
    return hashlib.sha256(normalize_scene(scene).encode("utf-8")).hexdigest()


def load_origin_scene(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        origin = json.loads(text)
        if not isinstance(origin, dict) or not isinstance(origin.get("scene"), str):
            raise ValueError("origin.json must contain a scene string.")
        origin = copy.deepcopy(origin)
        origin["scene"] = normalize_scene(origin["scene"])
        expected = scene_hash(origin["scene"])
        if origin.get("sha256") is not None and origin["sha256"] != expected:
            raise ValueError("Saved origin hash does not match its scene; do not silently alter a frozen origin.")
        origin["sha256"] = expected
        origin.setdefault("source", "imported")
        origin["loaded_from"] = str(path)
        return origin
    scene = normalize_scene(text)
    return {"schema_version": SCHEMA_VERSION, "scene": scene, "sha256": scene_hash(scene),
            "source": "imported-text", "loaded_from": str(path), "created_at": now()}


def generate_origin_scene(model, tokenizer, args, out):
    """Freeze one sentence once; no origin-generation cache enters a trial."""
    if args.origin_seed < 0 or args.origin_attempts < 1 or args.origin_max_new_tokens < 1:
        raise ValueError("origin-seed must be nonnegative; origin-attempts/max-new-tokens must be positive.")
    prompt = ("連想情景を展開する起点となる、具体的な情景を一文だけ書いてください。"
              "物、場所、光、動き、音などを自由に選び、句点（。）で終えてください。"
              "説明、分析、見出し、箇条書き、複数の文は不要です。") if args.language == "ja" else (
              "Write exactly one sentence depicting a concrete starting scene. "
              "Choose objects, a place, light, movement, or sounds freely. "
              "End with sentence punctuation. Return only the sentence, without a heading or explanation.")
    if args.origin_prompt:
        prompt = Path(args.origin_prompt).read_text(encoding="utf-8")
    prompt_text, prompt_ids, used_template = render_prompt(tokenizer, [{"role": "user", "content": prompt}], args.template)
    out = Path(out)
    attempts = []
    hooks = make_hooks(model, enabled=False)
    try:
        for attempt in range(args.origin_attempts):
            seed = args.origin_seed + attempt
            options = GenerationOptions(max_new_tokens=args.origin_max_new_tokens,
                temperature=args.origin_temperature, top_p=args.origin_top_p, seed=seed,
                save_logits=False, stop_on_sentence=True, max_context=args.max_context)
            session = LiveSession()
            relative = f"origin_generation/attempt_{attempt:02d}"
            print(f"Generating origin: attempt {attempt + 1}/{args.origin_attempts}, seed={seed}", flush=True)
            try:
                result = record_generation(model, tokenizer, prompt_ids, out / relative,
                    options=options, hooks=hooks, session=session,
                    metadata={"role": "origin_generation", "attempt": attempt, "origin_seed": seed,
                              "prompt_text": prompt_text, "template": used_template})
            finally:
                session.reset()
            check = check_one_sentence(result["response"], result["stop_reason"])
            attempts.append({"attempt": attempt, "seed": seed, "directory": relative,
                "stop_reason": result["stop_reason"], "censored": result["censored"],
                "cache_discarded": True, "sentence_check": check})
            atomic_json(out / "origin_generation" / "attempts.json", attempts)
            if check["eligible"]:
                scene = normalize_scene(result["response"])
                origin = {"schema_version": SCHEMA_VERSION, "scene": scene, "sha256": scene_hash(scene),
                    "source": "model-generated", "language": args.language, "created_at": now(),
                    "sentence_check": check, "provenance": {"model": args.model, "revision": args.revision,
                        "backend": getattr(args, "backend", "torch"),
                        "resolved_commit": getattr(model.config, "_commit_hash", None),
                        "prompt_text": prompt_text, "template": used_template,
                        "origin_seed": args.origin_seed, "accepted_attempt": attempt, "attempts": attempts,
                        "cache_discarded_before_trials": True}}
                atomic_json(out / "origin.json", origin)
                return origin
        raise ValueError(f"Could not obtain a single-sentence origin after {args.origin_attempts} attempts. "
                         "Inspect origin_generation/attempts.json or provide --scene / --scene-file; no sentence was repaired.")
    finally:
        hooks.close()


def write_imagination_transcript(out):
    out = Path(out)
    origin_path = out / "origin.json"
    origin = json.loads(origin_path.read_text(encoding="utf-8")) if origin_path.exists() else {}
    lines = ["# Sentence-scene experiment", "", "## Frozen origin", "", origin.get("scene", "(not accepted)"), "",
             "Live: cumulative KV. Reset: independent restart from the same origin at every step.", "",
             "Responses below are complete decoded samples, including surface failures and capped outputs.", ""]
    for i, row in enumerate(read_jsonl(out / "responses.jsonl")):
        lines += [f"## R{i} · {row['trial_id']} · S{row['step_index']} · {row['condition']} · seed {row['sample_seed']}",
                  "", "```text", row["response"], "```", "",
                  f"Eligible: {row['eligible']}; stop: {row['stop_reason']}; "
                  f"past/new/output tokens: {row['past_tokens']}/{row['new_prompt_tokens']}/{row['n_tokens']}", ""]
    (out / "transcript.md").write_text("\n".join(lines), encoding="utf-8")


def run_imagination(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}. Choose a new experiment directory.")
    if args.steps < 1 or args.repeats < 1:
        raise ValueError("--steps and --repeats must be positive.")
    if args.no_cache:
        raise ValueError("imagine uses actual KV sessions; --no-cache is available in run reset/carry.")
    if args.seeds is not None and args.n_seeds is not None:
        raise ValueError("Choose either --seeds or --n-seeds.")
    if args.n_seeds is not None and args.n_seeds < 1:
        raise ValueError("--n-seeds must be positive.")
    if args.similarity == "sentence-transformer" and not args.embedding_model:
        raise ValueError("--similarity sentence-transformer requires --embedding-model.")
    if args.origin_prompt and (args.scene is not None or args.scene_file):
        raise ValueError("--origin-prompt is only used when the origin is model-generated.")
    seeds = parse_seeds(args.seeds) if args.seeds is not None else list(range(args.seed, args.seed + (args.n_seeds or 3)))
    conditions = ("live", "reset") if args.condition == "both" else (args.condition,)
    schedule = build_trial_plan(["origin"], seeds, args.repeats, "fixed", (101,), conditions)
    for trial in schedule:
        trial.pop("bank_order", None)
        trial.pop("order_seed", None)
        trial["steps"] = args.steps
    origin = load_origin_scene(args.scene_file) if args.scene_file else None
    if args.scene is not None:
        scene = normalize_scene(args.scene)
        origin = {"schema_version": SCHEMA_VERSION, "scene": scene, "sha256": scene_hash(scene),
                  "source": "provided", "created_at": now(),
                  "provenance": {"generation_cache_used": False}}
    requested_language = args.language
    args.language = args.language or (origin.get("language") if origin else None) or "ja"
    if args.language not in ("ja", "en"):
        raise ValueError("Saved origin language is unsupported; choose --language ja or en.")
    options = dataclasses.replace(generation_options(args), stop_on_newline=False, stop_on_sentence=True)
    options.validate()
    prepare_runtime(args, args.seed)
    model, tokenizer = load_model(args)
    no_hooks = make_hooks(model, enabled=False)
    manifest = runtime_metadata(model, tokenizer, args, no_hooks)
    manifest.update(kind="imagination_experiment", completed_trials=0, completed_responses=0,
        requested_language=requested_language, effective_language=args.language,
        origin_cache_policy="Discard all origin-generation KV; each trial begins with fresh KV and the frozen scene text only.",
        condition_semantics={"live": "Origin on first turn, fixed continuation cue thereafter; preserve exact token IDs and cumulative KV.",
                             "reset": "Fresh KV and the SAME frozen origin at EVERY step; independent origin restarts, not a scene chain."},
        unit_policy="One punctuation-terminated sentence per turn; heuristic boundary with no token/text trimming. "
                    "All sampled tokens remain in the live state, including failed/overshooting/capped turns; no retries within a trial.",
        randomness={"origin": "separate RNG: origin_seed + explicit retry index; accepted scene frozen once",
                    "response": "one continuous NumPy RNG per trial, initialized from sample_seed; no per-sentence reseeding",
                    "repeats": "same seed repeats, not additional independent seeds"},
        warmup="No dedicated warm-up. Generated origins precede trials; supplied origins may give a cold first trial.",
        cache_serialization="Only ledger/status are saved; KV is live in process and is not restored on restart.")
    out.mkdir(parents=True)
    (out / "responses.jsonl").touch()
    atomic_json(out / "manifest.json", manifest)
    hooks = None
    session = None
    completed = []
    try:
        if origin is None:
            origin = generate_origin_scene(model, tokenizer, args, out)
        else:
            origin.setdefault("language", args.language)
            atomic_json(out / "origin.json", origin)
        manifest["origin_sha256"] = origin["sha256"]
        atomic_json(out / "schedule.json", schedule)
        hooks = make_hooks(model, args.layers, args.hook, enabled=not args.no_hooks)
        manifest["hook_sites"] = hooks.sites
        atomic_json(out / "vocab.json", [str(tokenizer.convert_ids_to_tokens(i)) for i in range(model_vocab_size(model))])
        atomic_json(out / "manifest.json", manifest)
        protocol = "imagination_en" if args.language == "en" else "imagination"
        continuation = CONTINUE_CUE_EN if args.language == "en" else CONTINUE_CUE
        print(f"Imagination: {args.steps} sentences x {len(schedule)} trials | origin={origin['sha256'][:12]}")
        print(f"Origin: {origin['scene']}", flush=True)
        for trial in schedule:
            trial_dir = out / "trials" / trial["trial_id"]
            trial_dir.mkdir(parents=True)
            trial_manifest = {**manifest, "kind": "imagination_trial", "trial": trial,
                              "status": "running", "completed_events": 0}
            atomic_json(trial_dir / "manifest.json", trial_manifest)
            session = LiveSession()
            rng = np.random.default_rng(trial["sample_seed"])
            event_options = dataclasses.replace(options, seed=trial["sample_seed"])
            trial_events = []
            try:
                for step in range(args.steps):
                    if trial["condition"] == "reset":
                        session.reset()
                    cue = origin["scene"] if step == 0 or trial["condition"] == "reset" else continuation
                    framing = build_live_prompt(tokenizer, protocol, cue, session, args.template)
                    directory = f"events/{step:06d}"
                    metadata = {**trial, "event_id": step, "step_index": step, "step": step, "group_id": 0,
                        "cue": cue, "origin_sha256": origin["sha256"], "protocol": "imagination",
                        "history_context": trial["condition"],
                        "parent_event_id": step - 1 if step and trial["condition"] == "live" else None,
                        "prompt_text": framing["text"], "template": framing["wire_format"],
                        "boundary_dedup_ids": framing["boundary_dedup_ids"], "session_reset_count": session.reset_count}
                    result = record_generation(model, tokenizer, framing["ids"], trial_dir / directory,
                        options=event_options, hooks=hooks, session=session, rng=rng, metadata=metadata)
                    check = check_one_sentence(result["response"], result["stop_reason"])
                    keys = ("trial_id", "event_id", "step_index", "sample_seed", "repeat_index", "condition",
                        "cue", "origin_sha256", "response", "unit_text", "stop_reason", "censored", "n_tokens",
                        "prompt_tokens", "past_tokens", "new_prompt_tokens", "cache_reused",
                        "first_token_model_ms", "first_token_latency_ms", "response_model_ms", "response_wall_ms",
                        "terminal_consume_ms", "event_wall_ms")
                    row = {k: result[k] for k in keys}
                    token_rows = read_jsonl(trial_dir / directory / "tokens.jsonl")
                    row.update(sentence_check=check, eligible=check["eligible"], surface_ok=check["surface_ok"],
                        reasons=check["reasons"], generated_tokens=result["n_tokens"],
                        event_directory=str((trial_dir / directory).relative_to(out)),
                        raw_entropy_mean=float(np.mean([t["raw_entropy_nats"] for t in token_rows])),
                        raw_surprisal_mean=float(np.mean([t["raw_surprisal_nats"] for t in token_rows])))
                    event = {**row, "directory": directory, "status": "complete", "protocol": "imagination",
                        "history_context": trial["condition"], "parent_event_id": metadata["parent_event_id"],
                        "content_token_indices": result["content_token_indices"]}
                    trial_events.append(event)
                    completed.append(row)
                    append_jsonl(trial_dir / "events.jsonl", event)
                    append_jsonl(out / "responses.jsonl", row)
                    print(f"[{trial['trial_id']} S{step:02d} seed={trial['sample_seed']} {trial['condition']}] "
                          f"{result['response'].strip()!r} | {'eligible' if check['eligible'] else ','.join(check['reasons'])}", flush=True)
                trial_manifest.update(status="complete", finished_at=now())
                manifest["completed_trials"] += 1
            except BaseException as exc:
                trial_manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                    error_type=type(exc).__name__, error=str(exc), finished_at=now())
                raise
            finally:
                atomic_jsonl(trial_dir / "events.jsonl", trial_events)
                save_session(trial_dir / "session.json", session)
                session.reset()
                trial_manifest["completed_events"] = len(trial_events)
                manifest["completed_responses"] = len(completed)
                atomic_json(trial_dir / "manifest.json", trial_manifest)
                write_transcript(trial_dir)
                atomic_json(out / "manifest.json", manifest)
        manifest.update(status="generation_complete", generation_finished_at=now())
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                        error_type=type(exc).__name__, error=str(exc), finished_at=now())
        raise
    finally:
        if hooks is not None:
            hooks.close()
        no_hooks.close()
        if session is not None:
            session.reset()
        atomic_jsonl(out / "responses.jsonl", completed)
        atomic_json(out / "manifest.json", manifest)
        write_imagination_transcript(out)
    try:
        analyze_imagination(out, args, model=model, tokenizer=tokenizer)
        manifest.update(status="complete", finished_at=now())
        atomic_json(out / "manifest.json", manifest)
        if not args.no_plot:
            for path in plot_imagination(out):
                print(f"Plot: {path}")
        manifest.update(status="complete", finished_at=now())
    except BaseException as exc:
        manifest.update(status="analysis_interrupted" if isinstance(exc, KeyboardInterrupt) else "analysis_error",
                        error_type=type(exc).__name__, error=str(exc), finished_at=now())
        raise
    finally:
        atomic_json(out / "manifest.json", manifest)
    return out


def similarity_rows(responses, bank_vectors, response_vectors, schedule):
    """All-bank cosine + deterministic ranks and within-trial comparisons."""
    bank_vectors, response_vectors = normalized_vectors(bank_vectors), normalized_vectors(response_vectors)
    matrix = np.clip(response_vectors @ bank_vectors.T, -1, 1).astype(np.float32)
    orders = {t["trial_id"]: t["bank_order"] for t in schedule}
    previous = {}
    result = []
    for i, row in enumerate(responses):
        values = matrix[i]
        cue = row["bank_index"]
        entry = {**row, **dict.fromkeys(("cue_cosine", "nearest_bank_index", "nearest_bank_cosine",
            "nearest_other_bank_index", "nearest_other_cosine", "cue_rank", "mean_bank_cosine",
            "previous_response_cosine", "previously_seen_max_cosine", "future_bank_max_cosine"))}
        finite = np.isfinite(values)
        if finite.all():
            best = int(np.argmax(values))
            others = [j for j in range(len(values)) if j != cue]
            entry.update(cue_cosine=float(values[cue]), nearest_bank_index=best,
                nearest_bank_cosine=float(values[best]), cue_rank=int(np.count_nonzero(values > values[cue])) + 1,
                mean_bank_cosine=float(values.mean()))
            if others:
                other = others[int(np.argmax(values[others]))]
                entry.update(nearest_other_bank_index=other, nearest_other_cosine=float(values[other]))
            order = orders[row["trial_id"]]
            seen, future = order[:row["position"]], order[row["position"] + 1:]
            entry["previously_seen_max_cosine"] = float(values[seen].max()) if seen else None
            entry["future_bank_max_cosine"] = float(values[future].max()) if future else None
        if row["trial_id"] in previous:
            j = previous[row["trial_id"]]
            value = float(np.dot(response_vectors[i], response_vectors[j]))
            entry["previous_response_cosine"] = float(np.clip(value, -1, 1)) if math.isfinite(value) else None
        previous[row["trial_id"]] = i
        result.append(entry)
    return result, matrix


def summarize_cues(responses):
    groups = {}
    for row in responses:
        groups.setdefault((row["bank_index"], row["condition"]), []).append(row)
    summaries = []
    for (bank_index, condition), rows in sorted(groups.items()):
        eligible = [r for r in rows if r["eligible"]]
        summary = {"bank_index": bank_index, "cue": rows[0]["cue"], "condition": condition,
            "n": len(rows), "eligible_n": len(eligible),
            "unique_responses": sorted({r["response"].strip() for r in eligible}),
            "independent_seed_count": len({r["sample_seed"] for r in rows}),
            "distinct_seed_count": len({r["sample_seed"] for r in rows}),
            "eligible_seed_count": len({r["sample_seed"] for r in eligible}),
            "sample_seeds": sorted({r["sample_seed"] for r in rows}),
            "repeat_indices": sorted({r["repeat_index"] for r in rows}),
            "order_seeds": sorted({r["order_seed"] for r in rows if r["order_seed"] is not None}),
            "aggregation": "eligible responses only; descriptive mean/population std, no inferential independence claim"}
        for key in ("first_token_model_ms", "first_token_latency_ms", "response_model_ms", "response_wall_ms", "cue_cosine"):
            values = [r[key] for r in eligible if r.get(key) is not None]
            summary[key + "_mean"] = float(np.mean(values)) if values else None
            summary[key + "_std"] = float(np.std(values)) if values else None
            summary[key + "_n"] = len(values)
        summaries.append(summary)
    return summaries


def encode_similarity_texts(texts, args, *, model=None, tokenizer=None, destination=None):
    """Shared post-generation encoder for banks and scenes; never accesses live state."""
    metadata = {}
    if args.similarity == "self":
        if model is None or tokenizer is None:
            model, tokenizer = load_model(args)
        model.eval()
        encoder_metadata = {"backend": "torch"}
        if getattr(model, "_association_backend", None) == "mlx":
            embeddings, token_ids, encoder_metadata = mlx_encode_isolated(model, tokenizer, texts, args.max_context)
        else:
            import torch
            device = input_device(model)
            params = inspect.signature(model.forward).parameters
            embeddings, token_ids = [], []
            model_limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", args.max_context)
            with torch.inference_mode():
                for text in texts:
                    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
                    if not ids or len(ids) > min(args.max_context, int(model_limit or args.max_context)):
                        raise ValueError("Similarity text is empty after tokenization or exceeds context; no silent truncation.")
                    token_ids.append(ids)
                    kwargs = {"input_ids": torch.tensor([ids], device=device),
                        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
                        "use_cache": False, "output_hidden_states": True, "return_dict": True}
                    if "logits_to_keep" in params:
                        kwargs["logits_to_keep"] = 1
                    elif "num_logits_to_keep" in params:
                        kwargs["num_logits_to_keep"] = 1
                    output = model(**kwargs)
                    if not getattr(output, "hidden_states", None):
                        raise ValueError("Model does not expose hidden_states for --similarity self.")
                    embeddings.append(output.hidden_states[-1][0].float().mean(dim=0).cpu().numpy())
        vectors = normalized_vectors(np.stack(embeddings))
        metadata.update(label="isolated-LM representation cosine (proxy)", model=args.model,
            revision=args.revision, resolved_commit=getattr(model.config, "_commit_hash", None),
            pooling="mean of final hidden_states over raw-text tokens; no special tokens or chat template",
            normalization="unit L2 norm; zero/nonfinite vector becomes missing",
            encoder_input_ids=token_ids, generation_history_used=False)
        metadata.update(encoder_metadata)
    elif args.similarity == "sentence-transformer":
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import is_sentence_transformer_model
        if not args.embedding_model:
            raise ValueError("--embedding-model is required for sentence-transformer similarity.")
        if not is_sentence_transformer_model(args.embedding_model,
                revision=args.embedding_revision, local_files_only=args.local_files_only):
            raise ValueError("--embedding-model must have a saved SentenceTransformer modules.json. "
                             "Automatic pooling over a plain LM is not used; choose --similarity self for that proxy.")
        encoder = SentenceTransformer(args.embedding_model, device=args.embedding_device,
            revision=args.embedding_revision, local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code)
        vectors = normalized_vectors(encoder.encode(texts, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True))
        modules = {name: type(module).__name__ for name, module in encoder.named_children()}
        metadata.update(label="sentence-encoder cosine", model=args.embedding_model,
            revision=args.embedding_revision, device=args.embedding_device, modules=modules,
            pooling="SentenceTransformer model's own modules", max_seq_length=encoder.max_seq_length,
            prompts=getattr(encoder, "prompts", {}), default_prompt_name=getattr(encoder, "default_prompt_name", None),
            normalization="encode(normalize_embeddings=True), unit L2 norm",
            saved_pipeline_verified=True,
            truncation_policy="SentenceTransformer native max_seq_length; inputs beyond it may be truncated by the encoder",
            resolved_commits=sorted({config._commit_hash for module in encoder.modules()
                if (config := getattr(module, "config", None)) is not None and getattr(config, "_commit_hash", None)}),
            package_version=importlib.metadata.version("sentence-transformers"))
        if destination is not None:
            (Path(destination) / "encoder_config.txt").write_text(str(encoder), encoding="utf-8")
    else:
        raise ValueError("Similarity method must be self, sentence-transformer, or none.")
    return vectors, metadata


def analyze_experiment(out, args, *, model=None, tokenizer=None):
    """Post-generation isolated encoding; never reuses a trial KV or sampling RNG."""
    out = Path(out)
    bank = load_word_bank(out / "bank.json")
    responses = read_jsonl(out / "responses.jsonl")
    schedule = json.loads((out / "schedule.json").read_text(encoding="utf-8"))
    destination = out / "analysis"
    destination.mkdir(exist_ok=True)
    metadata = {"schema_version": SCHEMA_VERSION, "method": args.similarity, "bank_sha256": bank["sha256"],
        "row_count": len(responses), "row_keys": [[r["trial_id"], r["event_id"]] for r in responses],
        "created_at": now(), "performed_after_generation": True,
        "response_text": "full decoded response stripped only at edges; no first-word repair",
        "primary_eligibility": "single-word surface check AND not length-censored"}
    augmented = responses
    if args.similarity != "none":
        texts = list(dict.fromkeys(bank["words"] + [r["response"].strip() for r in responses if r["response"].strip()]))
        vectors, encoder_metadata = encode_similarity_texts(texts, args, model=model,
            tokenizer=tokenizer, destination=destination)
        metadata.update(encoder_metadata)
        lookup = {text: vectors[i] for i, text in enumerate(texts)}
        bank_vectors = np.stack([lookup[w] for w in bank["words"]])
        response_vectors = np.full((len(responses), vectors.shape[1]), np.nan, dtype=np.float32)
        for i, response in enumerate(responses):
            if response["response"].strip():
                response_vectors[i] = lookup[response["response"].strip()]
        augmented, matrix = similarity_rows(responses, bank_vectors, response_vectors, schedule)
        for name, values in (("bank_vectors", bank_vectors), ("response_vectors", response_vectors), ("similarity", matrix)):
            temporary = destination / (name + ".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            os.replace(temporary, destination / (name + ".npy"))
        atomic_json(destination / "embedding_texts.json", texts)
    else:
        metadata["label"] = "similarity disabled"
    temporary = destination / "responses.jsonl.tmp"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in augmented), encoding="utf-8")
    os.replace(temporary, destination / "responses.jsonl")
    with (destination / "responses.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(augmented[0]) if augmented else ["trial_id", "event_id", "cue", "response"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in augmented:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    atomic_json(destination / "cue_summary.json", summarize_cues(augmented))
    atomic_json(destination / "metadata.json", metadata)
    return destination


def summarize_scenes(responses):
    groups = {}
    for row in responses:
        groups.setdefault((row["step_index"], row["condition"]), []).append(row)
    summaries = []
    for (step, condition), rows in sorted(groups.items()):
        eligible = [r for r in rows if r["eligible"]]
        summary = {"step_index": step, "condition": condition, "n": len(rows), "eligible_n": len(eligible),
            "distinct_seed_count": len({r["sample_seed"] for r in rows}),
            "eligible_seed_count": len({r["sample_seed"] for r in eligible}),
            "sample_seeds": sorted({r["sample_seed"] for r in rows}),
            "repeat_indices": sorted({r["repeat_index"] for r in rows}),
            "aggregation": "eligible current responses only; descriptive mean/population std; "
                           "no inferential independence or equivalence of prior contexts claimed"}
        for key in ("origin_cosine", "previous_scene_cosine", "first_token_model_ms", "first_token_latency_ms",
                    "response_model_ms", "response_wall_ms", "n_tokens", "past_tokens"):
            values = [float(r[key]) for r in eligible if r.get(key) is not None and math.isfinite(float(r[key]))]
            summary.update({key + "_mean": float(np.mean(values)) if values else None,
                            key + "_std": float(np.std(values)) if values else None, key + "_n": len(values)})
        summaries.append(summary)
    return summaries


def analyze_imagination(out, args, *, model=None, tokenizer=None):
    """Compare isolated sentence/origin representations AFTER generation and cache release."""
    out = Path(out)
    origin = load_origin_scene(out / "origin.json")
    responses = read_jsonl(out / "responses.jsonl")
    destination = out / "analysis"
    destination.mkdir(exist_ok=True)
    metadata = {"schema_version": SCHEMA_VERSION, "method": args.similarity,
        "origin_sha256": origin["sha256"], "row_count": len(responses),
        "row_keys": [[r["trial_id"], r["event_id"]] for r in responses], "created_at": now(),
        "performed_after_generation": True,
        "response_text": "full decoded response stripped only at edges; no sentence trimming or repair",
        "primary_eligibility": "one-sentence surface heuristic AND not length-censored",
        "previous_scene": "origin at step 0, exact prior step within same trial thereafter; reset adjacent outputs were not used as context",
        "matrix_axes": "rows and columns follow responses.jsonl / row_keys; all sampled outputs, including ineligible"}
    augmented = responses
    if args.similarity != "none":
        texts = list(dict.fromkeys([origin["scene"]] + [r["response"].strip() for r in responses if r["response"].strip()]))
        vectors, encoder_metadata = encode_similarity_texts(texts, args, model=model,
            tokenizer=tokenizer, destination=destination)
        metadata.update(encoder_metadata)
        lookup = {text: vectors[i] for i, text in enumerate(texts)}
        origin_vectors = lookup[origin["scene"]][None]
        response_vectors = np.full((len(responses), vectors.shape[1]), np.nan, dtype=np.float32)
        for i, row in enumerate(responses):
            if row["response"].strip():
                response_vectors[i] = lookup[row["response"].strip()]
        augmented, matrix = scene_similarity_rows(responses, origin_vectors, response_vectors)
        for name, values in (("origin_vectors", origin_vectors), ("response_vectors", response_vectors), ("similarity", matrix)):
            temporary = destination / (name + ".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            os.replace(temporary, destination / (name + ".npy"))
        atomic_json(destination / "embedding_texts.json", texts)
    else:
        metadata["label"] = "similarity disabled"
    atomic_jsonl(destination / "responses.jsonl", augmented)
    with (destination / "responses.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(augmented[0]) if augmented else ["trial_id", "event_id", "step_index", "response"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in augmented:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    atomic_json(destination / "step_summary.json", summarize_scenes(augmented))
    atomic_json(destination / "metadata.json", metadata)
    return destination


def write_experiment_transcript(out):
    out = Path(out)
    rows = read_jsonl(out / "responses.jsonl")
    lines = ["# Bank experiment transcript", "", "R = response row; B = original frozen bank index.", ""]
    for i, row in enumerate(rows):
        lines += [f"## R{i} · {row['trial_id']} · B{row['bank_index']} · {row['condition']} · seed {row['sample_seed']}",
                  "", f"Cue: {row['cue']}", "", "```text", row["response"], "```", "",
                  f"Eligible: {row['eligible']}; stop: {row['stop_reason']}; past/new/output tokens: "
                  f"{row['past_tokens']}/{row['new_prompt_tokens']}/{row['generated_tokens']}", ""]
    (out / "transcript.md").write_text("\n".join(lines), encoding="utf-8")


def write_transcript(out):
    events = read_jsonl(Path(out) / "events.jsonl")
    lines = ["# Association Lab transcript", "", "Event numbers match figure labels.", ""]
    for event in events:
        lines += [f"## Event {event['event_id']} · {event['protocol']}", "",
                  f"Cue: {event['cue']}", "", "```text", event["response"], "```", "",
                  f"Stop: {event['stop_reason']}; tokens: {event['generated_tokens']}; context: {event['history_context']}", ""]
    (Path(out) / "transcript.md").write_text("\n".join(lines), encoding="utf-8")


def _read_plot_events(run_dir, event_id=None):
    """Load committed, complete events; never consume a memmap's unwritten tail."""
    import json
    from pathlib import Path

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    events = []
    index = root / "events.jsonl"
    if not index.exists():
        raise ValueError("No committed events.jsonl is present in this run.")
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event_id is not None and int(event["event_id"]) != int(event_id):
            continue
        directory = root / event.get("directory", f"events/{int(event['event_id']):06d}")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if str(meta.get("status", "")).lower() not in {"complete", "completed"}:
            continue
        n = int(meta.get("n_tokens", event.get("generated_tokens", 0)))
        if int(meta.get("valid_rows", 0)) < n + 1:
            raise ValueError(f"Event {event['event_id']} lacks its terminal consumed state.")
        rows = [json.loads(s) for s in (directory / "tokens.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
        if len(rows) != n or any(int(row["t"]) != i for i, row in enumerate(rows)):
            raise ValueError(f"Event {event['event_id']} has inconsistent token records.")
        events.append({"event": event, "meta": meta, "rows": rows, "directory": directory})
    if not events:
        raise ValueError("No completed events match this selection.")
    return root, manifest, events


def _plot_array_path(entry):
    """Accept a filename or a small file/path descriptor in the array map."""
    if isinstance(entry, dict):
        return entry.get("file", entry.get("path"))
    return entry


def _plot_site_order(manifest, events):
    names = [x["name"] if isinstance(x, dict) else x for x in manifest.get("hook_sites", [])]
    for item in events:
        for name in item["meta"].get("hidden", {}):
            if name not in names:
                names.append(name)
    return names


def _plot_is_demo(manifest):
    model = manifest.get("model", {})
    if isinstance(model, dict):
        if model.get("synthetic") is True or model.get("untrained") is True:
            return True
        identifier = str(model.get("identifier", "")).lower()
    else:
        identifier = str(model).lower()
    return identifier.startswith(("tiny-random-", "tiny_random_"))


def _plot_parent_edges(events, manifest):
    """Only explicit retained-text/word-chain dependencies receive arrows."""
    positions = {item["event"]["event_id"]: i for i, item in enumerate(events)}
    chain = manifest.get("config", {}).get("chain") is True
    edges = []
    for i, item in enumerate(events):
        event = item["event"]
        parent = event.get("parent_event_id")
        dependency = event.get("history_context") in ("carry", "live") or chain
        if dependency and parent is not None and parent in positions:
            j = positions[parent]
            if j < i:
                edges.append((j, i))
    return edges


def plot_run(run_dir, event_id=None):
    """Make offline diagnostics from saved arrays; neither torch nor weights are used.

    Exact raw logits and hidden states remain in the run. The vocabulary heatmap
    is explicitly binned when necessary, so every ID contributes with bounded
    plot memory. PCA is a within-run descriptive projection, fitted to event
    means, with event endpoints projected into that same fitted basis.
    """
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, manifest, events = _read_plot_events(run_dir, event_id)
    out = root / "plots"
    if event_id is not None:
        out = out / f"event_{int(event_id):06d}"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    demo = _plot_is_demo(manifest)
    heading = "UNTRAINED / SYNTHETIC DEMO | " if demo else ""
    offsets = []
    rows = []
    for item in events:
        offsets.append(len(rows))
        rows.extend(item["rows"])
    x = np.arange(len(rows))

    def series(key):
        return np.asarray([r.get(key, np.nan) if r.get(key) is not None else np.nan for r in rows], dtype=np.float64)

    def boundaries(axes):
        for ax in np.atleast_1d(axes).flat:
            for offset in offsets[1:]:
                ax.axvline(offset - 0.5, color="0.55", linewidth=0.65, linestyle=":")
            ax.grid(alpha=0.15)
        for item, offset in zip(events, offsets):
            if len(events) <= 20:
                np.atleast_1d(axes).flat[0].text(offset, 1.015, f"E{item['event']['event_id']}", transform=np.atleast_1d(axes).flat[0].get_xaxis_transform(), fontsize=8)

    def save(fig, name):
        path = out / name
        temporary = path.with_name(path.stem + ".tmp.png")
        fig.savefig(temporary, format="png", dpi=155, bbox_inches="tight")
        plt.close(fig)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Empty plot output: {temporary}")
        os.replace(temporary, path)
        paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    axes[0].plot(x, series("raw_entropy_nats"), label="Raw model entropy", linewidth=1.2)
    axes[0].plot(x, series("policy_entropy_nats"), label="Sampling-policy entropy", linewidth=1, alpha=0.8)
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(x, series("raw_surprisal_nats"), label="Sampled token: raw surprisal", linewidth=1.1)
    axes[1].plot(x, series("policy_surprisal_nats"), label="Sampled token: policy surprisal", linewidth=0.9, alpha=0.8)
    axes[1].set_ylabel("Surprisal (nats)")
    axes[1].legend(loc="best", fontsize=8)
    forward = series("instrumented_forward_ms")
    axes[2].plot(x, forward, label="Instrumented forward (hooks included)", linewidth=1)
    host = series("capture_to_cpu_ms") + series("sampling_stats_ms") + series("array_write_ms")
    axes[2].plot(x, host, label="Host capture + sampling/stats + array write", linewidth=0.9, alpha=0.8)
    starts = np.asarray([v for v in offsets if v < len(rows)], dtype=int)
    if len(starts):
        axes[2].scatter(starts, forward[starts], marker="s", s=28, color="#d35c31", label="Prefill / first-token forward", zorder=4)
    if np.any(forward > 0):
        axes[2].set_yscale("log")
    axes[2].set_ylabel("Wall time (ms, log)")
    axes[2].set_xlabel("Generated token index (concatenated events; dotted lines reset context measurement)")
    axes[2].legend(loc="best", fontsize=8)
    boundaries(axes)
    fig.suptitle(heading + "Token dynamics | raw model and actual sampling policy", fontsize=13)
    fig.text(0.5, 0.005, "Terminal consume-only passes are saved separately; timings are instrumented runtime observations.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    save(fig, "token_dynamics.png")

    names = _plot_site_order(manifest, events)
    fig, state_axes = plt.subplots(2, 1, sharex=True, figsize=(12, max(6.2, min(20, 0.50 * len(names) + 4))))
    drift = np.full((len(names), len(rows)), np.nan, dtype=np.float32)
    norms = np.full_like(drift, np.nan)
    for si, name in enumerate(names):
        for item, offset in zip(events, offsets):
            entry = item["meta"].get("hidden", {}).get(name)
            n = len(item["rows"])
            if not entry or n < 1:
                continue
            arr = np.load(item["directory"] / _plot_array_path(entry), mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2 or arr.shape[0] < n + 1:
                raise ValueError(f"Hidden array for {name} has incompatible shape.")
            for start in range(0, n, 128):
                stop = min(n, start + 128)
                current = np.asarray(arr[start + 1:stop + 1], dtype=np.float64)
                norms[si, offset + start:offset + stop] = np.linalg.norm(current, axis=1)
            # t=0 deliberately remains NaN: never compare an event's first
            # generated state with a prior event or the last prompt state.
            for start in range(1, n, 128):
                stop = min(n, start + 128)
                previous = np.asarray(arr[start:stop], dtype=np.float64)
                current = np.asarray(arr[start + 1:stop + 1], dtype=np.float64)
                den = np.linalg.norm(previous, axis=1) * np.linalg.norm(current, axis=1)
                cosine = np.divide(np.einsum("ij,ij->i", previous, current), den, out=np.full(len(den), np.nan), where=den > 0)
                drift[si, offset + start:offset + stop] = 1 - np.clip(cosine, -1, 1)
    if names and len(rows):
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        im = state_axes[0].imshow(drift, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap, vmin=0, vmax=2)
        fig.colorbar(im, ax=state_axes[0], label="1 - cosine(previous, current)", shrink=0.8)
        # log1p keeps zero norms visible while compressing large inter-site
        # magnitude differences. Exact untransformed vectors remain on disk.
        im_norm = state_axes[1].imshow(np.log1p(norms), aspect="auto", interpolation="nearest", origin="upper", cmap=cmap, vmin=0)
        fig.colorbar(im_norm, ax=state_axes[1], label="ln(1 + L2 norm)", shrink=0.8)
        for ax in state_axes:
            ax.set_yticks(np.arange(len(names)), labels=names, fontsize=max(6, 9 - len(names) // 20))
        boundaries(state_axes)
        state_axes[1].set_xlabel("Generated token index (same event boundaries as token_dynamics)")
    else:
        for ax in state_axes:
            ax.text(0.5, 0.5, "No saved consumed-token states.\nEnable hooks to plot state direction and magnitude.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    state_axes[0].set_title(heading + "Consumed-token direction change | first token of each event is masked", pad=20)
    state_axes[1].set_title("Consumed-token magnitude | first tokens included; natural-log display", pad=12)
    fig.tight_layout()
    save(fig, "state_drift.png")

    common = set(names)
    for item in events:
        common.intersection_update(item["meta"].get("hidden", {}))
    ordered_common = [n for n in names if n in common]
    norm_sites = [n for n in ordered_common if n.endswith((".norm", ".ln_f", ".final_layer_norm")) or n == "final_norm"]
    site = norm_sites[-1] if norm_sites else ordered_common[-1] if ordered_common else None
    vectors, endpoints, kept = [], [], []
    if site:
        for item in events:
            indices = item["meta"].get("content_token_indices", item["event"].get("content_token_indices", []))
            indices = [int(i) for i in indices if 0 <= int(i) < len(item["rows"])]
            if not indices:
                continue
            arr = np.load(item["directory"] / _plot_array_path(item["meta"]["hidden"][site]), mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2 or arr.shape[0] < int(item["meta"]["valid_rows"]):
                raise ValueError("Trajectory hidden array disagrees with metadata.")
            accum = np.zeros(arr.shape[1], dtype=np.float64)
            for k in range(0, len(indices), 128):
                accum += np.asarray(arr[np.asarray(indices[k:k + 128]) + 1], dtype=np.float64).sum(axis=0)
            vector = accum / len(indices)
            endpoint = np.asarray(arr[indices[-1] + 1], dtype=np.float64)
            if vectors and vector.shape != vectors[0].shape:
                raise ValueError("Cannot fit one PCA to differently sized hook sites.")
            vectors.append(vector)
            endpoints.append(endpoint)
            kept.append(item)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    if len(vectors) >= 2:
        matrix = np.stack(vectors)
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        power = float(np.square(singular).sum())
        if power > np.finfo(float).eps:
            basis = vt[:2].T
            points = centered @ basis
            end_points = (np.stack(endpoints) - mean) @ basis
            if points.shape[1] == 1:
                points = np.column_stack([points[:, 0], np.zeros(len(points))])
                end_points = np.column_stack([end_points[:, 0], np.zeros(len(end_points))])
            explained = np.pad(np.square(singular[:2]) / power, (0, max(0, 2 - len(singular[:2]))))
            ax.scatter(points[:, 0], points[:, 1], s=55, label="Mean of consumed content-token states", zorder=3)
            ax.scatter(end_points[:, 0], end_points[:, 1], s=36, marker="x", color="#bb5c31", label="Last content-token state (same PCA)", zorder=3)
            for i, item in enumerate(kept):
                ax.annotate(f"E{item['event']['event_id']}", points[i], xytext=(5, 5), textcoords="offset points", fontsize=9)
            for parent, child in _plot_parent_edges(kept, manifest):
                ax.annotate("", xy=points[child], xytext=points[parent], arrowprops={"arrowstyle": "->", "color": "0.6", "lw": 0.9})
            ax.set_xlabel(f"PC1 ({explained[0]:.1%} of pooled-event variance)")
            ax.set_ylabel(f"PC2 ({explained[1]:.1%} of pooled-event variance)")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.18)
        else:
            ax.text(0.5, 0.5, "Pooled event vectors are identical; no PCA variance.", ha="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "PCA needs at least two completed events\nwith content tokens and a common hook site.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(heading + "Event states | shared within-run PCA\n" + (site or "No common hook site"))
    fig.text(0.5, 0.015, "Numeric event IDs map to transcript.md. Axes are descriptive; no cross-run basis alignment is implied.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "event_trajectory.png")

    chosen = next((i for i in events if i["meta"].get("arrays", {}).get("raw_logits")), None)
    if chosen and chosen["rows"]:
        source = chosen["directory"] / _plot_array_path(chosen["meta"]["arrays"]["raw_logits"])
        arr = np.load(source, mmap_mode="r", allow_pickle=False)
        n = len(chosen["rows"])
        if arr.ndim != 2 or arr.shape[0] < n:
            raise ValueError("Raw-logits shape disagrees with token records.")
        vocab = arr.shape[1]
        nbv, nbt = min(vocab, 2048), min(n, 512)
        vb = np.linspace(0, vocab, nbv + 1, dtype=np.int64)
        tb = np.linspace(0, n, nbt + 1, dtype=np.int64)
        heat = np.empty((nbv, nbt), dtype=np.float32)
        mask_fraction = np.empty_like(heat)
        # At most one full-vocabulary vector and a bounded display matrix live
        # in RAM. Every vocabulary ID contributes, including low-logit tails.
        for ti in range(nbt):
            accum = np.zeros(nbv, dtype=np.float64)
            finite_count = np.zeros(nbv, dtype=np.int64)
            for t in range(tb[ti], tb[ti + 1]):
                row = np.asarray(arr[t], dtype=np.float64)
                if np.isnan(row).any() or np.isposinf(row).any() or not np.isfinite(row).any():
                    raise ValueError("Raw logits contain NaN/+inf or an entirely masked distribution.")
                finite = np.isfinite(row)
                accum += np.add.reduceat(np.where(finite, row, 0.0), vb[:-1])
                finite_count += np.add.reduceat(finite.astype(np.int64), vb[:-1])
            heat[:, ti] = np.divide(accum, finite_count, out=np.full(nbv, np.nan), where=finite_count > 0)
            mask_fraction[:, ti] = 1 - finite_count / (np.diff(vb) * (tb[ti + 1] - tb[ti]))
        has_mask = bool(np.any(mask_fraction > 0))
        fig, logits_axes = plt.subplots(2 if has_mask else 1, 1, figsize=(11, 9 if has_mask else 6.5), sharex=True, squeeze=False)
        ax = logits_axes[0, 0]
        extent = (-0.5, n - 0.5, -0.5, vocab - 0.5)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#aaaaaa")
        im = ax.imshow(np.ma.masked_invalid(heat), aspect="auto", origin="lower", extent=extent, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax, label="Finite raw logits (bin mean where downsampled)")
        ax.set_title(heading + f"E{chosen['event']['event_id']} | full-vocabulary raw-logit overview")
        if has_mask:
            masked_ax = logits_axes[1, 0]
            mask_im = masked_ax.imshow(mask_fraction, aspect="auto", origin="lower", extent=extent, interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
            fig.colorbar(mask_im, ax=masked_ax, label="Fraction of entries equal to -inf")
            masked_ax.set_title("Masked vocabulary entries | finite-only means above; all-masked bins are gray")
        for axes_row in logits_axes:
            axes_row[0].set_ylabel("Vocabulary token ID (every ID included)")
        logits_axes[-1, 0].set_xlabel("Generated token / predictor row (terminal unsampled row excluded)")
        fig.text(0.5, 0.015, f"Source {n} x {vocab}; display {nbt} x {nbv} bins. Exact arrays retain every finite value and -inf mask.", ha="center", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.text(0.5, 0.5, "No saved raw logits in this selection.\nEnable full-logit saving for a vocabulary overview.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(heading + "Raw-logit overview unavailable")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "logits_overview.png")
    return paths



def plot_experiment(experiment_dir):
    """Plot committed bank-experiment observations without loading torch or a model.

    Row IDs R0, R1, ... follow the root responses.jsonl order. Bank IDs B0, B1,
    ... follow bank.json. These IDs avoid changing text or relying on a local
    Japanese font. The raw text and exact matrix remain in the saved files.
    """
    import json
    import os
    from pathlib import Path

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(experiment_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"No experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "bank_experiment":
        raise ValueError("plot_experiment expects kind='bank_experiment'.")
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    warnings = []

    def json_object(path, default):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def json_lines(path):
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        result = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not text.endswith("\n") and manifest.get("status") != "complete":
                    warnings.append(f"Ignored uncommitted final line of {path.name}.")
                else:
                    raise ValueError(f"Invalid JSON in {path}, line {i + 1}") from None
        return result

    bank = json_object(root / "bank.json", {})
    words = bank if isinstance(bank, list) else bank.get("words", [])
    rows = json_lines(root / "responses.jsonl")
    analysis = json_object(root / "analysis" / "metadata.json", {})
    analyzed_rows = json_lines(root / "analysis" / "responses.jsonl")

    def row_key(row):
        if "trial_id" not in row or "event_id" not in row:
            raise ValueError("Each response needs trial_id and event_id for plot alignment.")
        return str(row["trial_id"]), str(row["event_id"])

    root_keys = [row_key(row) for row in rows]
    analysis_keys = [row_key(row) for row in analyzed_rows]
    if len(set(root_keys)) != len(root_keys) or len(set(analysis_keys)) != len(analysis_keys):
        raise ValueError("Duplicate trial/event response IDs; plot alignment is ambiguous.")
    if set(analysis_keys) - set(root_keys):
        raise ValueError("Analysis refers to responses absent from root responses.jsonl.")
    by_key = {row_key(row): row for row in analyzed_rows}
    # Analysis contributes derived fields. Identity and measured response fields
    # stay authoritative in the root summary, even when analysis is partial.
    rows = [{**by_key.get(key, {}), **row} for key, row in zip(root_keys, rows)]
    if isinstance(bank, dict) and analysis.get("bank_sha256") and bank.get("sha256"):
        if analysis["bank_sha256"] != bank["sha256"]:
            raise ValueError("Analysis bank hash disagrees with bank.json.")

    similarity = None
    matrix_path = root / "analysis" / "similarity.npy"
    if matrix_path.exists():
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape != (len(analyzed_rows), len(words)):
            raise ValueError("Similarity matrix shape must match analysis responses and bank words.")
        similarity = np.full((len(rows), len(words)), np.nan, dtype=np.float32)
        location = {key: i for i, key in enumerate(root_keys)}
        for i, key in enumerate(analysis_keys):
            similarity[location[key]] = matrix[i]
        finite = similarity[np.isfinite(similarity)]
        if finite.size and (finite.min() < -1.0001 or finite.max() > 1.0001):
            raise ValueError("Cosine matrix contains values outside [-1, 1].")

    def number(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError, OverflowError):
            return np.nan

    def values(key):
        return np.asarray([number(row.get(key)) for row in rows], dtype=float)

    n = len(rows)
    b = len(words)
    eligible = np.asarray([row.get("eligible") is True for row in rows], dtype=bool)
    censored = np.asarray([row.get("censored") is True or row.get("stop_reason") == "length"
                           for row in rows], dtype=bool)
    bank_index = values("bank_index")
    cue_cosine = values("cue_cosine")
    if similarity is not None:
        for i in range(n):
            j = bank_index[i]
            if np.isfinite(j) and j.is_integer() and 0 <= int(j) < b:
                matrix_value = float(similarity[i, int(j)])
                if np.isfinite(cue_cosine[i]) and np.isfinite(matrix_value):
                    if not np.isclose(cue_cosine[i], matrix_value, rtol=1e-5, atol=1e-5):
                        raise ValueError(f"Response R{i} cue cosine disagrees with the matrix.")
                elif np.isfinite(matrix_value):
                    cue_cosine[i] = matrix_value

    seeds = sorted({str(row.get("sample_seed", "?")) for row in rows},
                   key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x))
    seed_index = {seed: i for i, seed in enumerate(seeds)}
    colors = plt.get_cmap("tab10" if len(seeds) <= 10 else "turbo")

    def seed_color(row):
        i = seed_index[str(row.get("sample_seed", "?"))]
        return colors(i if len(seeds) <= 10 else i / max(1, len(seeds) - 1))

    conditions = list(dict.fromkeys(str(row.get("condition", "?")) for row in rows))
    condition_markers = {"live": "o", "reset": "s"}
    condition_lines = {"live": "-", "reset": "--"}
    trial_ids = list(dict.fromkeys(str(row["trial_id"]) for row in rows))
    model = manifest.get("model", {})
    synthetic = (manifest.get("synthetic") is True or manifest.get("untrained") is True or
                 isinstance(model, dict) and (model.get("synthetic") is True or model.get("untrained") is True))
    prefix = "UNTRAINED / SYNTHETIC DEMO | " if synthetic else ""
    status = str(manifest.get("status", "unknown"))
    label = str(analysis.get("label") or {
        "self": "Isolated-LM representation cosine (proxy)",
        "sentence-transformer": "Sentence-encoder cosine",
        "none": "Similarity disabled",
    }.get(analysis.get("method", manifest.get("config", {}).get("similarity")), "Similarity unavailable"))
    if analysis.get("method") == "self" and "proxy" not in label.lower():
        label += " (proxy)"
    # Keep externally supplied model names/labels from consuming the figure.
    short_label = label if len(label) <= 100 else label[:97] + "..."
    summary = (f"{n} completed responses | {int(eligible.sum())} eligible | "
               f"{int(censored.sum())} capped | {len(trial_ids)} trials | {len(seeds)} distinct sample seeds")
    if status not in {"complete", "completed"}:
        summary += f" | status: {status}"

    def save(fig, name):
        path = out / name
        temporary = path.with_name(path.stem + ".tmp.png")
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Empty plot output: {temporary}")
        os.replace(temporary, path)
        paths.append(path)

    def empty(ax, message):
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
                color="#555555", fontsize=11, linespacing=1.5)
        ax.set_axis_off()

    def decorate(axes):
        for ax in np.atleast_1d(axes).flat:
            ax.grid(alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    def legends(fig, extra=None):
        handles = [Line2D([], [], color="#444444", marker=condition_markers.get(c, "D"),
                          linestyle="", markersize=6, label=f"{c}: eligible") for c in conditions]
        if np.any(~eligible):
            handles.append(Line2D([], [], color="#b23b3b", marker="x", linestyle="", markersize=6,
                                  label="Ineligible / capped"))
        if len(seeds) <= 10:
            handles.extend(Line2D([], [], color=seed_color({"sample_seed": seed}), marker="o",
                                  linestyle="", markersize=5, label=f"seed {seed}") for seed in seeds)
        elif seeds:
            handles.append(Line2D([], [], color="none", label=f"Colors: {len(seeds)} seeds in numeric order"))
        if extra:
            handles.extend(extra)
        if handles:
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
                       ncol=min(6, len(handles)), fontsize=8, frameon=False)

    def point_cloud(ax, x, y, *, offsets=None):
        finite = np.isfinite(x) & np.isfinite(y)
        xx = x if offsets is None else x + offsets
        for condition in conditions:
            mask = finite & eligible & np.asarray([str(row.get("condition", "?")) == condition for row in rows])
            indices = np.flatnonzero(mask)
            if indices.size:
                ax.scatter(xx[indices], y[indices], c=[seed_color(rows[i]) for i in indices],
                           marker=condition_markers.get(condition, "D"), s=28, alpha=0.72,
                           linewidths=0.45, edgecolors="white", zorder=3)
        rejected = finite & ~eligible
        if np.any(rejected):
            ax.scatter(xx[rejected], y[rejected], marker="x", s=25, color="#b23b3b", alpha=0.75,
                       linewidths=0.9, zorder=3)
        return int(finite.sum())

    def seed_offsets(width=0.3):
        # Deterministic display offsets only; never sample or alter measured values.
        keys = list(dict.fromkeys((str(row["trial_id"]), row.get("condition")) for row in rows))
        index = {key: i for i, key in enumerate(keys)}
        return np.asarray([(index[(str(row["trial_id"]), row.get("condition"))] / max(1, len(keys) - 1) - 0.5) * 2 * width
                           if len(keys) > 1 else 0.0 for row in rows])

    # 1. Actual measured model time against cue similarity. Do not fit a pooled
    # regression: context length, response length, order and seeds are repeated.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.9))
    metrics = [("first_token_model_ms", "First-token model forward (ms)"),
               ("response_model_ms", "All predictor forwards (ms)")]
    for ax, (metric, title) in zip(axes, metrics):
        count = point_cloud(ax, cue_cosine, values(metric))
        if not count:
            empty(ax, "No completed responses." if not n else
                  "No usable cue cosine / timing pairs.\nRun similarity analysis to fill this panel.")
        else:
            ax.set_xlabel("Response-to-presented-cue cosine")
            ax.set_ylabel(title)
        ax.set_title(title)
    decorate(axes)
    fig.suptitle(prefix + "Timing and cue affinity\n" + short_label, fontsize=13)
    fig.text(0.5, 0.90, summary, ha="center", fontsize=8)
    legends(fig)
    fig.text(0.5, 0.013,
             "Instrumented forward timings include hooks; total excludes final consume. Repeats of one seed are not new random draws.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.16 if len(seeds) > 4 else 0.13, 1, 0.88))
    save(fig, "latency_similarity.png")

    # 2. Full response x frozen-bank matrix; neither select only top candidates
    # nor silently drop invalid response rows. The status gutter makes exclusions visible.
    height = max(6.0, min(16.0, n * 0.18 + 3.0))
    fig = plt.figure(figsize=(max(9.5, min(15.0, b * 0.28 + 4.5)), height))
    grid = fig.add_gridspec(1, 3, width_ratios=(10, 0.22, 0.40), wspace=0.10)
    ax = fig.add_subplot(grid[0, 0])
    status_ax = fig.add_subplot(grid[0, 1])
    color_ax = fig.add_subplot(grid[0, 2])
    if similarity is None or not n or not b or not np.isfinite(similarity).any():
        status_ax.set_axis_off()
        color_ax.set_axis_off()
        empty(ax, "No completed responses." if not n else
              "Full-bank similarity matrix unavailable.\nEnable --similarity self or sentence-transformer.")
    else:
        from matplotlib.colors import ListedColormap
        state = np.where(eligible, 0, np.where(censored, 2, 1))
        status_ax.imshow(state[:, None], aspect="auto", interpolation="nearest", origin="upper",
                         cmap=ListedColormap(["#d6e6d9", "#b23b3b", "#e19a43"]), vmin=-0.5, vmax=2.5)
        status_ax.set_xticks([0], labels=["status"], rotation=90, fontsize=8)
        status_ax.set_yticks([])
        for spine in status_ax.spines.values():
            spine.set_visible(False)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        finite = similarity[np.isfinite(similarity)]
        low, high = float(finite.min()), float(finite.max())
        if high - low < 1e-6:
            low, high = max(-1.0, low - 0.01), min(1.0, high + 0.01)
        im = ax.imshow(np.ma.masked_invalid(similarity), aspect="auto", origin="upper",
                       interpolation="nearest", cmap=cmap, vmin=low, vmax=high)
        valid_cues = np.isfinite(bank_index) & (bank_index >= 0) & (bank_index < b) & (bank_index == np.floor(bank_index))
        ax.scatter(bank_index[valid_cues], np.flatnonzero(valid_cues), s=16 if n <= 150 else 5,
                   marker="s", facecolors="none", edgecolors="white", linewidths=0.65)
        col_ticks = np.unique(np.linspace(0, b - 1, min(b, 32), dtype=int))
        row_ticks = np.unique(np.linspace(0, n - 1, min(n, 38), dtype=int))
        ax.set_xticks(col_ticks, labels=[f"B{i}" for i in col_ticks], fontsize=8,
                      rotation=90 if b > 20 else 0)
        ax.set_yticks(row_ticks, labels=[f"R{i}" for i in row_ticks], fontsize=8)
        for i in range(1, n):
            if rows[i]["trial_id"] != rows[i - 1]["trial_id"]:
                ax.axhline(i - 0.5, color="white", linewidth=0.55, alpha=0.7)
        ax.set_xlabel("Frozen bank index (bank.json); outlined cell = presented cue")
        ax.set_ylabel("Response row (root responses.jsonl)")
        fig.colorbar(im, cax=color_ax, label="Cosine (shared observed color range)")
    fig.suptitle(prefix + "Each response against the whole bank\n" + short_label, fontsize=13)
    fig.text(0.5, 0.92, summary, ha="center", fontsize=8)
    fig.text(0.5, 0.02,
             "Status: green = eligible; red = rule failure; orange = cap. Gray = unavailable; white lines = trial boundaries. All rows retained.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.89, bottom=0.13, top=0.86)
    save(fig, "bank_affinity.png")

    # 3. Repetition is visible at each fixed bank index. Raw observations plus
    # within-condition eligible means; no confidence intervals or p values.
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    offsets = seed_offsets()
    cue_metrics = [(cue_cosine, "Cue cosine"),
                   (values("first_token_model_ms"), "First-token model forward (ms)")]
    condition_colors = {c: ("#265a88" if c == "live" else "#c16b35" if c == "reset" else "#505050") for c in conditions}
    for ax, (metric, ylabel) in zip(axes[:2], cue_metrics):
        count = point_cloud(ax, bank_index, metric, offsets=offsets)
        for ci, condition in enumerate(conditions):
            cond = np.asarray([str(row.get("condition", "?")) == condition for row in rows])
            for j in range(b):
                mask = eligible & cond & (bank_index == j) & np.isfinite(metric)
                if mask.any():
                    center = j + (ci - (len(conditions) - 1) / 2) * 0.10
                    ax.plot([center - 0.075, center + 0.075], [np.mean(metric[mask])] * 2,
                            color=condition_colors[condition], linewidth=2.2, zorder=5)
        if not count:
            empty(ax, "No eligible or marked observations for this metric.")
        ax.set_ylabel(ylabel)
    if n and b:
        width = min(0.72 / max(1, len(conditions)), 0.36)
        for ci, condition in enumerate(conditions):
            xs = np.arange(b) + (ci - (len(conditions) - 1) / 2) * width
            counts, distinct = [], []
            for j in range(b):
                selected = [row for row in rows if row.get("eligible") is True and
                            str(row.get("condition", "?")) == condition and number(row.get("bank_index")) == j]
                counts.append(len(selected))
                distinct.append(len({str(row.get("response", "")).strip() for row in selected}))
            axes[2].bar(xs, counts, width=width * 0.92, facecolor="none", edgecolor=condition_colors[condition],
                        linewidth=1.0, alpha=0.8, label=f"{condition}: eligible observations")
            axes[2].bar(xs, distinct, width=width * 0.52, color=condition_colors[condition], alpha=0.65,
                        label=f"{condition}: distinct raw responses")
        axes[2].legend(loc="best", fontsize=8, ncol=min(2, max(1, len(conditions))))
        ticks = np.unique(np.linspace(0, b - 1, min(b, 36), dtype=int))
        axes[2].set_xticks(ticks, labels=[f"B{i}" for i in ticks], fontsize=8)
        axes[2].set_xlim(-0.65, b - 0.35)
        axes[2].set_ylim(bottom=0)
    else:
        empty(axes[2], "No completed responses.")
    axes[2].set_ylabel("Count (eligible only)")
    axes[2].set_xlabel("Frozen bank index; horizontal jitter separates trials, without changing measured values")
    decorate(axes)
    fig.suptitle(prefix + "Variation for the same cues\n" + short_label, fontsize=13)
    fig.text(0.5, 0.92, summary, ha="center", fontsize=8)
    extra = [Line2D([], [], color=condition_colors[c], linewidth=2.2, label=f"{c}: eligible mean") for c in conditions]
    legends(fig, extra)
    fig.text(0.5, 0.014,
             "Means pool displayed eligible observations, not independent estimates. Exact raw-string diversity; repeated seeds are counted separately.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.91))
    save(fig, "cue_variation.png")

    # 4. Keep each trial's progression separate. Prefix and response lengths make
    # the obvious runtime confounders visible beside the observed durations.
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for trial_id in trial_ids:
        indices = [i for i, row in enumerate(rows) if str(row["trial_id"]) == trial_id]
        indices.sort(key=lambda i: (number(rows[i].get("position")) if np.isfinite(number(rows[i].get("position"))) else i))
        if not indices:
            continue
        indices = np.asarray(indices, dtype=int)
        first = rows[int(indices[0])]
        color = seed_color(first)
        condition = str(first.get("condition", "?"))
        linestyle = condition_lines.get(condition, ":")
        position = values("position")[indices]
        for ax, metric in zip(axes[:2], ["first_token_model_ms", "response_model_ms"]):
            ys = values(metric)[indices]
            ax.plot(position, ys, color=color, linestyle=linestyle, linewidth=0.85,
                    marker=condition_markers.get(condition, "D"), markersize=3, alpha=0.65)
            bad = ~eligible[indices]
            if bad.any():
                ax.scatter(position[bad], ys[bad], marker="x", s=24, color="#b23b3b", zorder=4)
        axes[2].plot(position, values("past_tokens")[indices], color=color, linestyle=linestyle,
                     linewidth=1, alpha=0.7)
        axes[2].plot(position, values("new_prompt_tokens")[indices], color=color, linestyle=":",
                     linewidth=0.8, marker="+", markersize=4, alpha=0.6)
        axes[3].plot(position, values("generated_tokens")[indices], color=color, linestyle=linestyle,
                     linewidth=0.85, marker=condition_markers.get(condition, "D"), markersize=3, alpha=0.65)
    if not n:
        for ax in axes:
            empty(ax, "No completed responses.")
    axes[0].set_ylabel("First forward (ms)")
    axes[1].set_ylabel("All predictors (ms)")
    axes[2].set_ylabel("Tokens before response")
    axes[3].set_ylabel("Generated tokens")
    axes[3].set_xlabel("Cue presentation position within each trial (zero-based; order may differ by order seed)")
    if n:
        axes[2].legend(handles=[Line2D([], [], color="#555555", linestyle="-", label="Past cached prefix (solid live / dashed reset)"),
                               Line2D([], [], color="#555555", linestyle=":", marker="+", label="Newly appended prompt tokens")],
                       fontsize=8, loc="best")
    decorate(axes)
    fig.suptitle(prefix + "Within-trial progression: time and context size", fontsize=13)
    fig.text(0.5, 0.94, summary, ha="center", fontsize=8)
    legends(fig)
    fig.text(0.5, 0.014,
             "One line per trial; all completed timings retained. Solid = live, dashed = reset. Shared seed colors do not imply independent repetitions.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save(fig, "trial_progress.png")

    # Machine-readable index makes figure IDs and any partial-log omission
    # unambiguous, without requiring the plotting environment to render Japanese.
    plot_index = {
        "bank_ids": [{"bank_id": f"B{i}", "bank_index": i, "word": word} for i, word in enumerate(words)],
        "response_ids": [{"row_id": f"R{i}", "row_index": i, "trial_id": row["trial_id"],
                          "event_id": row["event_id"], "position": row.get("position"),
                          "bank_index": row.get("bank_index"), "cue": row.get("cue"),
                          "response": row.get("response"), "eligible": row.get("eligible"),
                          "sample_seed": row.get("sample_seed"), "repeat_index": row.get("repeat_index"),
                          "order_seed": row.get("order_seed"), "condition": row.get("condition")}
                         for i, row in enumerate(rows)],
        "similarity_label": label,
        "warnings": warnings,
    }
    index_path = out / "index.json"
    index_temp = index_path.with_name("index.json.tmp")
    index_temp.write_text(json.dumps(plot_index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(index_temp, index_path)
    return paths


def scene_similarity_rows(responses, origin_vector, response_vectors):
    """Isolated cosine observations, keyed by exact trial/step identities.

    All decoded responses are retained, including rule failures. The previous
    scene is the origin at step 0, otherwise exactly step_index - 1 in the same
    trial. A missing predecessor stays missing; it is never silently skipped.
    Matrix row/column order is the original responses order, not sorted order.
    """
    origin = np.asarray(origin_vector, dtype=np.float32)
    if origin.ndim == 2 and origin.shape[0] == 1:
        origin = origin[0]
    vectors = np.asarray(response_vectors, dtype=np.float32)
    if origin.ndim != 1 or not origin.size:
        raise ValueError("Origin vector must be a nonempty 1-D vector (or one row).")
    if vectors.ndim != 2 or vectors.shape != (len(responses), len(origin)):
        raise ValueError("Scene vectors must have one row per response and the origin's width.")
    origin = normalized_vectors(origin[None])[0]
    vectors = normalized_vectors(vectors)
    matrix = np.clip(vectors @ vectors.T, -1, 1).astype(np.float32)
    origin_cosines = np.clip(vectors @ origin, -1, 1)
    identities, trials = {}, {}
    for index, row in enumerate(responses):
        step = row.get("step_index")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("Every scene response needs a nonnegative integer step_index.")
        key = (str(row["trial_id"]), int(step))
        if key in identities:
            raise ValueError(f"Duplicate trial/step identity: {key}")
        identities[key] = index
        trials.setdefault(key[0], []).append(index)

    def finite(value):
        value = float(value)
        return value if math.isfinite(value) else None

    result = []
    for index, row in enumerate(responses):
        trial, step = str(row["trial_id"]), int(row["step_index"])
        previous_index = identities.get((trial, step - 1))
        previous_kind = "origin" if step == 0 else "response" if previous_index is not None else "missing"
        previous_cosine = (origin_cosines[index] if step == 0 else
                           matrix[index, previous_index] if previous_index is not None else np.nan)
        # Include the fixed origin among previous scenes. Stable ties select the
        # origin, then the earliest observed prior step; this is descriptive.
        candidates = [-1] + sorted((i for i in trials[trial] if int(responses[i]["step_index"]) < step),
                                    key=lambda i: int(responses[i]["step_index"]))
        candidate_values = [origin_cosines[index] if i == -1 else matrix[index, i] for i in candidates]
        valid_candidates = [(i, float(v)) for i, v in zip(candidates, candidate_values) if np.isfinite(v)]
        best_index, best_value = max(valid_candidates, key=lambda item: item[1]) if valid_candidates else (None, None)
        result.append({**row, "origin_cosine": finite(origin_cosines[index]),
            "previous_scene_cosine": finite(previous_cosine), "previous_scene_kind": previous_kind,
            "previous_response_index": previous_index,
            "previous_scene_used_as_context": (True if step == 0 else row.get("condition") == "live") if previous_kind != "missing" else None,
            "previous_response_eligible": (bool(responses[previous_index].get("eligible"))
                                           if previous_index is not None else None),
            "nearest_prior_scene_index": best_index, "nearest_prior_scene_cosine": best_value})
    return result, matrix


def plot_imagination(experiment_dir):
    """Offline exploratory plots; no model, torch, MLX, fitting or imputation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(experiment_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "imagination_experiment":
        raise ValueError("plot_imagination expects kind='imagination_experiment'.")
    origin = json.loads((root / "origin.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / "responses.jsonl")
    analyzed = read_jsonl(root / "analysis" / "responses.jsonl")
    meta_path = root / "analysis" / "metadata.json"
    analysis = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    def key(row):
        return str(row["trial_id"]), int(row["step_index"])

    root_keys, analysis_keys = [key(r) for r in rows], [key(r) for r in analyzed]
    if len(set(root_keys)) != len(rows) or len(set(analysis_keys)) != len(analyzed):
        raise ValueError("Duplicate trial/step identity in scene observations or analysis.")
    if set(analysis_keys) - set(root_keys):
        raise ValueError("Scene analysis refers to responses absent from the root index.")
    if (analysis.get("origin_sha256") and origin.get("sha256") and
            analysis["origin_sha256"] != origin["sha256"]):
        raise ValueError("Scene analysis and origin.json have different origin hashes.")
    by_key = {key(r): r for r in analyzed}
    for row in rows:
        saved = by_key.get(key(row))
        if saved and saved.get("response") != row.get("response"):
            raise ValueError("Scene analysis is stale: decoded responses differ from the root index.")
    # Root measurements are authoritative; analysis adds derived fields only.
    rows = [{**by_key.get(key(r), {}), **r} for r in rows]
    n = len(rows)
    matrix = None
    matrix_path = root / "analysis" / "similarity.npy"
    if matrix_path.exists():
        loaded = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if loaded.shape != (len(analyzed), len(analyzed)):
            raise ValueError("Scene cosine matrix must match analysis row count on both axes.")
        finite = loaded[np.isfinite(loaded)]
        if finite.size and (finite.min() < -1.0001 or finite.max() > 1.0001):
            raise ValueError("Scene cosine matrix contains values outside [-1, 1].")
        if not np.allclose(loaded, loaded.T, equal_nan=True, rtol=1e-5, atol=1e-5):
            raise ValueError("Scene cosine matrix must be symmetric.")
        matrix = np.full((n, n), np.nan, dtype=np.float32)
        locations = {k: i for i, k in enumerate(root_keys)}
        indices = [locations[k] for k in analysis_keys]
        matrix[np.ix_(indices, indices)] = loaded

    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else np.nan
        except (ValueError, TypeError, OverflowError):
            return np.nan

    def values(name):
        return np.asarray([number(r.get(name, r.get("generated_tokens") if name == "n_tokens" else None))
                           for r in rows], dtype=float)

    eligible = np.asarray([r.get("eligible") is True for r in rows], dtype=bool)
    censored = np.asarray([r.get("censored") is True or r.get("stop_reason") == "length" for r in rows], dtype=bool)
    trial_ids = list(dict.fromkeys(str(r["trial_id"]) for r in rows))
    trials = {trial: np.asarray(sorted((i for i, r in enumerate(rows) if str(r["trial_id"]) == trial),
                                      key=lambda i: int(rows[i]["step_index"])), dtype=int)
              for trial in trial_ids}
    seeds = sorted({str(r.get("sample_seed", "?")) for r in rows},
                   key=lambda value: (not value.lstrip("-").isdigit(), int(value) if value.lstrip("-").isdigit() else value))
    colors = plt.get_cmap("tab10" if len(seeds) <= 10 else "turbo")
    seed_indices = {seed: i for i, seed in enumerate(seeds)}

    def color(row):
        index = seed_indices[str(row.get("sample_seed", "?"))]
        return colors(index if len(seeds) <= 10 else index / max(1, len(seeds) - 1))

    def style(row):
        return "--" if row.get("condition") == "reset" else "-"

    label = str(analysis.get("label") or ("Similarity disabled" if analysis.get("method") == "none" else "Similarity unavailable"))
    if analysis.get("method") == "self" and "proxy" not in label.lower():
        label += " (proxy)"
    label = label[:120]
    synthetic = manifest.get("synthetic") is True or manifest.get("untrained") is True
    if isinstance(manifest.get("model"), dict):
        synthetic = synthetic or manifest["model"].get("synthetic") is True or manifest["model"].get("untrained") is True
    prefix = "UNTRAINED / SYNTHETIC DEMO | " if synthetic else ""
    summary = (f"{n} responses | {int(eligible.sum())} eligible | {int(censored.sum())} capped | "
               f"{len(trial_ids)} trials | {len(seeds)} distinct sample seeds")
    if manifest.get("status") not in ("complete", "completed"):
        summary += f" | status: {manifest.get('status', 'unknown')}"
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, filename):
        path = out / filename
        temporary = path.with_name(path.stem + ".tmp.png")
        fig.savefig(temporary, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        os.replace(temporary, path)
        paths.append(path)

    def empty(ax, text):
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#555555", fontsize=10)
        ax.set_axis_off()

    def decorate(axes):
        for ax in np.asarray(axes).flat:
            ax.grid(alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    def legend(fig):
        handles = [Line2D([], [], color="#555555", linestyle="-", label="live"),
                   Line2D([], [], color="#555555", linestyle="--", label="reset: origin restart"),
                   Line2D([], [], color="#555555", marker="o", linestyle="", label="eligible"),
                   Line2D([], [], color="#ae3434", marker="x", linestyle="", label="rule failure"),
                   Line2D([], [], color="#c27a25", marker="^", linestyle="", label="length cap")]
        if len(seeds) <= 10:
            handles += [Line2D([], [], color=color({"sample_seed": seed}), linewidth=2, label=f"seed {seed}") for seed in seeds]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
                   ncol=min(6, len(handles)), frameon=False, fontsize=8)

    def points(ax, x, y):
        good = np.isfinite(x) & np.isfinite(y)
        for mask, marker, fixed in ((good & eligible, "o", None),
                                    (good & ~eligible & ~censored, "x", "#ae3434"),
                                    (good & censored, "^", "#c27a25")):
            indices = np.flatnonzero(mask)
            if indices.size:
                ax.scatter(x[indices], y[indices], s=23, marker=marker,
                           c=[color(rows[i]) for i in indices] if fixed is None else fixed,
                           alpha=0.78, zorder=4)
        return int(good.sum())

    def series(ax, name, ylabel):
        y = values(name)
        x = values("step_index")
        for indices in trials.values():
            row = rows[int(indices[0])]
            # Missing recorded steps break the line rather than suggesting an
            # adjacent transition that was never observed.
            xx, yy = [], []
            for j, index in enumerate(indices):
                if j and x[index] != x[indices[j - 1]] + 1:
                    xx.append(np.nan)
                    yy.append(np.nan)
                xx.append(x[index])
                yy.append(y[index])
            ax.plot(xx, yy, color=color(row), linestyle=style(row), linewidth=0.85, alpha=0.65)
        if not points(ax, x, y):
            empty(ax, "No completed responses." if not n else "Metric unavailable for these responses.")
        ax.set_ylabel(ylabel)

    # Origins and predecessor comparisons are separate: the predecessor at step
    # 0 is the origin; later it is exactly the previous response in that trial.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    series(axes[0], "origin_cosine", "Cosine to fixed origin")
    series(axes[1], "previous_scene_cosine", "Cosine to preceding scene")
    axes[1].set_xlabel("Generated scene step (zero-based)")
    decorate(axes)
    fig.suptitle(prefix + "Scene sequence: origin and adjacent-output similarity\n" + label, fontsize=12)
    fig.text(0.5, 0.91, summary, ha="center", fontsize=8)
    legend(fig)
    fig.text(0.5, 0.006, "Step 0 predecessor = origin. Later reset comparisons concern adjacent outputs, not conditioning. Cosine axes are autoscaled.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 0.90))
    save(fig, "scene_path.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    for ax, similarity, timing, xlabel, ylabel in (
            (axes[0], "origin_cosine", "first_token_model_ms", "Cosine to fixed origin", "First predictor forward (ms)"),
            (axes[1], "previous_scene_cosine", "response_model_ms", "Cosine to preceding scene", "All predictor forwards (ms)")):
        if not points(ax, values(similarity), values(timing)):
            empty(ax, "No available cosine / timing pairs.")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    decorate(axes)
    fig.suptitle(prefix + "Measured model time and scene affinity\n" + label, fontsize=12)
    fig.text(0.5, 0.88, summary, ha="center", fontsize=8)
    legend(fig)
    fig.text(0.5, 0.006, "Instrumented timings include observation overhead; predictor sum excludes terminal consume. No pooled fit or causal claim.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.20, 1, 0.86))
    save(fig, "scene_latency.png")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, name, ylabel in zip(axes, ("first_token_model_ms", "response_model_ms", "past_tokens", "n_tokens"),
                               ("First predictor (ms)", "All predictors (ms)", "Previously cached tokens", "Generated tokens")):
        series(ax, name, ylabel)
    for indices in trials.values():
        axes[2].plot(values("step_index")[indices], values("new_prompt_tokens")[indices],
                     color=color(rows[int(indices[0])]), linestyle=":", marker="+", markersize=3, alpha=0.6)
    if n:
        axes[2].legend(handles=[Line2D([], [], color="#555555", linestyle="-", label="Past prefix (solid live / dashed reset)"),
                                Line2D([], [], color="#555555", linestyle=":", marker="+", label="New prompt tokens")],
                       fontsize=8, loc="best")
    axes[-1].set_xlabel("Generated scene step (zero-based)")
    decorate(axes)
    fig.suptitle(prefix + "Scene generation: time, retained context and response length", fontsize=12)
    fig.text(0.5, 0.95, summary, ha="center", fontsize=8)
    legend(fig)
    fig.text(0.5, 0.005, "Reset restarts from the same origin at every step. Reused seeds remain repeated measurements, not independent random draws.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save(fig, "scene_progress.png")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    finite = matrix[np.isfinite(matrix)] if matrix is not None else np.asarray([])
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (-1., 1.)
    if high - low < 1e-6:
        low, high = max(-1., low - 0.01), min(1., high + 0.01)
    fig, ax = plt.subplots(figsize=(10, 8))
    if not finite.size:
        empty(ax, "Response-to-response cosine matrix unavailable.")
    else:
        im = ax.imshow(np.ma.masked_invalid(matrix), interpolation="nearest", cmap=cmap, vmin=low, vmax=high)
        ticks = np.unique(np.linspace(0, n - 1, min(n, 28), dtype=int))
        ax.set_xticks(ticks, labels=[f"R{i}" for i in ticks], rotation=90, fontsize=7)
        ax.set_yticks(ticks, labels=[f"R{i}" for i in ticks], fontsize=7)
        for i in range(1, n):
            if rows[i]["trial_id"] != rows[i - 1]["trial_id"]:
                ax.axhline(i - .5, color="white", linewidth=.6)
                ax.axvline(i - .5, color="white", linewidth=.6)
        bad = np.flatnonzero(~eligible)
        if bad.size:
            ax.scatter(bad, bad, marker="x", s=16, color="#d44d4d", linewidths=.8)
        fig.colorbar(im, ax=ax, label="Cosine (observed color range)", shrink=.85)
        ax.set_xlabel("Response row in root responses.jsonl")
        ax.set_ylabel("Response row in root responses.jsonl")
    fig.suptitle(prefix + "All recorded scenes: pairwise representation cosine\n" + label, fontsize=12)
    fig.text(.5, .91, summary, ha="center", fontsize=8)
    fig.text(.5, .015, "White boundaries separate successive trial blocks; red diagonal crosses mark ineligible responses. Gray = missing. No dimensionality reduction.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .89))
    save(fig, "scene_similarity_matrix.png")

    # Per-trial origin+response matrices expose recurrence within each trajectory
    # without conflating different seeds or conditions. Every trial is emitted.
    if matrix is not None and trial_ids:
        for start in range(0, len(trial_ids), 12):
            selected = trial_ids[start:start + 12]
            columns = min(3, len(selected))
            lines = (len(selected) + columns - 1) // columns
            fig, axes = plt.subplots(lines, columns, figsize=(4.3 * columns, 4.2 * lines + 1.0), squeeze=False)
            for ax, trial in zip(axes.flat, selected):
                indices = trials[trial]
                count = len(indices)
                block = np.full((count + 1, count + 1), np.nan, dtype=np.float32)
                block[1:, 1:] = matrix[np.ix_(indices, indices)]
                affinities = values("origin_cosine")[indices]
                block[0, 1:] = block[1:, 0] = affinities
                if np.isfinite(affinities).any():
                    block[0, 0] = 1.0
                ax.imshow(np.ma.masked_invalid(block), interpolation="nearest", cmap=cmap, vmin=low, vmax=high)
                ticks = np.unique(np.linspace(0, count, min(count + 1, 13), dtype=int))
                labels = ["O" if i == 0 else f"S{rows[int(indices[i - 1])]['step_index']}" for i in ticks]
                ax.set_xticks(ticks, labels=labels, rotation=90, fontsize=7)
                ax.set_yticks(ticks, labels=labels, fontsize=7)
                first = rows[int(indices[0])]
                ax.set_title(f"{trial} | seed {first.get('sample_seed')} | {first.get('condition')}\n"
                             f"repeat {first.get('repeat_index', first.get('repeat', 0))}", fontsize=9)
                bad = np.flatnonzero(~eligible[indices]) + 1
                ax.scatter(bad, bad, marker="x", color="#d44d4d", s=17, linewidths=.8)
            for ax in axes.flat[len(selected):]:
                ax.set_axis_off()
            fig.suptitle(prefix + f"Within-trial recurrence | cosine range {low:.3f} to {high:.3f}", fontsize=12)
            fig.text(.5, .012, "O = fixed origin; S = zero-based scene step. Same color scale as global matrix. Red diagonal crosses = ineligible.",
                     ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, .93))
            save(fig, f"scene_recurrence_{start // 12:02d}.png")

    atomic_json(out / "index.json", {"kind": "imagination_experiment", "origin": origin,
        "similarity_label": label, "matrix_row_order": "root responses.jsonl",
        "response_ids": [{"row_id": f"R{i}", "row_index": i, "trial_id": row["trial_id"],
                          "step_index": row["step_index"], "event_id": row.get("event_id"),
                          "sample_seed": row.get("sample_seed"), "repeat_index": row.get("repeat_index", row.get("repeat")),
                          "condition": row.get("condition"), "response": row.get("response"),
                          "eligible": row.get("eligible"), "censored": bool(censored[i])} for i, row in enumerate(rows)],
        "plots": [path.name for path in paths]})
    return paths


def inspect_run(run_dir, event_id=0, token_index=0, top_n=15):
    """Inspect one exact saved predictor distribution without loading the model."""
    import json
    import numpy as np

    root, manifest, events = _read_plot_events(run_dir, event_id)
    item = events[0]
    t = int(token_index)
    if t < 0 or t >= len(item["rows"]):
        raise ValueError(f"token_index must be in [0, {len(item['rows']) - 1}].")
    if int(top_n) < 1:
        raise ValueError("top_n must be positive.")
    row = item["rows"][t]
    result = {"event_id": int(event_id), "token_index": t, "response": item["event"].get("response", item["meta"].get("response")), "token": row}
    entry = item["meta"].get("arrays", {}).get("raw_logits")
    if not entry:
        result["top_raw"] = row.get("top_raw", [])[:int(top_n)]
        result["distribution_source"] = "saved top_raw only; full logits were disabled"
        return result
    arr = np.load(item["directory"] / _plot_array_path(entry), mmap_mode="r", allow_pickle=False)
    logits = np.asarray(arr[t], dtype=np.float64)
    if np.isnan(logits).any() or np.isposinf(logits).any() or not np.isfinite(logits).any():
        raise ValueError("Saved raw logits contain NaN/+inf or an entirely masked distribution.")
    z = logits - logits.max()
    logp = z - np.log(np.exp(z).sum())
    k = min(int(top_n), len(logits))
    ids = np.argsort(-logits, kind="stable")[:k]
    vocab_path = root / "vocab.json"
    if not vocab_path.exists() and manifest.get("kind") == "bank_trial":
        vocab_path = root.parent.parent / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.exists() else {}
    if isinstance(vocab, dict) and "tokens" in vocab:
        vocab = vocab["tokens"]
    fallback = {int(p["id"]): p.get("piece", "") for p in row.get("top_raw", [])}

    def piece(i):
        if isinstance(vocab, list) and i < len(vocab):
            v = vocab[i]
            return v.get("piece", v.get("token", "")) if isinstance(v, dict) else v
        if isinstance(vocab, dict) and str(i) in vocab:
            v = vocab[str(i)]
            return v.get("piece", v.get("token", "")) if isinstance(v, dict) else v
        return fallback.get(i, "<not saved>")

    result["top_raw"] = [{"id": int(i), "piece": piece(int(i)), "logit": float(logits[i]) if np.isfinite(logits[i]) else None,
                          "masked": bool(np.isneginf(logits[i])), "p": float(np.exp(logp[i]))} for i in ids]
    result["distribution_source"] = "saved full raw logits"
    result["vocab_size"] = len(logits)
    return result


def add_model_arguments(parser):
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch", help="Inference engine; MLX uses mlx-lm natively")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--tokenizer", help="Tokenizer ID/path if different from model")
    parser.add_argument("--revision", help="Model/tokenizer commit or revision")
    parser.add_argument("--adapter", help="PEFT adapter (Torch) or native mlx-lm adapter (MLX)")
    parser.add_argument("--device", default="auto", help="Torch: auto/cpu/cuda/cuda:0/mps; MLX: auto/cpu/gpu")
    parser.add_argument("--device-map", choices=["auto"], help="Accelerate placement/offload")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="CUDA bitsandbytes NF4; optional dependencies")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")


def add_run_arguments(parser):
    parser.add_argument("--out", required=True, help="New run directory (never overwrite)")
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="association")
    parser.add_argument("--cue", action="append", help="Starting cue; repeat for independent groups")
    parser.add_argument("--cues-file", help="UTF-8 one cue per line")
    parser.add_argument("--steps", type=int, default=1, help="Responses per initial cue")
    parser.add_argument("--chain", action="store_true", help="Use the previous association as the next cue")
    parser.add_argument("--context", choices=["reset", "carry", "live"], help="Default: reset for association/free; live KV for imagination. carry re-prefills text.")
    parser.add_argument("--interactive", action="store_true", help="Accept manual next cues between responses")
    parser.add_argument("--template", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--prompt-file", help="Exact raw prefix file; appended cue, context reset only")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-new-tokens", type=int, help="Default 24 association / 128 scene or free")
    parser.add_argument("--max-context", type=int, default=4096, help="Prompt+generation budget; never silently truncate")
    parser.add_argument("--temperature", type=float, default=1.0, help="0=greedy, default=1")
    parser.add_argument("--top-k", type=int, default=0, help="0=full vocabulary")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-candidates", type=int, default=12)
    parser.add_argument("--no-stop-newline", action="store_true", help="Do not stop association at the first newline")
    parser.add_argument("--no-stop-sentence", action="store_true", help="Allow multiple sentences in run imagination; default stops at one punctuation boundary")
    parser.add_argument("--layers", default="sample", help="sample, all, none, or comma-separated zero-based indices (negative allowed)")
    parser.add_argument("--hook", action="append", help="Additional exact module path or glob; repeatable")
    parser.add_argument("--no-hooks", action="store_true")
    parser.add_argument("--hidden-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--no-logits", action="store_true", help="Disable full-vocabulary arrays, retain scalar stats")
    parser.add_argument("--save-policy-logits", action="store_true", help="Also save temperature/top-k/top-p processed scores")
    parser.add_argument("--no-cache", action="store_true", help="Recompute full prefixes for debugging/comparison")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cpu-threads", type=int)


def add_bank_arguments(parser):
    parser.add_argument("--bank-size", type=int, default=24)
    parser.add_argument("--bank-seed", type=int, default=2026)
    parser.add_argument("--bank-language", choices=["ja", "en"], default="ja", help="Bank generation and association instruction language")
    parser.add_argument("--bank-temperature", type=float, default=0.9)
    parser.add_argument("--bank-top-p", type=float, default=1.0)
    parser.add_argument("--bank-max-new-tokens", type=int, default=512)
    parser.add_argument("--bank-attempts", type=int, default=3, help="Bounded logged attempts, separate fresh cache each time")
    parser.add_argument("--bank-prompt", help="UTF-8 prompt file replacing the default bank request")


def add_experiment_arguments(parser):
    parser.add_argument("--out", required=True)
    parser.add_argument("--bank", help="Frozen bank.json or UTF-8 one-word-per-line file; omitted: generate once")
    add_bank_arguments(parser)
    parser.set_defaults(bank_language=None)  # explicit flag > frozen bank language > ja
    parser.add_argument("--seed", type=int, default=43, help="Starting response seed when --seeds is omitted")
    parser.add_argument("--seeds", help="Explicit comma-separated response seeds")
    parser.add_argument("--n-seeds", type=int, help="Consecutive response seeds starting at --seed; default 3")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat identical seeds; repeated measurements, not new random samples")
    parser.add_argument("--order", choices=["fixed", "shuffle"], default="fixed")
    parser.add_argument("--order-seeds", default="101", help="Comma-separated order seeds, shared across response seeds/conditions")
    parser.add_argument("--condition", choices=["live", "reset", "both"], default="live")
    parser.add_argument("--template", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=24, help="Cap per response; one word may need several tokens")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-candidates", type=int, default=12)
    parser.add_argument("--no-stop-newline", action="store_true")
    parser.add_argument("--layers", default="sample")
    parser.add_argument("--hook", action="append")
    parser.add_argument("--no-hooks", action="store_true")
    parser.add_argument("--hidden-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--no-logits", action="store_true")
    parser.add_argument("--save-policy-logits", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--similarity", choices=["self", "sentence-transformer", "none"], default="self")
    parser.add_argument("--embedding-model", help="SentenceTransformer model ID/directory; required for that method")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-revision")


def add_imagination_arguments(parser):
    parser.add_argument("--out", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scene", help="Initial scene text, frozen and replayed for every trial")
    source.add_argument("--scene-file", help="Saved origin.json or UTF-8 text; omitted scene: model generates origin once")
    parser.add_argument("--language", choices=["ja", "en"], help="Instruction language: explicit > saved origin > ja")
    parser.add_argument("--origin-seed", type=int, default=2026)
    parser.add_argument("--origin-temperature", type=float, default=0.9)
    parser.add_argument("--origin-top-p", type=float, default=1.0)
    parser.add_argument("--origin-max-new-tokens", type=int, default=128)
    parser.add_argument("--origin-attempts", type=int, default=3)
    parser.add_argument("--origin-prompt", help="UTF-8 prompt file for model-generated origin only")
    parser.add_argument("--steps", type=int, default=8, help="One sentence per turn, per trial")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--seeds", help="Comma-separated unique response seeds")
    parser.add_argument("--n-seeds", type=int, help="Consecutive response seeds from --seed; default 3")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--condition", choices=["live", "reset", "both"], default="live",
                        help="live=continuous scene chain; reset=independent SAME-origin restart at every step")
    parser.add_argument("--template", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Cap per sentence; punctuation stops earlier")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-candidates", type=int, default=12)
    parser.add_argument("--layers", default="sample")
    parser.add_argument("--hook", action="append")
    parser.add_argument("--no-hooks", action="store_true")
    parser.add_argument("--hidden-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--no-logits", action="store_true")
    parser.add_argument("--save-policy-logits", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--similarity", choices=["self", "sentence-transformer", "none"], default="self")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-revision")
    parser.set_defaults(no_stop_newline=True)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Load a local/HF causal LM and record associations or scenes")
    add_model_arguments(run)
    add_run_arguments(run)
    bank = commands.add_parser("bank", help="Generate and freeze a reusable stimulus bank, then release its KV cache")
    add_model_arguments(bank)
    add_bank_arguments(bank)
    bank.add_argument("--out", required=True, help="New bank directory containing bank.json and generation traces")
    bank.add_argument("--template", choices=["auto", "plain", "chat"], default="auto")
    bank.add_argument("--max-context", type=int, default=4096)
    bank.add_argument("--cpu-threads", type=int)
    experiment = commands.add_parser("experiment", help="One fixed bank, repeated seeds/orders, persistent or reset KV, all-bank similarity")
    add_model_arguments(experiment)
    add_experiment_arguments(experiment)
    imagine = commands.add_parser("imagine", help="Frozen origin, fresh trial cache, one-sentence live scene chains across seeds")
    add_model_arguments(imagine)
    add_imagination_arguments(imagine)
    demo = commands.add_parser("demo", help="Offline random tiny model: recorder/plot smoke test, no language findings")
    add_run_arguments(demo)
    demo.add_argument("--architecture", choices=["llama", "gpt2"], default="llama")
    demo.set_defaults(max_new_tokens=12, steps=3, cpu_threads=1, max_context=2048, protocol="free",
                      cue=["a mirror"], no_stop_newline=True, temperature=0.8)
    plot = commands.add_parser("plot", help="Replot saved traces without torch/model weights")
    plot.add_argument("run_dir")
    plot.add_argument("--event", type=int)
    summary = commands.add_parser("inspect", help="Inspect full raw logits for one generation position")
    summary.add_argument("run_dir")
    summary.add_argument("--event", type=int, default=0)
    summary.add_argument("--token", type=int, default=0)
    summary.add_argument("--top", type=int, default=20)
    modules = commands.add_parser("modules", help="List actual module paths for hook selection")
    add_model_arguments(modules)
    modules.add_argument("--match", default="*")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_session(args)
        elif args.command == "demo":
            run_session(args, synthetic=True)
        elif args.command == "bank":
            run_bank(args)
        elif args.command == "experiment":
            run_experiment(args)
        elif args.command == "imagine":
            run_imagination(args)
        elif args.command == "plot":
            manifest = json.loads((Path(args.run_dir) / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("kind") in ("bank_experiment", "imagination_experiment") and args.event is not None:
                raise ValueError("Use plot EXPERIMENT/trials/t0000 --event N for a token-level event plot.")
            if manifest.get("kind") == "imagination_experiment":
                paths = plot_imagination(Path(args.run_dir))
            elif manifest.get("kind") == "bank_experiment":
                paths = plot_experiment(Path(args.run_dir))
            else:
                paths = plot_run(Path(args.run_dir), args.event)
            for path in paths:
                print(path)
        elif args.command == "inspect":
            print(json.dumps(inspect_run(Path(args.run_dir), args.event, args.token, args.top), ensure_ascii=False, indent=2))
        elif args.command == "modules":
            model, _ = load_model(args)
            modules = mlx_model_modules(model).items() if getattr(model, "_association_backend", None) == "mlx" else model.named_modules()
            for name, module in modules:
                if name and fnmatch.fnmatchcase(name, args.match):
                    print(f"{name}\t{type(module).__name__}")
    except KeyboardInterrupt:
        print("Interrupted; committed trace rows remain available.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
