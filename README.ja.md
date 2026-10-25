# sr_association_lab

[English](README.md) · [記録形式](docs/SCHEMA.md) · [検証](docs/VALIDATION.md) · [MITライセンス](LICENSE)

ローカル言語モデルが、一つの連想や情景から次へ移る過程を観察するための実験ツールです。
一語ずつの連想と、一文ずつ続く情景を生成し、全文・token IDs・raw logits・選択した内部状態・時間を一緒に保存します。
**MLX** と **PyTorch / Transformers** が同じ記録形式を使います。

起点の情景や単語バンクを固定し、seedを変えて複数のtrialを実行します。
KV cacheを継続する条件と、毎回同じ起点から始める条件を比較できます。
実行コードは [`sr_association_lab.py`](sr_association_lab.py) 一つです。
サーバー、外部の採点モデル、APIキーは不要です。

## セットアップ

Python 3.11以降を使い、推論backendを選びます。

```bash
git clone https://github.com/RyoSpiralArchitect/sr_association_lab.git
cd sr_association_lab
python3 -m venv .venv
source .venv/bin/activate
```

Apple SiliconでMLXを使う場合:

```bash
python -m pip install -r requirements-mlx.txt
```

PyTorchを使う場合は、[実行先に合うPyTorch](https://pytorch.org/get-started/locally/)を入れてから:

```bash
python -m pip install -r requirements.txt
```

PyTorchのdeviceは `cpu`、`cuda`、`mps`、MLXは `cpu`、`gpu`（Apple SiliconのMetal）です。
両方で `auto` も指定できます。backendの既定値は `torch` です。
モデルIDを指定するとHugging Faceからファイルを取得する場合があります。
手元のファイルだけを使うときはローカルパスと `--local-files-only` を指定します。

## 一つの起点から情景を続ける

`--model` にbackendが読み込めるモデルを指定します。MLXとローカルGPT-2での例:

```bash
python sr_association_lab.py imagine \
  --backend mlx \
  --model ./models/gpt2 \
  --device gpu \
  --scene 'At a night station, there was a light on the ridge of the platform,' \
  --language en \
  --steps 8 \
  --n-seeds 4 --seed 43 \
  --temperature 0.9 --top-p 0.95 \
  --max-new-tokens 96 \
  --local-files-only \
  --out runs/night-station
```

起点を `origin.json` に固定し、各seedを新しいcacheから開始します。
`live` のtrial内では、そこまでに生成したtoken IDsとKVをそのまま保持し、
継続を求めるcueだけを追加して、次の一文を生成します。

- `--scene-file` は保存した `origin.json` またはUTF-8の情景テキストを読み込みます。
- 起点指定を省略すると、モデルが一度だけ起点を生成します。その生成cacheを破棄してからtrialを開始します。
- `--condition both` は `live` と `reset` を比較します。`reset` は毎回、**同じ固定起点**から始めます。
- `--language ja` は日本語の指示を使います。実際に出力できる言語や品質はモデルに依存します。
- `--seeds 43,44,45,46` でseedを列挙できます。`--repeats` は同じseedの反復測定です。

一文の停止条件は句読点に基づくheuristicです。tokenが文境界を越えた場合も、
生成したtokenと全文を保存します。空の応答、形式に合わない応答、長さ打ち切りも履歴に残ります。

手動で働きかける場合は、同じモデル・backend・出力先の指定に
`run --protocol imagination --interactive` を組み合わせます。
`imagine` と `run` は引数が異なるので、それぞれの `--help` を参照してください。

## 単語バンクの連想を比較する

`bank` でバンクを生成・保存するか、一行一語のテキストを用意します。
例えば `words.txt` に `moon`、`door`、`water` を一行ずつ保存して:

```bash
python sr_association_lab.py experiment \
  --backend torch --model ./models/local-model --device cpu \
  --bank words.txt --bank-language en \
  --seeds 43,44,45 --condition both \
  --order shuffle --order-seeds 101,102 \
  --max-new-tokens 24 --local-files-only \
  --out runs/word-associations
```

バンクの語と順序はchecksum付きで固定します。提示順と応答の乱数は独立です。
応答用のNumPy RNGはtrial内で継続し、一語ごとに初期化しません。
一語の適合判定は表面上の簡易検査で、日本語の形態素解析ではありません。

## 記録を見る

```bash
python sr_association_lab.py plot runs/night-station
python sr_association_lab.py plot runs/night-station/trials/t0000
python sr_association_lab.py inspect runs/night-station/trials/t0000 \
  --event 0 --token 0 --top 20
```

`plot` と `inspect` は保存済みの記録を読み、モデルの重みをロードしません。

| 出力 | 内容 |
| --- | --- |
| `manifest.json` | 設定、ソースhash、モデル・環境、実行状態 |
| `origin.json` / `bank.json` | 固定した刺激とchecksum |
| `schedule.json` | seed、反復、条件、提示順 |
| `transcript.md` / `responses.jsonl` | 応答の全文と適合判定 |
| `trials/*/events/*/` | token記録、logits、内部状態、metadata |
| `trials/*/session.json` | 消費済みtokenの正確な列とsession状態 |
| `analysis/` | 埋め込み、類似度、CSV/JSON集計 |
| `plots/` | 情景の推移、再帰的な類似、時間、文脈長、ラベル対応 |

**N token生成したeventにはN+1行の有効記録があります。**
行0はprompt末尾の状態です。行 `t` はtoken `t` を予測し、行 `t+1` がそのtokenを消費した状態です。
最後の行では最後のtokenを一度だけ取り込み、追加のsamplingは行いません。
`meta.json.valid_rows` だけがデータで、配列の未使用領域は測定値ではありません。

samplingはCPU上のNumPy float64で、temperature → top-k → top-pの順です。
raw logitsとsampling処理後のscoreを区別し、同期したforward時間、CPU転送、sampling、保存時間を分けます。

`--similarity self` は生成完了後、各テキストを独立に入力したLM表現のcosineです。
意味類似度専用の指標とは区別します。外部encoderを使う場合は
`--similarity sentence-transformer --embedding-model PATH` と追加の依存ライブラリが必要です。
`--similarity none` で無効にできます。

### cosine以外から保存結果を読む

```bash
python sr_association_lab.py evaluate runs/night-station
# 任意: ローカルNLIモデルで文のペアを判定（PyTorchが必要）
python sr_association_lab.py evaluate runs/night-station \
  --nli-model ./models/nli-deberta --nli-device cpu \
  --out runs/night-station/evaluation-nli
```

`evaluation/report.html` でseedを切り替えながら本文を比較できます。生成はやり直しません。
完全反復、連続する3単位の反復、共有する語と元の文中の位置、任意のNLIによる
**含意・中立・矛盾候補**を分けて記録します。総合的な品質点は作らず、`eligible` も変更しません。

`--condition both` の結果はseed・反復番号・段階を揃え、初回を除いて差を集計します。
同じseedの反復を独立したseedとして数えません。resetの前の応答は文脈に使われていないことを明示します。
NLIの値は校正済みの正答確率ではなく、確認する文を探す補助です。新しい出来事は中立になり得ますし、
普通の状態変化が矛盾に見える場合もあります。

モデルは既定ではローカルからのみ読み、取得には `--allow-download` の明示が必要です。
Hubモデルは `--nli-revision` で固定できます。長すぎる文のペアは切り詰めず、判定不能として残します。
[評価方法と制約](docs/EVALUATION.md)も参照してください。

## 制約と読み方

- 記録するのはモデルの挙動とアクセスできる状態です。人間の連想や心理機構、意味的な品質を証明するものではありません。
- `eligible` は表面条件と長さ打ち切りの判定で、内容の品質点ではありません。
- 文脈上限は `--max-context` とモデル本来の上限の小さい方です。超過時は次のforward前に止め、履歴を勝手に削りません。
- 検証したGPT-2の上限は1024 tokenで、日本語の長い実験ではそこに達しました。必要に応じて段階数や各応答の上限を減らします。
- baseモデルは指示に十分従わない場合があります。生の文章を続ける実験には `run --prompt-file` と `--context reset` が使えます。
- 観測できる層はモデル構造に依存します。`modules` で実際のパスを確認し、`--layers`、`--hook`、`--no-hooks` で選択します。
- 全語彙logitsは大きくなります。`--no-logits` はtoken統計を残して配列保存を省き、`--no-plot` は自動描画を省きます。
- session保存はtoken列と状態の記録です。KV tensorを保存して別プロセスから再開する機能はありません。`--out` には新しいディレクトリを指定します。

## 開発と検証

```bash
python -m pip install pytest
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

テストは小型モデルとtokenizerをローカルで作り、学習済み重みをダウンロードしません。
依存がないbackendのテストはskipされます。MLXだけなら:

```bash
python -m pytest -q tests/test_mlx.py tests/test_compatibility.py
```

PyTorchを導入済みなら、未学習モデルで生成と描画を試せます:

```bash
python sr_association_lab.py demo --architecture gpt2 --out runs/demo
```

変更はGitのコミットと記録内のソースhashで追い、アプリのリリース番号は管理しません。
保存形式の識別子は互換性のために保持します。
[検証内容](docs/VALIDATION.md)と[開発への参加](CONTRIBUTING.md)も参照してください。

## ライセンス

[MIT](LICENSE)。モデルの重みと依存ライブラリには、それぞれのライセンスが適用されます。
