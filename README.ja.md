# Scale-RAE for ComfyUI（日本語概要）

公式 **Scale-RAE**（[ZitengWangNYU/Scale-RAE](https://github.com/ZitengWangNYU/Scale-RAE)、論文
[arXiv:2601.16208](https://arxiv.org/abs/2601.16208)）による text-to-image を ComfyUI で実行するノードパックです。Qwen2.5-1.5B の言語モデルが
プロンプトを読み、2.4B の rectified-flow DiT が視覚エンコーダの特徴空間で画像を生成し、RAE デコーダが画素に戻します。すべて ComfyUI の
プロセス内で動き、ノードが何かをダウンロードすることはありません。

> **非公式**の統合です。Scale-RAE の著者（New York University）、Meta、Alibaba（Qwen）、Comfy Org とは関係がなく、承認も受けていません。
> 詳細は英語の [README.md](README.md) と [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。

## ノード

| ノード | 役割 |
| --- | --- |
| **Scale-RAE Loader** | `models/scale_rae` から generator とデコーダを選び、dtype・device・常駐の有無を指定 |
| **Scale-RAE Text to Image** | プロンプト・seed・guidance・ステップ数・バッチ数 → `IMAGE`（224 x 224）と JSON レポート |
| **Scale-RAE Unload** | 常駐モデルを解放 |

対応モデルは `nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL` と `nyu-visionx/webssl300m_decoder`（WebSSL 版、224 x 224 出力）です。
SigLIP-2 版にはまだ対応していません。

## 検証済みの範囲（Windows 11 / RTX 5090 / ComfyUI v0.38.0）

- 公式環境（Python 3.10、torch 2.8.0、transformers 4.37.0）で、公式 `inference/cli.py` の経路と **7 件中 7 件で画像も 256 x 1024 の
  latent もバイト単位で一致**しました。
- ComfyUI 環境（torch 2.14.1、transformers 5.20.1）の実キュー実行は、同じ環境のインプロセス実行と 7 件中 7 件で一致し、公式実行との
  差は画素あたり最大 1/255（PSNR 65 dB 以上）でした（ライブラリのバージョン差による丸め）。
- float32 / float16 / bfloat16、CPU、バッチ、`image_start` の 3 モード、guidance の 2 モード、テキストのみの応答、中断、Unload、
  keep_loaded off を実モデルで確認しました。
- クリーンインストール（新規 ComfyUI v0.38.0 + 新規 Python 3.12 環境 + このリポジトリの git clone + requirements + `tools/download_models.py`
  によるデコーダ取得と SHA-256 検証）でも、開発環境と 7 件中 7 件で同じ画像になり、同梱ワークフローがそのまま動きました。
- CI（GitHub Actions、Ubuntu / Windows、transformers 4.50.3 と最新、Python 3.10 / 3.12）では小さなランダムモデルで CPU テストと
  ComfyUI 実キューを実行し、同梱した公式コードが固定コミットとバイト単位で一致することも検査しています。

## インストール

1. ComfyUI（V3 ノード API、v0.38.0 で確認）、Python 3.10 以上、float32 なら空き VRAM 約 18 GB、float16 / bfloat16 なら約 10 GB の
   NVIDIA GPU（CPU でも動きますが遅い）。
2. このフォルダを `ComfyUI/custom_nodes/` に置き、`python -m pip install -r custom_nodes/ComfyUI-ScaleRAE/requirements.txt`。
3. モデル（合計約 17.6 GB、モデルカードはどちらも MIT）を取得し SHA-256 を検証:
   `python custom_nodes/ComfyUI-ScaleRAE/tools/download_models.py --models-dir ComfyUI/models/scale_rae`
4. ComfyUI を再起動し、`workflows/gui/`（または `workflows/api/`）のワークフローを読み込みます。

## 使い方の要点

- **プロンプト**: このモデルはチャットモデルです。「Can you generate a photo of ...?」「Generate an image of ...」のような**依頼文**だと
  自分で画像を開始します（公式の経路）。「A field of sunflowers」のような**説明文だけ**だとテキストで答えます（説明文 20 件中 0 件、
  依頼文 60 件中 60 件が画像を開始。seed には依存しません）。
- **image_start**: `auto`（既定）は公式の経路で、テキストのみの応答だったときだけ同じ seed でプロンプト直後から画像を開始します（レポートに
  `image_forced: true` とスキップしたテキストが残ります）。`model` は公式の挙動のみ（テキスト応答はエラー）、`force` は常にプロンプト直後
  から画像を開始します。
- **guidance / cfg_mode**: 既定の guidance 1.0 は公式の既定値で、論文の評価設定です。公開コードでは guidance > 1 が**効きません**
  （upstream issue #7）。`official-cli` はその挙動を再現し、`standard`（既定）は毎ステップ classifier-free guidance を適用します。
- **dtype**: `float32` は公式と一致。`float16` は VRAM が約半分で float32 に近い結果（PSNR 37〜56 dB）、`bfloat16` は同じメモリで差が
  大きめ（26〜33 dB）。RAE デコーダは常に float32 で動かします（float16 ではオーバーフローして全画像が同じ平坦な絵になったため）。
- **seed / batch_size**: 同じ条件・seed・環境なら同じ画像。バッチは seed, seed + 1, ... を使います。CPU と GPU では同じ seed でも初期
  ノイズが異なります。

## ライセンス

- このノードパックのコード: **MIT**（[LICENSE](LICENSE)、第三者表記は [NOTICE](NOTICE)）。
- `scale_rae_comfy/vendor/scale_rae`: 公式リポジトリのコード（**MIT**）。OpenAI の diffusion 実装・BAAI EVA-02（MIT）と Hugging Face
  ViTMAE（Apache-2.0）由来のコードを含みます。
- モデルの重みは**同梱していません**。generator とデコーダはモデルカード上 MIT です。デコーダは Meta の WebSSL エンコーダ
  （`facebook/webssl-dino300m-full2b-224`、重みは **CC-BY-NC-4.0**）の特徴空間から画像を再構成します。本パックはこのエンコーダを
  ダウンロードも実行もしませんが、その非商用条件がその特徴で学習したモデルに及ぶかどうかはモデルカードに書かれていません。商用利用の前に
  ご自身で確認してください。
