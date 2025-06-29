## 仮想試着用データセットの作成と画像選別プロセス
本プロジェクトでは、仮想試着システムの精度向上を目的として、実環境においても高精度な試着結果が得られるよう、多様な画像から学習に適した衣服画像を厳選するデータセット作成手法を提案・実装しています。

<img width="641" alt="image" src="https://github.com/user-attachments/assets/4ecaec13-4c33-40bb-9b8b-4bc9eb79c1dc" />

## 背景
  仮想試着システムは、ECサイトでの購入時に試着できないという課題を解決する技術ですが、従来の研究では理想的な環境下の画像（例：VTONデータセット）で学習されており、スマートフォン等で撮影された実画像には弱いという課題がありました。

## アプローチ概要
### 人物検出：
  YOLOv10を用いて、1人のみが写っている画像を抽出。

### 背景除去：
  PP-Mattingにより背景を取り除き、ノイズを低減。

### 姿勢推定：
  OpenPoseで肩や腰などのキーポイントを検出し、学習に不適な画像（例：側面向き、キーポイント欠損）を除外。

### 姿勢分類：
  自作アルゴリズムにより正面向きで衣服全体が確認できる画像を選別。

### 衣服抽出：
  CIHP_PGNベースのセグメンテーションにより衣服部分を抽出し、仮想試着用の衣服モデルを生成。

## 成果と目的
  このプロセスにより、実環境においても高品質な仮想試着を実現可能な画像データセットの構築を目指します。作成されたデータセットは、今後のモデル再学習と比較評価を通じて、手法の有効性を検証していきます。

