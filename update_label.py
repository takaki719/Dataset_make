import os
import pandas as pd

# CSVファイルを読み込む
file_path = "./test/labels.csv"
labels_df = pd.read_csv(file_path)

# 画像が保存されているフォルダーのパス
image_folder_path = '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/back_ground'

# CSVファイルのパスを確認
csv_image_paths = labels_df['ImagePath'].tolist()

# フォルダー内の画像ファイルをリスト化
folder_image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# フォルダー内の画像ファイル名（拡張子を除く）をリスト化
folder_image_names = [os.path.splitext(f)[0] for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

# CSVファイルの画像パスのファイル名（拡張子を除く）をリスト化
csv_image_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_image_paths]

# フォルダー内に存在する画像パスを特定
valid_image_paths = [path for path in csv_image_paths if os.path.splitext(os.path.basename(path))[0] in folder_image_names]

# 更新されたDataFrameを作成
updated_labels_df = labels_df[labels_df['ImagePath'].isin(valid_image_paths)].copy()

# 'png'を'jpg'に変換
updated_labels_df.loc[:, 'ImagePath'] = updated_labels_df['ImagePath'].str.replace('.png', '.jpg')

# 更新されたCSVファイルを保存
updated_labels_df.to_csv('./test/updated_labels1.csv', index=False)
