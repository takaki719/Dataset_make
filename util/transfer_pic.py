import pandas as pd
import os
import shutil

# CSVファイルのパスを指定
csv_file_path = '/media/il/local2/Virtual_try_on/Preprocessing/test/labels.csv'  # CSVファイルのパスを適宜変更してください

# データを読み込む
data = pd.read_csv(csv_file_path)

# 画像のファイル名を取得し、拡張子をJPGに変更
data['Filename'] = data['ImagePath'].apply(lambda x: os.path.splitext(os.path.basename(x))[0] + '.jpg')

# 画像を収集する元のディレクトリを指定
source_dir = '/media/il/local2/Virtual_try_on/Preprocessing/input/test/test/image'  # 画像が保存されているディレクトリのパスを適宜変更してください

# 画像を集めるターゲットディレクトリを指定
target_dir = '/media/il/local2/Virtual_try_on/Preprocessing/input/fafafa'  # 集めたい画像を保存するディレクトリのパスを適宜変更してください
os.makedirs(target_dir, exist_ok=True)

# 見つからないファイル名のリストを作成
missing_files = []

# 指定されたファイル名の画像を元のディレクトリからターゲットディレクトリにコピー
for filename in data['Filename']:
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    
    # デバッグ情報を表示
    print(f'Checking file: {source_path}')
    
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        print(f'Copied file: {source_path}')
    else:
        missing_files.append(filename)
        print(f'File not found: {filename}')

# 結果を表示
collected_images = os.listdir(target_dir)
print(f'Collected images: {collected_images}')
print(f'Missing files: {missing_files}')
