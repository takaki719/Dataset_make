import os
from PIL import Image

# フォルダーのパスを指定
folder_path = '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/yolo_detected'  # ここにフォルダーのパスを入力

# 保存先フォルダーを指定
output_folder = '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/resize'
os.makedirs(output_folder, exist_ok=True)

# 画像のリサイズサイズを指定
new_size = (192, 256)

# フォルダー内のファイルをリスト
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 画像を開く
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        
        # 画像をリサイズ
        img_resized = img.resize(new_size)
        
        # リサイズした画像を保存
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

print("全ての画像がリサイズされました。")
