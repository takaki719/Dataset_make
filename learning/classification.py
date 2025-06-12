import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import glob
import os
# モデルのロード
model = tf.keras.models.load_model('image_classifier_model.keras')

import json
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}


def preprocess_image(img_path):
    # 画像の読み込みとリサイズ
    img = image.load_img(img_path, target_size=(150, 150))
    # 画像を配列に変換
    img_array = image.img_to_array(img)
    # 次元を拡張してバッチサイズ1にする
    img_array = np.expand_dims(img_array, axis=0)
    # ピクセル値を0-1の範囲にスケーリング
    img_array /= 255.0
    return img_array

def predict_images(img_paths):
    results = []
    for img_path in img_paths:
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class = (prediction > 0.5).astype('int32')[0][0]
        class_name = class_labels[str(predicted_class)]
        results.append((img_path, class_name))
    return results

# 画像パスのリスト
img_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# 一括予測の実行
predictions = predict_images(img_paths)

# 結果の表示
for img_path, class_name in predictions:
    print(f"Image {img_path} is classified as: {class_name}")

def predict_images_in_folder(folder_path):
    # 対応する画像拡張子を指定
    img_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    predictions = predict_images(img_paths)
    
    for img_path, class_name in predictions:
        print(f"Image {os.path.basename(img_path)} is classified as: {class_name}")

# フォルダのパスを指定
folder_path = 'path_to_your_folder'

# フォルダ内の画像を分類
predict_images_in_folder(folder_path)

