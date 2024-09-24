import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# CSVファイルのロード
file_path = './test/updated_labels.csv'
data = pd.read_csv(file_path)

# ラベルをエンコード
label_encoder = LabelEncoder()

# CSVファイルのロード
file_path = './test/updated_labels.csv'
data = pd.read_csv(file_path)

# ラベルをエンコード
data['Label'] = label_encoder.fit_transform(data['Label'])

# ファイルパスの置換
data['ImagePath'] = data['ImagePath'].str.replace(
    '/media/il/local2/Virtual_try_on/Preprocessing/test/train/',
    '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/back_ground/',
    regex=False
)

# ファイルの存在を確認
data['Exists'] = data['ImagePath'].apply(lambda x: os.path.exists(x))
data = data[data['Exists']]
data = data.drop(columns=['Exists'])

# データが空でないか確認
if len(data) == 0:
    print("DataFrame is empty after filtering. Please check the file paths and ensure that the images exist.")
    exit()

# ラベルとファイルパスの取得
labels = data['Label'].values
image_paths = data['ImagePath'].values

# 訓練データとテストデータに分割
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# 事前学習済みモデル（VGG16）の読み込み
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# 特徴量の抽出関数
def extract_features(img_paths):
    features = []
    for idx, img_path in enumerate(img_paths):
        try:
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            feature = base_model.predict(img_array)
            features.append(feature.flatten())

            if (idx + 1) % 100 == 0 or (idx + 1) == len(img_paths):
                print(f"Processed {idx+1}/{len(img_paths)} images")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return np.array(features)

# 訓練データの特徴量抽出
print("Extracting features from training data...")
train_features = extract_features(train_paths)
print("Shape of train_features:", train_features.shape)

# テストデータの特徴量抽出
print("Extracting features from testing data...")
test_features = extract_features(test_paths)
print("Shape of test_features:", test_features.shape)

# 特徴量のスケーリング
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# ランダムフォレストの訓練
print("Training Random Forest classifier...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_features, train_labels)

# テストデータでの評価
print("Evaluating classifier on test data...")
test_predictions = classifier.predict(test_features)

# クラス名を文字列に変換
class_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(test_labels, test_predictions, target_names=class_names))

# 任意の画像に対する予測関数
def predict_image_rf(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = base_model.predict(img_array)
        feature = scaler.transform([feature.flatten()])
        prediction = classifier.predict(feature)
        predicted_label = label_encoder.inverse_transform(prediction)
        return predicted_label[0]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# 任意の画像に対して予測を実行
# img_path = 'path_to_your_image.jpg'
# print(predict_image_rf(img_path))
