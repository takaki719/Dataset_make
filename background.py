import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# CSVファイルのロード
file_path = './test/updated_labels.csv'
data = pd.read_csv(file_path)

# ラベルをエンコード
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# ファイルパスの置換（正規表現を無効化）
data['ImagePath'] = data['ImagePath'].str.replace(
    '/media/il/local2/Virtual_try_on/Preprocessing/test/train/',
    '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/image/',
    regex=False
)

# ファイルの存在を確認し、存在するファイルのみを使用
data['Exists'] = data['ImagePath'].apply(lambda x: os.path.exists(x))
data = data[data['Exists']]
data = data.drop(columns=['Exists'])

# ラベルとファイルパスの取得
labels = data['Label'].values
image_paths = data['ImagePath'].values

# 訓練データとテストデータに分割
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# 事前学習済みモデル（VGG16）の読み込み（特徴抽出用）
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# 特徴量の抽出関数
def extract_features(img_paths):
    features = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = base_model.predict(img_array)
        features.append(feature.flatten())
    return np.array(features)

# 訓練データの特徴量抽出
print("Extracting features from training data...")
train_features = extract_features(train_paths)

# テストデータの特徴量抽出
print("Extracting features from testing data...")
test_features = extract_features(test_paths)

# 特徴量のスケーリング
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# SVMの訓練
print("Training SVM classifier...")
classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(train_features, train_labels)

# テストデータでの評価
print("Evaluating classifier on test data...")
test_predictions = classifier.predict(test_features)
print(classification_report(test_labels, test_predictions, target_names=label_encoder.classes_))

# 任意の画像に対する予測関数
def predict_image_svm(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature = base_model.predict(img_array)
    feature = scaler.transform([feature.flatten()])
    prediction = classifier.predict(feature)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# 任意の画像に対して予測を実行
# img_path = 'path_to_your_image.jpg'
# print(predict_image_svm(img_path))
