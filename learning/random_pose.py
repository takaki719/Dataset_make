import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# キーポイントデータをロードしてフラットなベクトルに変換する関数
def load_and_preprocess_json(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        if 'people' in data and len(data['people']) > 0:
            person_data = data['people'][0]
            # キーの存在を確認し、キーポイントを取得
            if 'pose_keypoints_2d' in person_data:
                keypoints = person_data['pose_keypoints_2d']
            elif 'pose_keypoints' in person_data:
                keypoints = person_data['pose_keypoints']
            else:
                print(f"Key 'pose_keypoints_2d' or 'pose_keypoints' not found in {json_path}")
                return None
            keypoints = np.array(keypoints)
            return keypoints
        else:
            print(f"No people detected in {json_path}")
            return None
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {json_path}: {e}")
        return None

# CSVファイルのロード
file_path = './test/updated_labels.csv'
data = pd.read_csv(file_path)

# 画像パスをJSONファイルのパスに変更
data['ImagePath'] = data['ImagePath'].str.replace(
    '/media/il/local2/Virtual_try_on/Preprocessing/test/train/', 
    '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/json/',
    regex=False
)
data['ImagePath'] = data['ImagePath'].str.replace('.jpg', '_keypoints.json', regex=False)

# ラベルを文字列に変換
data['Label'] = data['Label'].astype(str)

# 特徴量とラベルのリストを初期化
features = []
labels = []

# データフレームを反復処理し、特徴量とラベルを収集
for idx, row in data.iterrows():
    json_path = row['ImagePath']
    label = row['Label']
    keypoints = load_and_preprocess_json(json_path)
    if keypoints is not None:
        features.append(keypoints)
        labels.append(label)
    else:
        # キーポイントが取得できない場合、そのサンプルをスキップ
        pass

# リストをNumPy配列に変換
features = np.array(features)
labels = np.array(labels)

# 特徴量とラベルの数が一致するか確認
assert features.shape[0] == labels.shape[0], "Number of samples in features and labels do not match"

# ラベルを数値にエンコード
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 訓練データとテストデータに分割（ストラティファイドサンプリング）
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# 特徴量のスケーリング
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# ランダムフォレストモデルの訓練
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_features, train_labels)

# テストデータでの予測
rf_predictions = rf_model.predict(test_features)

# モデルの評価
rf_accuracy = accuracy_score(test_labels, rf_predictions)
print('Random Forest Test accuracy:', rf_accuracy)
print('\nClassification Report:')
print(classification_report(test_labels, rf_predictions, target_names=label_encoder.classes_))
print('\nConfusion Matrix:')
print(confusion_matrix(test_labels, rf_predictions))
