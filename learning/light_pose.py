import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import early_stopping
import joblib  # モデルの保存用

# キーポイントデータをロードしてフラットなベクトルに変換する関数
def load_and_preprocess_json(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        if 'people' in data and len(data['people']) > 0:
            person_data = data['people'][0]
            # キーポイントのキーを確認
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

# 特徴量のスケーリング（必要に応じて）
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# test_features = scaler.transform(test_features)

# LightGBM用のデータセットを作成
train_data = lgb.Dataset(train_features, label=train_labels)
test_data = lgb.Dataset(test_features, label=test_labels, reference=train_data)

# パラメータの設定
params = {
    'objective': 'binary',  # クラス数が2以上の場合は 'multiclass' に変更し、 'num_class' を設定
    'metric': 'binary_logloss',  # クラス数が2以上の場合は 'multi_logloss' に変更
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1
}

# 早期停止のコールバックを設定
callbacks = [early_stopping(stopping_rounds=10)]

# モデルの訓練
print("Training LightGBM model...")
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    callbacks=callbacks
)

# テストデータでの予測
lgb_predictions = lgb_model.predict(test_features, num_iteration=lgb_model.best_iteration)
lgb_predictions_binary = (lgb_predictions > 0.5).astype(int)

# 精度の評価
lgb_accuracy = accuracy_score(test_labels, lgb_predictions_binary)
print('LightGBM Test accuracy:', lgb_accuracy)

# 詳細な評価指標の表示
print('\nClassification Report:')
print(classification_report(test_labels, lgb_predictions_binary, target_names=label_encoder.classes_))

print('\nConfusion Matrix:')
print(confusion_matrix(test_labels, lgb_predictions_binary))

# モデルの保存
model_path = 'lightgbm_model.txt'
lgb_model.save_model(model_path)
print(f"Model saved to {model_path}")

# 必要に応じてラベルエンコーダーも保存
label_encoder_path = 'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_path)
print(f"Label encoder saved to {label_encoder_path}")

# 新しいデータに対する予測関数の定義
def predict_from_json(json_path):
    keypoints = load_and_preprocess_json(json_path)
    if keypoints is not None:
        keypoints = keypoints.reshape(1, -1)
        # keypoints = scaler.transform(keypoints)  # スケーリングを使用した場合はコメントを解除
        prediction_proba = lgb_model.predict(keypoints, num_iteration=lgb_model.best_iteration)
        prediction = (prediction_proba > 0.5).astype(int)
        predicted_label = label_encoder.inverse_transform(prediction)
        return predicted_label[0]
    else:
        print("Keypoints could not be extracted from the JSON file.")
        return None

# 予測の例
# json_path = '/path/to/new/json_file_keypoints.json'
# predicted_label = predict_from_json(json_path)
# print(f"Predicted label: {predicted_label}")
