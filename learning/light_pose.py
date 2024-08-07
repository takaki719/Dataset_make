import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import early_stopping

# JSONファイルからキーポイントデータをロードしてフラットなベクトルに変換
def load_and_preprocess_json(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        if 'people' in data and len(data['people']) > 0:
            keypoints = data['people'][0]['pose_keypoints_2d']  # キーポイントデータを抽出
        else:
            keypoints = [0] * 75  # キーポイントが存在しない場合のデフォルト値（25個のキーポイント * x, y, confidence の3要素）
    except (json.JSONDecodeError, FileNotFoundError) as e:
        keypoints = [0] * 75  # ファイルが見つからないか、読み込めない場合のデフォルト値
    return np.array(keypoints)

# CSVファイルのロード
file_path = './test/updated_labels.csv'
data = pd.read_csv(file_path)

# 画像パスの変更
data['ImagePath'] = data['ImagePath'].str.replace(
    '/media/il/local2/Virtual_try_on/Preprocessing/test/train/', 
    '/media/il/local2/Virtual_try_on/Preprocessing/output/json/'
)
data['ImagePath'] = data['ImagePath'].str.replace('.jpg', '_keypoints.json')

# データの前処理
data['Label'] = data['Label'].astype(str)  # 文字列に変換

# 訓練データとテストデータに分割
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 特徴量とラベルの作成
train_features = np.array([load_and_preprocess_json(path) for path in train_df['ImagePath']])
test_features = np.array([load_and_preprocess_json(path) for path in test_df['ImagePath']])

# ラベルを数値に変換
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['Label'])
test_labels = label_encoder.transform(test_df['Label'])

# LightGBM用のデータセットを作成
train_data = lgb.Dataset(train_features, label=train_labels)
test_data = lgb.Dataset(test_features, label=test_labels, reference=train_data)

# パラメータの設定
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 早期停止のコールバックを設定
callbacks = [early_stopping(stopping_rounds=10)]

# モデルの訓練
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=callbacks
)

# テストデータでの予測
lgb_predictions = lgb_model.predict(test_features)
lgb_predictions_binary = (lgb_predictions > 0.5).astype(int)

# 精度の評価
lgb_accuracy = accuracy_score(test_labels, lgb_predictions_binary)
print('LightGBM Test accuracy:', lgb_accuracy)
