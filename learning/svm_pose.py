import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

# JSONファイルからキーポイントデータをロードしてフラットなベクトルに変換
def load_and_preprocess_json(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        if 'people' in data and len(data['people']) > 0:
            keypoints = data['people'][0]['pose_keypoints']
            keypoints = np.array(keypoints).reshape(-1, 3)
            # キーポイントのx, y座標を取得
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            confidences = keypoints[:, 2]

            # 信頼度が低いキーポイントをNaNに置換
            x_coords[confidences < 0.1] = np.nan
            y_coords[confidences < 0.1] = np.nan

            # 中心点（腰）の座標を取得
            if not np.isnan(x_coords[8]) and not np.isnan(y_coords[8]):
                center_x = x_coords[8]
                center_y = y_coords[8]
            else:
                # 腰のキーポイントがない場合、他のキーポイントの平均を使用
                center_x = np.nanmean(x_coords)
                center_y = np.nanmean(y_coords)

            # キーポイントを中心点に対して相対座標に変換
            x_coords = x_coords - center_x
            y_coords = y_coords - center_y

            # 欠損値をゼロで埋める
            x_coords = np.nan_to_num(x_coords)
            y_coords = np.nan_to_num(y_coords)

            # xとyの座標を結合
            keypoints = np.concatenate([x_coords, y_coords])

        else:
            # キーポイントが存在しない場合のデフォルト値
            keypoints = np.zeros(50)  # 25個のキーポイント * x, yの2要素
    except (json.JSONDecodeError, FileNotFoundError) as e:
        # ファイルが見つからないか、読み込めない場合のデフォルト値
        keypoints = np.zeros(50)
    return keypoints

# ファイルパスの設定
file_path = './test/updated_labels.csv'
data = pd.read_csv(file_path)

# 画像パスの変更
data['ImagePath'] = data['ImagePath'].str.replace(
    '/media/il/local2/Virtual_try_on/Preprocessing/test/train/', 
    '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/json/',
    regex=False
)
data['ImagePath'] = data['ImagePath'].str.replace('.jpg', '_keypoints.json', regex=False)

# ファイルの存在確認
data['Exists'] = data['ImagePath'].apply(lambda x: os.path.exists(x))
data = data[data['Exists']].drop(columns=['Exists'])

# データの前処理
data['Label'] = data['Label'].astype(str)  # 文字列に変換

# 特徴量とラベルの作成
features = np.array([load_and_preprocess_json(path) for path in data['ImagePath']])
labels = data['Label'].values

# 特徴量のスケーリング
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ラベルを数値に変換
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Stratified K-Fold クロスバリデーションの設定
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ハイパーパラメータの設定
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# モデルの定義
svm_model = SVC()

# グリッドサーチの設定
grid_search = GridSearchCV(svm_model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)

# モデルの訓練
grid_search.fit(features_scaled, labels_encoded)

# ベストモデルの取得
best_model = grid_search.best_estimator_

# クロスバリデーションの結果
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# テストデータでの評価
# データを再度分割
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# ベストモデルで再訓練
best_model.fit(X_train, y_train)

# テストデータでの予測
svm_predictions = best_model.predict(X_test)

# 精度の評価
svm_accuracy = accuracy_score(y_test, svm_predictions)
print('SVM Test accuracy:', svm_accuracy)

# 詳細な評価指標
print(classification_report(y_test, svm_predictions, target_names=label_encoder.classes_))

# 混同行列
conf_mat = confusion_matrix(y_test, svm_predictions)
print("Confusion Matrix:")
print(conf_mat)
