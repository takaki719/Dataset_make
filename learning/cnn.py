import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 転移学習のためのベースモデル（VGG16）の読み込み
from tensorflow.keras.applications import VGG16
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
# コールバックの設定（早期終了とモデルチェックポイント）
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CSVファイルのロード
file_path = './test/updated_labels1.csv'
data = pd.read_csv(file_path)

# データの前処理
data['Label'] = data['Label'].astype(str)
data['ImagePath'] = data['ImagePath'].str.replace(
    '/media/il/local2/Virtual_try_on/Preprocessing/test/train/', 
    '/media/il/local2/Virtual_try_on/Preprocessing/experiment/output/back_ground/',
)
# 訓練データとテストデータに分割
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
print("Training data exists:", train_df['ImagePath'].apply(lambda x: os.path.exists(x)).value_counts())
print("Testing data exists:", test_df['ImagePath'].apply(lambda x: os.path.exists(x)).value_counts())
print("Unique labels in data:", data['Label'].unique())

# データが存在する行のみを使用
train_df = train_df[train_df['ImagePath'].apply(lambda x: os.path.exists(x))]
test_df = test_df[test_df['ImagePath'].apply(lambda x: os.path.exists(x))]

# 画像ジェネレーターの設定（データ拡張を含む）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    x_col='ImagePath', 
    y_col='Label', 
    target_size=(150, 150), 
    batch_size=20, 
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    x_col='ImagePath', 
    y_col='Label', 
    target_size=(150, 150), 
    batch_size=20, 
    class_mode='binary',
    shuffle=False
)

# steps_per_epoch と validation_steps の計算（切り上げ）
steps_per_epoch = int(np.ceil(train_generator.samples / train_generator.batch_size))
validation_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))

print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", validation_steps)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# ベースモデルの凍結
base_model.trainable = False

# モデルの構築
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])



early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# モデルの学習
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, model_checkpoint]
    
)

# ファインチューニングのためにベースモデルの一部を解凍
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# モデルの再コンパイル
model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# データジェネレーターをリセット
train_generator.reset()
test_generator.reset()

# steps_per_epoch と validation_steps の再計算（必要に応じて）
steps_per_epoch = int(np.ceil(train_generator.samples // train_generator.batch_size))
validation_steps = int(np.ceil(test_generator.samples // test_generator.batch_size))

# ファインチューニングの学習
history_fine = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, model_checkpoint]
)

# モデルの保存
model.save('image_classifier_model.keras')

# 損失値の可視化
plt.plot(history.history['loss'] + history_fine.history['loss'], label='train loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 精度の可視化
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# テストデータでの評価
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# 詳細な評価指標の表示
from sklearn.metrics import classification_report, confusion_matrix
# クラスインデックスの保存
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# 予測値の取得
test_generator.reset()
y_pred = model.predict(test_generator, steps=validation_steps)
y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)

# クラス名の取得
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
test_labels = test_generator.classes

# レポートの表示
print(classification_report(test_labels[:len(y_pred_classes)], y_pred_classes, target_names=labels.values()))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return 'Front' if prediction[0][0] > 0.5 else 'Not Front'

# 任意の画像に対して予測を実行
# img_path = 'path_to_your_image.jpg'
# print(predict_image(img_path))
