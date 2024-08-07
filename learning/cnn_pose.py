import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# CSVファイルのロード
file_path = './test/updated_labels.csv'
data = pd.read_csv(file_path)

# データの前処理
data['Label'] = data['Label'].astype(str)  # 文字列に変換
data['ImagePath'] = data['ImagePath'].str.replace(
    '/media/il/local2/Virtual_try_on/Preprocessing/test/train/', 
    '/media/il/local2/Virtual_try_on/Preprocessing/output/image/'
)
data['ImagePath'] = data['ImagePath'].str.replace('.jpg', '_rendered.png')


# 訓練データとテストデータに分割
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 画像ジェネレーターの設定
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    x_col='ImagePath', 
    y_col='Label', 
    target_size=(150, 150), 
    batch_size=20, 
    class_mode='binary',
    shuffle=True  # シャッフルを有効にする
)

test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    x_col='ImagePath', 
    y_col='Label', 
    target_size=(150, 150), 
    batch_size=20, 
    class_mode='binary',
    shuffle=True  # シャッフルを有効にする
)

# steps_per_epoch と validation_steps の計算
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = test_generator.samples // test_generator.batch_size

# モデルの構築
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# モデルの学習
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=15,
    validation_data=test_generator,
    validation_steps=validation_steps
)

# モデルの保存
model.save('image_classifier_model.h5')

# 損失値の可視化
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 精度の可視化
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# テストデータでの評価
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return '1' if prediction[0][0] > 0.5 else '0'

# 任意の画像に対して予測を実行
# img_path = 'path_to_your_image.jpg'
# print(predict_image(img_path))
