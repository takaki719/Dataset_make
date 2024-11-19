import os
import re
import shutil
import subprocess
import json
import cv2
import numpy as np
import time
from ultralytics import YOLO
from pathlib import Path
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageStat
import IPython
import os
import sys
import glob
from ACGPN.predict_pose import generate_pose_keypoints
import pathlib as Path

EXPERIMENT = "./test"

# フォルダパスの設定
input_folder = "/media/il/local2/Virtual_try_on/Preprocessing/Deepfashion/test/test/image"
#input_folder = "./input/fafafa"

output_folder = EXPERIMENT + "/output/back_ground1"
small_area_folder = EXPERIMENT + "/output/miss_file/small_area_images"
missing_body_parts_folder = EXPERIMENT + "/output/miss_file/missing_body_parts_images"
json_output_folder = EXPERIMENT + "/output/json"
alpha_folder = EXPERIMENT + "/output/miss_file/alpha_images"
yolo_output_folder = EXPERIMENT + "/output/yolo_detected"
multi_person_folder = EXPERIMENT + "/output/miss_file/multi_person_images"
missing_json_folder = EXPERIMENT + "/output/miss_file/missing_json"

classify_folder = EXPERIMENT + "/output/classify"

# 出力フォルダが存在しない場合は作成
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(small_area_folder, exist_ok=True)
os.makedirs(missing_body_parts_folder, exist_ok=True)   
os.makedirs(json_output_folder, exist_ok=True)
os.makedirs(alpha_folder, exist_ok=True)
os.makedirs(yolo_output_folder, exist_ok=True)
os.makedirs(multi_person_folder, exist_ok=True)
os.makedirs(classify_folder, exist_ok=True)
os.makedirs(missing_json_folder,exist_ok=True)

def back_ground(image_folder, output_folder):
    # YOLO model initialization
    model = YOLO("yolo11x-seg.pt")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all image files in the folder
    for image_name in os.listdir(image_folder):
        if image_name.endswith((".jpg", ".png")):
            image_path = os.path.join(image_folder, image_name)
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: {image_path}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_height, original_width = img.shape[:2]

            # Perform inference
            results = model(image_path)

            # Check if masks exist in the results
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()

                # Resize the masks to the original image size
                resized_masks = np.zeros((masks.shape[0], original_height, original_width))
                for i in range(masks.shape[0]):
                    resized_masks[i] = cv2.resize(masks[i], (original_width, original_height))

                # Use the mask of the first detected object
                mask = resized_masks[0]

                # Scale the mask to the 0-255 range and convert to uint8
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Apply the mask using bitwise AND
                mask_applied_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_uint8)

                # Save the image after applying the mask
                output_image_path = os.path.join(output_folder, f"{image_name}")
                cv2.imwrite(output_image_path, cv2.cvtColor(mask_applied_img, cv2.COLOR_RGB2BGR))
            else:
                print(f"No masks found for image: {image_path}")

    # Call the area process function if necessary
    area_process()
    #remove_low_quality_images("./experiment/output/yolo_detected", "./experiment/output/low_quality")
    #clean_and_rename_files()


def check_image_occupancy(image_path):
    try:
        # Load the image
        image = Image.open(image_path)
        width, height = image.size

        # Convert image to numpy array
        image_np = np.array(image)

        # Calculate the area of the image
        image_area = width * height

        # Extract non-black pixels (assuming the background is black)
        non_black_pixels = np.sum(np.all(image_np != [0, 0, 0], axis=-1))

        # Calculate the percentage of the image occupied by non-black pixels
        percentage_occupied = (non_black_pixels / image_area) * 100
        
        return percentage_occupied
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def area_process(source_folder = output_folder , destination_folder = small_area_folder):
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(source_folder, filename)
            occupancy = check_image_occupancy(image_path)
            if occupancy is not None and occupancy <= 15:
                dest_path = os.path.join(destination_folder, filename)
                shutil.move(image_path, dest_path)
                #print(f"Moved {filename} to {destination_folder}")
    
    print("Processing and moving images complete.")

# フォルダ内のすべての画像に対して予測を行い、ファイルを分類する関数の定
def is_blurry(image_path, threshold=200.0):
    """
    画像がぼやけているかどうかを判定する。
    :param image_path: 画像ファイルのパス
    :param threshold: ぼやけ度の閾値
    :return: ぼやけている場合はTrue、そうでない場合はFalse
    """
    image = cv2.imread(image_path)
    if image is None:
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def move_blurry_images(source_folder, destination_folder, threshold=200.0):
    """
    フォルダー内のぼやけた画像を別のフォルダーに移動する。
    :param source_folder: 元のフォルダーのパス
    :param destination_folder: 移動先のフォルダーのパス
    :param threshold: ぼやけ度の閾値
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, filename)
        destination_file_path = os.path.join(destination_folder, filename)
        
        if os.path.isfile(source_file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            if is_blurry(source_file_path, threshold):
                shutil.move(source_file_path, destination_file_path)
                print(f"Moved blurry image: {source_file_path} to {destination_file_path}")

REQUIRED_KEYPOINTS = {
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_hip': 11,
    'right_hip': 12,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10
}
# 信頼度の閾値
CONFIDENCE_THRESHOLD = 0.9

def check_keypoints(keypoints, required_keypoints, threshold):
    """
    必要なキーポイントがすべて検出されているかをチェックします。
    各キーポイントの信頼度が閾値以上である必要があります。
    """
    missing = []
    for name, idx in required_keypoints.items():
        if idx >= len(keypoints):
            missing.append(name)
            continue
        keypoint = keypoints[idx]
        if keypoint is None or keypoint[2] < threshold:
            missing.append(name)
    if missing:
        message = f"Missing keypoints: {', '.join(missing)}"
        return False, message
    return True, "All required keypoints are present."

def process_image_to_json(model, image_path, output_path):
    """
    画像を処理してキーポイントをJSONに保存します。
    """
    results = model(image_path)
    if not results:
        print(f"Failed to process {image_path}: No results from the model.")
        return False

    # 最初の検出結果を取得
    result = results[0]
    keypoints = result.keypoints  # 形状: (人数, キーポイント数, 3)

    # ここでは最初の人物のみを対象とします
    if len(keypoints.data) == 0:
        print(f"No keypoints detected in {image_path}")
        return False

    # キーポイントのデータをCPU上のNumPy配列に変換
    person_keypoints = keypoints.data[0].cpu().numpy().tolist()

    # キーポイントをチェック
    exists, message = check_keypoints(person_keypoints, REQUIRED_KEYPOINTS, CONFIDENCE_THRESHOLD)
    if not exists:
        print(f"Result: {message}")
        return False

    try:
        # JSONに保存
        with open(output_path, 'w') as f:
            json.dump({'keypoints': person_keypoints}, f, indent=4)
        print(f"Saved keypoints to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save JSON to {output_path}: {str(e)}")
        return False

def process_folder(input_folder, json_output_folder, failed_folder, model):
    """
    画像フォルダを処理し、キーポイントをJSONに変換し、失敗した画像を別フォルダに移動します。
    """
    if not os.path.exists(json_output_folder):
        os.makedirs(json_output_folder)

    if not os.path.exists(failed_folder):
        os.makedirs(failed_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png")):
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(json_output_folder, base_name + "_keypoints.json")

            print(f"Processing {input_path}...")
            success = process_image_to_json(model, input_path, output_path)
            if success:
                print(f"Saved result to {output_path}")
            else:
                print(f"Failed to process {input_path}. Moving to failed folder.")
                shutil.move(input_path, os.path.join(failed_folder, filename))

def process_json_folder(json_output_folder, input_folder, missing_body_parts_folder):
    """
    JSONフォルダを処理して、欠損しているキーポイントがある画像を別のフォルダに移動します。
    """
    if not os.path.exists(missing_body_parts_folder):
        os.makedirs(missing_body_parts_folder)

    # json_output_folder内のすべてのJSONファイルを処理
    for filename in os.listdir(json_output_folder):
        if filename.endswith("_keypoints.json"):  # JSONファイル名の形式をチェック
            json_path = os.path.join(json_output_folder, filename)
            with open(json_path, 'r') as f:
                keypoints_data = json.load(f)
            
            # キーポイントデータが存在するかどうかをチェック
            keypoints = keypoints_data.get('keypoints', [])
            exists, message = check_keypoints(keypoints, REQUIRED_KEYPOINTS, CONFIDENCE_THRESHOLD)
            
            if not exists:
                # JSONファイル名から画像ファイル名を生成
                base_name = filename.replace("_keypoints.json", "")
                
                # JPG, PNGの両方をチェック
                possible_extensions = [".jpg", ".png"]
                image_path = None
                for ext in possible_extensions:
                    potential_image_path = os.path.join(input_folder, base_name + ext)
                    if os.path.exists(potential_image_path):
                        image_path = potential_image_path
                        break

                if image_path:
                    # 画像を移動
                    print(f"Moving {image_path} to {missing_body_parts_folder}")
                    shutil.move(image_path, os.path.join(missing_body_parts_folder, os.path.basename(image_path)))
                    print(f"Moved {image_path} to {missing_body_parts_folder} due to missing keypoints: {message}")
                else:
                    print(f"Image file not found for {base_name}, skipping.")

    
def main():

    # YOLOv8-poseモデルをロード（事前にモデルをダウンロードしておく必要があります）
    model = YOLO("yolo11x-pose.pt")  # モデルパスを適宜変更してください

    # 画像フォルダを処理してJSONを生成
    #process_folder(output_folder, json_output_folder,missing_json_folder,model)

    # JSONフォルダを処理して欠損キーポイント画像を移動
    process_json_folder(json_output_folder, output_folder, missing_body_parts_folder)
    
def classify_and_crop_images(input_folder=input_folder, output_folder=yolo_output_folder, multi_person_folder=multi_person_folder, small_area_folder=small_area_folder, label_to_extract="person", threshold=0.4):
    # モデルのロード（YOLOv5）
    model = YOLO('yolo11x.pt')

    # フォルダ内の各画像を処理
    for img_filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_filename)

        # 画像を読み込み
        img = Image.open(img_path)
        img_width, img_height = img.size

        # オブジェクト検出の実行
        results = model(img_path)

        # 初期化
        person_count = 0
        max_area_ratio = 0

        # 検出結果を処理
        for result in results:
            for detection in result.boxes:
                label = result.names[int(detection.cls)]
                if label.lower() == label_to_extract:
                    person_count += 1
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = detection.xyxy[0].tolist()  # Tensorをリストに変換

                    # バウンディングボックスの面積比を計算
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    img_area = img_width * img_height
                    area_ratio = bbox_area / img_area
                    if area_ratio > max_area_ratio:
                        max_area_ratio = area_ratio

                    # 画像をクロップ
                    cropped_img = img.crop((x1, y1, x2, y2))
                    cropped_img_filename = f'{img_filename}'

                    # 条件に応じてクロップした画像を保存
                    if person_count >= 2:
                        cropped_img.save(os.path.join(multi_person_folder, cropped_img_filename))
                    elif person_count == 0:
                        cropped_img.save(os.path.join(multi_person_folder, cropped_img_filename))
                    elif max_area_ratio < threshold:
                        cropped_img.save(os.path.join(small_area_folder, cropped_img_filename))
                    else:
                        cropped_img.save(os.path.join(output_folder, cropped_img_filename))

def classify_and_crop_image(input_folder=input_folder, output_folder=yolo_output_folder, multi_person_folder=multi_person_folder, small_area_folder=small_area_folder, label_to_extract="person", threshold=0.3):
    # モデルのロード（YOLOv5）
    model = YOLO('yolo11x.pt')

    # フォルダ内の各画像を処理
    for img_filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_filename)

        # 画像を読み込み
        img = Image.open(img_path)
        img_width, img_height = img.size

        # オブジェクト検出の実行
        results = model(img_path)

        # 初期化
        person_count = 0
        max_area_ratio = 0

        # 検出結果を処理
        for result in results:
            for detection in result.boxes:
                label = result.names[int(detection.cls)]
                # 人間かどうかの判定
                if label.lower() == label_to_extract:
                    person_count += 1
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = detection.xyxy[0].tolist()

                    # バウンディングボックスの面積比を計算
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    img_area = img_width * img_height
                    area_ratio = bbox_area / img_area
                    if area_ratio > max_area_ratio:
                        max_area_ratio = area_ratio

                    # 画像をクロップ
                    cropped_img = img.crop((x1, y1, x2, y2))
                    cropped_img_filename = f'{img_filename}'

                    # 条件に応じてクロップした画像を保存
                    if person_count >= 2:
                        cropped_img.save(os.path.join(multi_person_folder, cropped_img_filename))
                    elif max_area_ratio < threshold:
                        cropped_img.save(os.path.join(small_area_folder, cropped_img_filename))
                    else:
                        cropped_img.save(os.path.join(output_folder, cropped_img_filename))
                else:
                    # 人間以外が映っている場合に保存
                    x1, y1, x2, y2 = detection.xyxy[0].tolist()
                    cropped_img = img.crop((x1, y1, x2, y2))
                    cropped_img_filename = f'{img_filename}'
                    cropped_img.save(os.path.join(multi_person_folder, cropped_img_filename))

def remove_similar_images_by_color(folder_path, reference_image_path, threshold=10):
    removed_images = []
    try:
        with Image.open(reference_image_path) as ref_img:
            ref_img = ref_img.resize((100, 100))
            ref_avg_color = np.array(ref_img).mean(axis=(0, 1))  # Calculate the average color of the reference image

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                with Image.open(img_path) as img:
                    img = img.resize((100, 100))
                    avg_color = np.array(img).mean(axis=(0, 1))
                    color_diff = np.linalg.norm(ref_avg_color - avg_color)  # Calculate color difference

                    if color_diff < threshold:  # If color difference is below threshold, remove the image
                        os.remove(img_path)
                        removed_images.append(filename)

        return removed_images
    except Exception as e:
        return str(e)

def classify(folder_path, output_folder):
    # モデルのロード
    model = tf.keras.models.load_model('image_classifier_model.keras')
    
    # クラスラベルの定義
    class_labels = {0: 'Not Front', 1: 'Front'}
    
    def preprocess_image(img_path):
        try:
            # 画像の読み込みとリサイズ
            img = image.load_img(img_path, target_size=(150, 150))
            # 画像を配列に変換
            img_array = image.img_to_array(img)
            # 次元を拡張してバッチサイズ1にする
            img_array = np.expand_dims(img_array, axis=0)
            # ピクセル値を0-1の範囲にスケーリング
            img_array /= 255.0
            return img_array
        except Exception as e:
            print(f"Error preprocessing image {img_path}: {e}")
            return None

    def classify_and_save_images(img_paths, output_folder):
        for img_path in img_paths:
            print(f"Processing {img_path}...")
            # 画像の前処理
            img_array = preprocess_image(img_path)
            if img_array is None:
                continue
            
            try:
                # 予測の実行
                prediction = model.predict(img_array)
                # 予測結果の解釈
                predicted_class = (prediction > 0.5).astype('int32')[0][0]
                # クラスラベルに基づいてフォルダ名を取得
                class_name = class_labels.get(predicted_class, 'Unknown')
                
                # 出力先フォルダのパスを作成
                class_folder = os.path.join(output_folder, class_name)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                # ファイル名を取得して新しいファイルパスを作成
                file_name = os.path.basename(img_path)
                destination = os.path.join(class_folder, file_name)

                # ファイルをコピー
                shutil.copy(img_path, destination)
                print(f"Image {file_name} is classified as: {class_name} and saved to {destination}")
            
            except Exception as e:
                print(f"Error classifying image {img_path}: {e}")

    def classify_images_in_folder_and_save(folder_path, output_folder):
        # 対応する画像拡張子を指定
        img_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        if not img_paths:
            print("No images found in the folder.")
        else:
            classify_and_save_images(img_paths, output_folder)

    classify_images_in_folder_and_save(folder_path, output_folder)


#62629

#remove_similar_images_by_color("./input/fafafa", "./experiment/output/miss_file/low_quality")
#classify_and_crop_image()
#back_ground(yolo_output_folder,output_folder)
main()
#classify(output_folder,classify_folder)
# 完了メッセージ
print("すべての画像処理と検出が完了しました。")
