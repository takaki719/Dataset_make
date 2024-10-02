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
import mediapipe as mp
from ACGPN.predict_pose import generate_pose_keypoints
# フォルダパスの設定
#input_folder = "./input/test"
input_folder = "./input/fafafa"
output_folder = "./experiment/output/back_ground"
small_area_folder = "./experiment/output/miss_file/small_area_images"
missing_body_parts_folder = "./experiment/output/miss_file/missing_body_parts_images"
json_output_folder = "./experiment/output/json"
coco_output_folder = "./experiment/output/json_coco"
image_output_folder ="./experiment/output/image"
alpha_folder = "./experiment/output/miss_file/alpha_images"
openpose_models_folder = "./modules/openpose/models"
yolo_output_folder = "./experiment/output/yolo_detected"
multi_person_folder = "./experiment/output/miss_file/multi_person_images"

# 出力フォルダが存在しない場合は作成
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(small_area_folder, exist_ok=True)
os.makedirs(missing_body_parts_folder, exist_ok=True)
os.makedirs(json_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(alpha_folder, exist_ok=True)
os.makedirs(yolo_output_folder, exist_ok=True)
os.makedirs(multi_person_folder, exist_ok=True)

def back_ground():
    #remove_low_quality_images("./experiment/output/yolo_detected", "./experiment/output/low_quality")
    model = YOLO("yolov8x-seg.pt")

    #処理する画像が入っているフォルダのパス
    image_folder = "./experiment/output/yolo_detected"
    output_folder = "./experiment/output/back_ground"
    
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)
    
    # フォルダ内の全ての画像ファイルを処理
    for image_name in os.listdir(image_folder):
        if image_name.endswith((".jpg", ".png")):
            image_path = os.path.join(image_folder, image_name)
            
            # 画像を読み込む
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_height, original_width = img.shape[:2]

            # 推論の実行
            results = model(image_path)

            # マスクデータの取得（最初の結果を使用）
            masks = results[0].masks.data.cpu().numpy()

            # マスクを元の画像サイズにリサイズ
            resized_masks = np.zeros((masks.shape[0], original_height, original_width))
            for i in range(masks.shape[0]):
                resized_masks[i] = cv2.resize(masks[i], (original_width, original_height))

            # 最初のオブジェクトのマスクを使用
            mask = resized_masks[0]

            # マスクを0から255の範囲にスケーリングしてuint8に変換
            mask_uint8 = (mask * 255).astype(np.uint8)

            # ビット演算でマスクを適用
            mask_applied_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_uint8)

            # マスク適用後の画像を保存
            output_image_path = os.path.join(output_folder, f"{image_name}")
            cv2.imwrite(output_image_path, cv2.cvtColor(mask_applied_img, cv2.COLOR_RGB2BGR))

    area_process()
    #remove_low_quality_images("./experiment/output/yolo_detected", "./experiment/output/low_quality")
    #clean_and_rename_files()

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

def remove_low_quality_images(source_folder, low_quality_folder, threshold=200):
    """
    Move low quality images based on sharpness to a different folder.
    """
    os.makedirs(low_quality_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(source_folder, filename)
            try:
                with Image.open(image_path) as img:
                    sharpness = calculate_sharpness(img)
                    if sharpness < threshold:
                        dest_path = os.path.join(low_quality_folder, filename)
                        shutil.move(image_path, dest_path)
                        print(f"Moved low quality image: {filename} to {low_quality_folder}")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

def calculate_sharpness(image):
    """
    Calculate the sharpness of an image.
    """
    image = image.convert("L")  # Convert to grayscale
    edges = image.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    return stat.mean[0]  # Return the mean of the edge image as the sharpness metric

def clean_and_rename_files():
    for filename in os.listdir(output_folder):
        if filename.endswith("_alpha.png"):
            # _alpha.pngファイルを移行
            alpha_image_path = os.path.join(output_folder, filename)
            new_alpha_path = os.path.join(alpha_folder, filename)
            shutil.move(alpha_image_path, new_alpha_path)
        elif filename.endswith("_rgba.png"):
            # _rgba.pngファイルをリネーム
            num_match = re.search(r"(\d+)_rgba\.png", filename)
            if num_match:
                num = num_match.group(1)
                new_filename = f"{num}.png"
                rgba_image_path = os.path.join(output_folder, filename)
                new_image_path = os.path.join(output_folder, new_filename)
                os.rename(rgba_image_path, new_image_path)

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

def copy_with_retries(src, dst, retries=3, delay=1):
    """
    ファイルをコピーし、失敗した場合にリトライします。
    :param src: コピー元ファイルパス
    :param dst: コピー先ファイルパス
    :param retries: リトライ回数
    :param delay: リトライ間の待機時間（秒）
    """
    for attempt in range(retries):
        try:
            shutil.copy(src, dst)
            if os.path.exists(dst):
                return True
        except Exception as e:
            print(f"Error copying {src} to {dst}: {e}")
        time.sleep(delay)
    return False

def resize_image(image_path, output_path, target_size):
    try:
        with Image.open(image_path) as img:
            original_size = img.size
            img = img.resize(target_size, Image.LANCZOS)
            img.save(output_path)
        resize_ratio = (original_size[0] / target_size[0], original_size[1] / target_size[1])
    except Exception as e:
        print(f"Failed to resize image {image_path}: {e}")
        return False, None
    return True, resize_ratio

def openpose_done(image_dir):
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    # 各画像に対してポーズキーポイントを生成
    for img in images:
        img_filename = os.path.basename(img)
        pose_path = os.path.join('./experiment/output/json',img_filename.replace('.jpg','_keypoints.json'))
        print(img, pose_path)
        generate_pose_keypoints(img, pose_path)


def batch_process_images(image_dir, json_output_folder, image_output_folder, batch_size, openpose_models_folder):
    # 画像ファイルのリストを取得
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    # 出力フォルダーが存在しない場合は作成
    os.makedirs(json_output_folder, exist_ok=True)
    os.makedirs(image_output_folder, exist_ok=True)
    
    # 画像をバッチに分割して処理
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_folder = os.path.join(image_dir, f"batch_{i//batch_size + 1}")
        
        # バッチ用フォルダーを作成
        os.makedirs(batch_folder, exist_ok=True)
        
        # バッチ内の画像をバッチ用フォルダーにコピー
        for image in batch:
            destination = os.path.join(batch_folder, os.path.basename(image))
            if not copy_with_retries(image, destination):
                print(f"Failed to copy {image} to {destination} after multiple attempts.")
                continue
        
        # OpenPoseコマンドを作成
        openpose_cmd = [
            "./modules/openpose/build/examples/openpose/openpose.bin",
            "--image_dir", batch_folder,
            "--hand",
            "--disable_blending",
            "--display", "0",
            "--write_json", json_output_folder,
            "--write_images", image_output_folder,
            "--num_gpu", "1",
            "--num_gpu_start", "0",
            "--model_folder", openpose_models_folder,
            "--net_resolution", "-656x368"  # ネットワーク解像度を低く設定
        ]
        
        print(f"Processing batch {i//batch_size + 1}: {batch}")
        process = subprocess.Popen(openpose_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error in batch {i//batch_size + 1}: {stderr.decode('utf-8')}")
        
        # バッチ用フォルダーを削除
        shutil.rmtree(batch_folder)
        
        # GPUメモリのクリーンアップ
        subprocess.run(["nvidia-smi", "--gpu-reset"], stderr=subprocess.PIPE)

    
def openpose_process():
    openpose_done(yolo_output_folder)#output_folder -> yolo

    # JSONフォルダーを解析
    json_folder = json_output_folder

    for json_filename in os.listdir(json_folder):
        if json_filename.endswith("_keypoints.json"):
            json_path = os.path.join(json_folder, json_filename)

            with open(json_path, 'r') as f:
                data = json.load(f)

                if 'people' in data and len(data['people']) > 0:
                    person = data['people'][0]
                    keypoints = person['pose_keypoints']

                    # 右肩、左肩、腰のインデックス
                    right_shoulder_idx = 2 * 3  # 右肩
                    left_shoulder_idx = 5 * 3  # 左肩
                    hips_idx = 8 * 3  # 腰

                    keypoints_present = all([
                        keypoints[right_shoulder_idx] > 0 and keypoints[right_shoulder_idx + 1] > 0,
                        keypoints[left_shoulder_idx] > 0 and keypoints[left_shoulder_idx + 1] > 0,
                        keypoints[hips_idx] > 0 and keypoints[hips_idx + 1] > 0
                    ])

                    # 必要なキー・ポイントが不足している場合
                    if not keypoints_present:
                        corresponding_image = os.path.join(output_folder, json_filename.replace("_keypoints.json", ".jpg"))
                        missing_body_parts_path = os.path.join(missing_body_parts_folder, json_filename.replace("_keypoints.json", ".jpg"))
                        if os.path.exists(corresponding_image):
                            os.rename(corresponding_image, missing_body_parts_path)
                        else:
                            print(f"File not found: {corresponding_image}")
                else:
                    print(f"No people detected in: {json_filename}")
        else:
                print(f"Invalid JSON structure in: {json_filename}")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_image_to_json(image_path, output_json_path):
    # 画像の読み込み
    image = cv2.imread(image_path)
    
    # RGBに変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Poseインスタンスの作成
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # 骨格推定を実行
        results = pose.process(image_rgb)
        
        # JSONに書き込むデータ
        landmarks_data = []

        # 骨格が検出された場合の処理
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append({
                    'id': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            # JSONファイルに結果を保存
            with open(output_json_path, 'w') as json_file:
                json.dump(landmarks_data, json_file, indent=4)
        else:
            print(f"No landmarks found for {image_path}")

# フォルダ内の全ての画像に対して骨格推定を実行
def process_folder(input_folder, output_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # フォルダ内の全てのファイルを処理
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            # ファイル名の拡張子を.jpgや.pngから.jsonに変換
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, base_name + ".json")

            print(f"Processing {input_path}...")
            process_image_to_json(input_path, output_path)
            print(f"Saved result to {output_path}")

REQUIRED_LANDMARKS = [11, 12, 23, 24]

# JSONから両肩と腰のデータが存在するかを確認する関数
def check_keypoints_in_json(json_file_path):
    # JSONファイルを開く
    with open(json_file_path, 'r') as json_file:
        landmarks = json.load(json_file)
    
    # 必要なランドマークが存在するかを確認
    existing_landmark_ids = [landmark['id'] for landmark in landmarks]
    
    # 両肩と腰のランドマークが全て揃っているか確認
    for required_id in REQUIRED_LANDMARKS:
        if required_id not in existing_landmark_ids:
            return False, f"Landmark ID {required_id} is missing"
    
    return True, "All required landmarks are present"

# フォルダ内の全てのJSONファイルをチェック
def process_json_folder(input_folder, output_folder, missing_body_parts_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(missing_body_parts_folder):
        os.makedirs(missing_body_parts_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_file_path = os.path.join(input_folder, filename)
            exists, message = check_keypoints_in_json(json_file_path)
            print(f"Result: {message}")

            # 必要なキーポイントが不足している場合、画像を別のフォルダに移動
            if not exists:
                corresponding_image = os.path.join(output_folder, filename.replace("_keypoints.json", ".jpg"))
                missing_body_parts_path = os.path.join(missing_body_parts_folder, filename.replace("_keypoints.json", ".jpg"))

                # 対応する画像が存在するかを確認して移動
                if os.path.exists(corresponding_image):
                    os.rename(corresponding_image, missing_body_parts_path)

model = load_model('image_classifier_model.h5')

# 予測関数の定義
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return '1' if prediction[0][0] > 0.5 else '0'

# フォルダ内のすべての画像に対して予測を行い、ファイルを分類する関数の定義
def classify_images_in_folder(source_folder= output_folder, dest_folder_1 =  "./test/result1" , dest_folder_0 =  "./test/result0"):
    # 各分類結果のフォルダが存在しない場合は作成
    if not os.path.exists(dest_folder_1):
        os.makedirs(dest_folder_1)
    if not os.path.exists(dest_folder_0):
        os.makedirs(dest_folder_0)
    
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(source_folder, filename)
            prediction = predict_image(img_path)
            if prediction == '1':
                shutil.copy(img_path, os.path.join(dest_folder_1, filename))
            else:
                shutil.copy(img_path, os.path.join(dest_folder_0, filename))

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



def classify_and_crop_images(input_folder=input_folder, output_folder=yolo_output_folder, multi_person_folder=multi_person_folder, small_area_folder=small_area_folder, label_to_extract="person", threshold=0.2):
    # モデルのロード（YOLOv5）
    model = YOLO('yolov10x.pt')

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
                    elif max_area_ratio < threshold:
                        cropped_img.save(os.path.join(small_area_folder, cropped_img_filename))
                    else:
                        cropped_img.save(os.path.join(output_folder, cropped_img_filename))


#classify_and_crop_images()
#move_blurry_images("./experiment/output/yolo_detected", "./experiment/output/low_quality")
#back_ground()
process_folder(output_folder,json_output_folder)    
process_json_folder(json_output_folder,output_folder,missing_body_parts_folder)       
#openpose_process()  
#remove_low_quality_images("./experiment/output/back_ground", "./experiment/output/low_quality")
# 完了メッセージ
print("すべての画像処理と検出が完了しました。")
