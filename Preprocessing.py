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
from rembg import remove, new_session
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

# フォルダパスの設定
#input_folder = "./input/test"
input_folder = "./experiment/input/fafafa"
output_folder = "./experiment/output/back_ground"
small_area_folder = "./experiment/output/miss_file/small_area_images"
missing_body_parts_folder = "./experiment/output/miss_file/missing_body_parts_images"
json_output_folder = "./experiment/output/json"
image_output_folder ="./experiment/output/image"
alpha_folder = "./experiment/output/miss_file/alpha_images"
openpose_models_folder = "./modules/openpose/models"
yolo_output_folder = "./experiment/output/yolo_detected"
multi_person_folder = "./experiment/output/miss_file/multi_person_images"

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)
os.makedirs(small_area_folder, exist_ok=True)
os.makedirs(missing_body_parts_folder, exist_ok=True)
os.makedirs(json_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(alpha_folder, exist_ok=True)
os.makedirs(yolo_output_folder, exist_ok=True)
os.makedirs(multi_person_folder, exist_ok=True)



def back_ground():
    remove_low_quality_images("./experiment/output/yolo_detected", "./experiment/output/low_quality")
    session = new_session()
    for file in Path("./experiment/output/yolo_detected").glob('*.jpg'):
        input_path = str(file)
        output_path = "./experiment/output/back_ground" + "/" + (file.stem + ".jpg")

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input, session=session)
                o.write(output)

        # 背景削除後の画像が正しく保存されているか確認
        if not any(fname.endswith('.jpg') for fname in os.listdir(yolo_output_folder)):
            print(f"No images found in {yolo_output_folder}. Please check the background removal step.")
            exit(1)
    area_process()
    remove_low_quality_images("./experiment/output/yolo_detected", "./experiment/output/low_quality")
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

def remove_low_quality_images(source_folder, low_quality_folder, threshold=10):
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

def copy_with_retries(src, dst, retries=3):
    for attempt in range(retries):
        try:
            shutil.copy(src, dst)
            return True
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
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

def detect_person_yolo(input_folder = input_folder,yolo_output_folder= yolo_output_folder):
    model = YOLO('yolov10x.pt')

# 入力ディレクトリ内のすべての画像ファイルに対して処理を行う
    for file_name in os.listdir(input_folder):
        # 画像ファイルのパス
        image_path = os.path.join(input_folder, file_name)
    
        # ファイルが画像かどうか確認
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # 画像を読み込む
        img = cv2.imread(image_path)
    
        if img is None:
            print(f"Error: Unable to read image {image_path}.")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # 推論を実行
        results = model(img_rgb)
    
        # 「PERSON」ラベルの数をカウント
        person_count = 0
        for result in results:
            for detection in result.boxes:
                label = result.names[int(detection.cls)]
                if label.lower() == 'person':
                    person_count += 1
    
        # 「PERSON」ラベルが1つだけ検出された場合、画像を出力ディレクトリに移動
        if person_count == 1:
            shutil.move(image_path, os.path.join(yolo_output_folder, file_name))
            print(f"Moved {image_path} to {yolo_output_folder}")
        else:
            # 「PERSON」ラベルが1つでない場合、画像を別のディレクトリに移動
            shutil.move(image_path, os.path.join(multi_person_folder, file_name))
    
    
def openpose_process():
    #openpose_done(output_folder)#output_folder -> yolo

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

move_blurry_images("./experiment/output/back_ground", "./experiment/output/low_quality")


#detect_person_yolo(input_folder="/media/il/local2/Virtual_try_on/DeepFashion/train/image")
#back_ground()
#area_process()                
#openpose_process()
#detect_person_yolo(input_folder="./experiment/output/back_ground",yolo_output_folder="./experiment/output/back_ground")
#classify_images_in_folder()
#remove_low_quality_images("./experiment/output/back_ground", "./experiment/output/low_quality")
# 完了メッセージ
print("すべての画像処理と検出が完了しました。")
