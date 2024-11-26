import os
import subprocess
from subprocess import PIPE
import cv2
import numpy as np
import glob
from PIL import Image
from scipy.ndimage import label

def resize_and_copy_images(folder_path, output_folder_path, width=192, height=256):
    # 出力フォルダーが存在しない場合に作成
    os.makedirs(output_folder_path, exist_ok=True)
    
    # フォルダー内のファイルを取得
    for filename in os.listdir(folder_path):
        # 入力ファイルパスの作成
        input_file_path = os.path.join(folder_path, filename)
        
        # 画像ファイルのみ処理
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(input_file_path) as img:
                    # 画像を指定サイズにリサイズ
                    resized_img = img.resize((width, height))
                    
                    # 出力ファイルパスを作成
                    output_file_path = os.path.join(output_folder_path, filename)
                    
                    # 新しいフォルダーに保存
                    resized_img.save(output_file_path)
            except Exception as e:
                print(f"{filename} の処理中にエラーが発生しました: {e}")

def resize_and_overwrite_images(folder_path, width=192, height=256):
    # フォルダー内のファイルを取得
    for filename in os.listdir(folder_path):
        # 入力ファイルパスの作成
        input_file_path = os.path.join(folder_path, filename)
        
        # 画像ファイルのみ処理
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(input_file_path) as img:
                    # 画像を指定サイズにリサイズ
                    resized_img = img.resize((width, height))
                    
                    # 同じファイルパスに上書き保存
                    resized_img.save(input_file_path)
            except Exception as e:
                print(f"{filename} の処理中にエラーが発生しました: {e}")

def extract_largest_connected_component(cleaned_image_array):
    """
    Extract the largest connected component from a binary image array.

    Parameters:
        cleaned_image_array (numpy.ndarray): Input binary image array.

    Returns:
        numpy.ndarray or None: Mask of the largest connected component, or None if no components exist.
    """
    # Label connected components in the image
    labeled_array, num_features = label(cleaned_image_array)
    
    if num_features == 0:  # No connected components
        print("No connected components found.")
        return None

    # Calculate the size of each component
    component_sizes = np.bincount(labeled_array.ravel())
    
    # Ensure there are components other than background
    if len(component_sizes) <= 1:  # Only background
        print("No non-background components found.")
        return None
    
    # Find the label of the largest connected component (ignoring background label 0)
    largest_component_label = component_sizes[1:].argmax() + 1

    # Create a mask to preserve only the largest connected component
    largest_component_mask = (labeled_array == largest_component_label)

    return (largest_component_mask * 255).astype(np.uint8)

def bounding_rect_img(image):
    """
    Crop the image to the bounding rectangle of non-white pixels.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Cropped image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(255 - gray)  # Find non-white areas
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def process_images(input_folder, output_folder, seg_folder, mask_folder):
    filepath_list = glob.glob(os.path.join(input_folder, '*.jpg'))
    
    for name in filepath_list:
        filename = os.path.basename(name)
        rename = filename.replace(".jpg", ".png")
        filename_seg = rename
        
        # 入力画像の読み込み
        image = cv2.imread(os.path.join(input_folder, filename))
        if image is None:
            print(f"Failed to load input image: {filename}, skipping.")
            continue
        
        # セグメンテーションマスクの読み込み
        seg_path = os.path.join(seg_folder, filename_seg)
        if not os.path.exists(seg_path):
            print(f"Segmentation file not found: {seg_path}, skipping.")
            continue
        
        image_seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if image_seg is None:
            print(f"Failed to load segmentation file: {seg_path}, skipping.")
            continue
        
        # セグメンテーション領域抽出
        img_mask = cv2.inRange(image_seg, 2, 2)
        result = cv2.bitwise_and(image, image, mask=img_mask)
        result[img_mask == 0] = [255, 255, 255]
        
        # クロップとリサイズ
        crop = bounding_rect_img(result)
        re_crop = cv2.resize(crop, (192, 256))
        
        # GrabCutによるマスク生成
        mask = np.zeros(re_crop.shape[:2], dtype="uint8")
        rect = (10, 10, re_crop.shape[1] - 20, re_crop.shape[0] - 20)
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        
        try:
            (mask, bgModel, fgModel) = cv2.grabCut(re_crop, mask, rect, bgModel, fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            print(f"GrabCut failed for {filename} with error: {e}, skipping.")
            continue
        
        outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype("uint8")
        outputMask = extract_largest_connected_component(outputMask)
        
        if outputMask is not None:
            # マスク適用と保存
            re_crop[outputMask == 0] = [255, 255, 255]
            cv2.imwrite(os.path.join(output_folder, filename), re_crop)
            cv2.imwrite(os.path.join(mask_folder, filename), outputMask)
        else:
            print(f"No valid mask generated for {filename}, skipping save.")


def keep_largest_non_white_contour_and_invert(image_path, output_path):
    # Load the image in color
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale and apply binary thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image to make white background black
    inverted = cv2.bitwise_not(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    
    # Apply binary thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found, exit
    if not contours:
        print("No contours found in the image.")
        return
    
    # Find the largest non-white contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask with only the largest non-white contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Invert the result to make the main object black and the background white
    inverted_result = cv2.bitwise_not(result_image)
    
    # Save and display the final result
    cv2.imwrite(output_path, inverted_result)

def process_folder_for_largest_contour(input_folder, output_folder):

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    # Process each image in the folder

    for image_path in glob.glob(os.path.join(input_folder, '*.jpg')):
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)       
        # Process each image to keep only the largest non-white contour/ and invert it
        keep_largest_non_white_contour_and_invert(image_path, output_path)

def main(input_paths, color_paths, mask_paths,label_paths,img_paths):
    
    num_steps = len(input_paths)
    for i in range(num_steps):
        input_folder = input_paths[i]
        color_folder = color_paths[i]
        mask_folder = mask_paths[i]
        label_folder = label_paths[i]
        img_folder = img_paths[i]
        # Ensure directories exist
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(color_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(label_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        # Run external segmentation scriptフォルダーから画像ファイルを読み込んで、最初の画像から000001_1.jpgとして、次は000002_1.jpgとして、ならべていきたい。
        #resize_and_copy_images(input_folder,img_folder) #img画像のリサイズ
        #subprocess.run([
        #    "/media/il/local2/Virtual_try_on/Preprocessing/pre_venv/bin/python", "/media/il/local2/Virtual_try_on/Preprocessing/modules/UniHCP-inference/inference_for_dataset.py",
        #    '--input_dir', input_folder,
        #    '--label_dir', label_folder,
        #], stdout=PIPE, stderr=PIPE)
        #subprocess.run([
        #    "/media/il/local2/Virtual_try_on/Preprocessing/pre_venv/bin/python", "/media/il/local2/Virtual_try_on/Preprocessing/name_.py",
        #], stdout=PIPE, stderr=PIPE)
        resize_and_overwrite_images(label_folder) #label画像のリサイズ
        process_images(img_folder, color_folder, label_folder, mask_folder)#画像の前処理
        
        #process_folder_for_largest_contour(output_folder,output_folder) #最大の輪郭を抽出
# Example usage with dynamic paths
input_paths = [
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/other/back_ground',
    #'/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/スケルトン全体の形状に基づく判定/forward',
    #'/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/四肢の対称性に基づく判定/forward',
    #'/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/左右の肩と腰のx座標の差,中心線の直線性/forward',
    #'/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/肩と腰の角度に基づく判定/forward',
    #'/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/身体の対称性/forward'
]
color_paths = [
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/color',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/スケルトン全体の形状に基づく判定/prepro/color',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/四肢の対称性に基づく判定/prepro/color',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/左右の肩と腰のx座標の差,中心線の直線性/prepro/color',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/肩と腰の角度に基づく判定/prepro/color',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/身体の対称性/prepro/color'
]
mask_paths = [
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/mask',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/スケルトン全体の形状に基づく判定/prepro/mask',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/四肢の対称性に基づく判定/prepro/mask',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/左右の肩と腰のx座標の差,中心線の直線性/prepro/mask',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/肩と腰の角度に基づく判定/prepro/mask',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/身体の対称性/prepro/mask'
]
label_paths = [
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/label',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro_tmp/label',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/スケルトン全体の形状に基づく判定/prepro/label',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/四肢の対称性に基づく判定/prepro/label',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/左右の肩と腰のx座標の差,中心線の直線性/prepro/label',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/肩と腰の角度に基づく判定/prepro/label',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/身体の対称性/prepro/label'
]
img_paths = [
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/img',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/スケルトン全体の形状に基づく判定/prepro/img',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/四肢の対称性に基づく判定/prepro/img',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/左右の肩と腰のx座標の差,中心線の直線性/prepro/img',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/肩と腰の角度に基づく判定/prepro/img',
    '/media/il/local2/Virtual_try_on/Preprocessing/test/output/判別結果/身体の対称性/prepro/img'
]



main(input_paths, color_paths,mask_paths,label_paths,img_paths)
