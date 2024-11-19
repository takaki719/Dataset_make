import os
from pathlib import Path

def compare_and_delete(folder1_path, folder2_path):
    # フォルダーのパスを取得
    folder1 = Path(folder1_path)
    folder2 = Path(folder2_path)

    # フォルダー内のファイル名を取得（拡張子を除いた名前を比較用キーとして抽出）
    folder1_files = {f.stem.split('_')[0] for f in folder1.iterdir() if f.is_file()}
    folder2_files = {f.stem.split('_')[0] for f in folder2.iterdir() if f.is_file()}

    # 両方のフォルダーに存在しないファイルを見つける
    only_in_folder1 = folder1_files - folder2_files

    # フォルダー1から削除
    for file_stem in only_in_folder1:
        for file_path in folder1.iterdir():
            if file_path.stem.split('_')[0] == file_stem:
                file_path.unlink()
                print(f"Deleted from folder1: {file_path.name}")



def rename_images_in_folder(folder_path):
    # フォルダーのパスを取得
    folder = Path(folder_path)

    # フォルダー内の画像ファイルを取得（jpg, png, jpegなど）
    image_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and f.is_file()
    ]

    # ファイルを名前順で並べ替え
    image_files.sort()

    # ファイルのリネーム
    for i, image_file in enumerate(image_files, start=1903):
        new_name = f"{i:06d}_1.jpg"  # 新しいファイル名
        new_path = folder / new_name
        
        # ファイルをリネーム
        image_file.rename(new_path)
        print(f"Renamed: {image_file.name} -> {new_name}")

def rename_images_in_folder_0(folder_path):
    # フォルダーのパスを取得
    folder = Path(folder_path)

    # フォルダー内の画像ファイルを取得（jpg, png, jpegなど）
    image_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and f.is_file()
    ]

    # ファイルを名前順で並べ替え
    image_files.sort()

    # ファイルのリネーム
    for i, image_file in enumerate(image_files, start=1903):
        new_name = f"{i:06d}_0.jpg"  # 新しいファイル名
        new_path = folder / new_name
        
        # ファイルをリネーム
        image_file.rename(new_path)
        print(f"Renamed: {image_file.name} -> {new_name}")

def rename_images_in_folder_json(folder_path):
    # フォルダーのパスを取得
    folder = Path(folder_path)

    # フォルダー内の画像ファイルを取得（jpg, png, jpegなど）
    image_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png",".json"] and f.is_file()
    ]

    # ファイルを名前順で並べ替え
    image_files.sort()

    # ファイルのリネーム
    for i, image_file in enumerate(image_files, start=1903):
        new_name = f"{i:06d}_0_keypoints.json"  # 新しいファイル名
        new_path = folder / new_name
        
        # ファイルをリネーム
        image_file.rename(new_path)
        print(f"Renamed: {image_file.name} -> {new_name}")

# フォルダーのパスを配列で指定
#folders = [
#    "/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/img",
#    "/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/mask",
#    "/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/color",
#    "/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/label",
#    "/media/il/local2/Virtual_try_on/Preprocessing/test/output/json",
#]
folders = [
    "/media/il/local2/Virtual_try_on/Preprocessing/test/prepro_tmp/img",
    "/media/il/local2/Virtual_try_on/Preprocessing/test/prepro_tmp/mask",
    "/media/il/local2/Virtual_try_on/Preprocessing/test/prepro_tmp/color",
    "/media/il/local2/Virtual_try_on/Preprocessing/test/prepro_tmp/label",
    "/media/il/local2/Virtual_try_on/Preprocessing/test/prepro_tmp/json",
]
# 配列内のフォルダーを順に比較して処理
#compare_and_delete(folders[0], folders[1])
#compare_and_delete(folders[1], folders[2])
#compare_and_delete(folders[2], folders[3])
#compare_and_delete(folders[3], folders[4])
#compare_and_delete(folders[4], folders[0])

# 最初のフォルダーでリネーム処理
#rename_images_in_folder_0(folders[0])
#rename_images_in_folder(folders[1])
#rename_images_in_folder(folders[2])
#rename_images_in_folder_0(folders[3])
#rename_images_in_folder_json(folders[4])
