import os
import shutil
import argparse
import glob

def copy_files(source_dir, target_dir, patterns):
    """
    指定したパターンにマッチするファイルをソースディレクトリからターゲットディレクトリにコピーします。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"ターゲットディレクトリを作成しました: {target_dir}")

    for pattern in patterns:
        # 絶対パスを取得
        full_pattern = os.path.join(source_dir, pattern)
        matched_files = glob.glob(full_pattern)
        if not matched_files:
            print(f"パターンにマッチするファイルがありません: {full_pattern}")
            continue
        for src_path in matched_files:
            file_name = os.path.basename(src_path)
            dest_path = os.path.join(target_dir, file_name)
            if os.path.isfile(src_path):
                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"コピー成功: {src_path} -> {dest_path}")
                except Exception as e:
                    print(f"コピー失敗: {src_path} -> {dest_path}\nエラー: {e}")
            else:
                print(f"ファイルが存在しません: {src_path}")

def move_files(source_dir, target_dir, patterns):
    """
    指定したパターンにマッチするファイルをソースディレクトリからターゲットディレクトリに移動します。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"ターゲットディレクトリを作成しました: {target_dir}")

    for pattern in patterns:
        # 絶対パスを取得
        full_pattern = os.path.join(source_dir, pattern)
        matched_files = glob.glob(full_pattern)
        if not matched_files:
            print(f"パターンにマッチするファイルがありません: {full_pattern}")
            continue
        for src_path in matched_files:
            file_name = os.path.basename(src_path)
            dest_path = os.path.join(target_dir, file_name)
            if os.path.isfile(src_path):
                try:
                    shutil.move(src_path, dest_path)
                    print(f"移動成功: {src_path} -> {dest_path}")
                except Exception as e:
                    print(f"移動失敗: {src_path} -> {dest_path}\nエラー: {e}")
            else:
                print(f"ファイルが存在しません: {src_path}")

def main():
    # ここに入力パス、出力パス、ファイルパターンを定義
    tasks = [
        {
            'action': 'copy',
            'source_dir': '/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/color',
            'target_dir': '/media/il/local2/Virtual_try_on/Preprocessing/kosei_takaki/DeepFashion_Try_On/Data_preprocessing/train_color',
            'patterns': ['*.jpg']
        },
        {
            'action': 'copy',
            'source_dir': '/media/il/local2/Virtual_try_on/Preprocessing/test/output/prepro/mask',
            'target_dir': '/media/il/local2/Virtual_try_on/Preprocessing/kosei_takaki/DeepFashion_Try_On/Data_preprocessing/train_edge',
            'patterns': ['*.jpg']
        },
        {
            'action': 'copy',
            'source_dir': '/media/il/local2/Virtual_try_on/Preprocessing//test/output/prepro/img',
            'target_dir': '/media/il/local2/Virtual_try_on/Preprocessing/kosei_takaki/DeepFashion_Try_On/Data_preprocessing/train_img',
            'patterns': ['*.jpg']
        },
        {
            'action': 'copy',
            'source_dir': '/media/il/local2/Virtual_try_on/Preprocessing//test/output/prepro/json',
            'target_dir': '/media/il/local2/Virtual_try_on/Preprocessing/kosei_takaki/DeepFashion_Try_On/Data_preprocessing/train_pose',
            'patterns': ['*.json']
        },
        {
            'action': 'copy',
            'source_dir': '/media/il/local2/Virtual_try_on/Preprocessing//test/output/prepro/label',
            'target_dir': '/media/il/local2/Virtual_try_on/Preprocessing/kosei_takaki/DeepFashion_Try_On/Data_preprocessing/train_label',
            'patterns': ['*.png']
        },
    ]

    for task in tasks:
        action = task['action']
        source_dir = task['source_dir']
        target_dir = task['target_dir']
        patterns = task['patterns']

        if action == 'copy':
            copy_files(source_dir, target_dir, patterns)
        elif action == 'move':
            move_files(source_dir, target_dir, patterns)
        else:
            print(f"無効なアクションが指定されました: {action}")

if __name__ == "__main__":
    main()
