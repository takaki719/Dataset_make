import os
import json
import shutil
#左右の肩と腰のx座標の差
#中心線の直線性
def is_facing_forward(keypoints):
    required_parts = {
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14
    }
    
    kp = keypoints
    parts = {}
    for part, idx in required_parts.items():
        if idx >= len(kp):
            print(f"キーポイントインデックス {idx} が範囲外です。")
            return False
        kp_idx = kp[idx]
        # キーポイントが (x, y, v) のリストであることを確認
        if isinstance(kp_idx, list) or isinstance(kp_idx, tuple):
            if len(kp_idx) == 3:
                x, y, v = kp_idx
            else:
                print(f"キーポイントの形式が不正です: {kp_idx}")
                return False
        else:
            print(f"キーポイントの形式が不正です: {kp_idx}")
            return False
        if v > 0:
            parts[part] = (x, y)
        else:
            print(f"{part} のキーポイントが見えていません。")
            return False  # 必要なキーポイントが見えていない場合はFalse
            
    # 左右のx座標の差を計算
    shoulder_distance = abs(parts['left_shoulder'][0] - parts['right_shoulder'][0])
    hip_distance = abs(parts['left_hip'][0] - parts['right_hip'][0])
    knee_distance = abs(parts['left_knee'][0] - parts['right_knee'][0])
    
    # 中心のx座標を計算
    shoulder_center = (parts['left_shoulder'][0] + parts['right_shoulder'][0]) / 2
    hip_center = (parts['left_hip'][0] + parts['right_hip'][0]) / 2
    knee_center = (parts['left_knee'][0] + parts['right_knee'][0]) / 2
    
    # 中心線のズレを計算
    center_diff1 = abs(shoulder_center - hip_center)
    center_diff2 = abs(hip_center - knee_center)
    
    threshold = 20  # 閾値は調整可能
    
    if center_diff1 < threshold and center_diff2 < threshold:
        return True
    else:
        return False

def main():
    # フォルダのパスを指定
    json_folder_path = './test/output/json'  # JSONファイルが保存されているフォルダ
    image_folder_path = './test/output/back_ground'  # 画像ファイルが保存されているフォルダ
    output_folder_forward = './test/output/forward'  # 正面を向いている画像を格納するフォルダ
    output_folder_not_forward = './test/output/not_forward'  # 正面を向いていない画像を格納するフォルダ

    # 出力フォルダを作成
    os.makedirs(output_folder_forward, exist_ok=True)
    os.makedirs(output_folder_not_forward, exist_ok=True)

    # JSONファイルのリストを取得
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_folder_path, json_file)
        print(f"処理中のJSONファイル: {json_file}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 対応する画像ファイル名を取得
        base_name = os.path.splitext(json_file)[0]  # '028254_keypoints'
        image_name = base_name.replace('_keypoints', '') + '.jpg'
        image_path = os.path.join(image_folder_path, image_name)

        if not os.path.exists(image_path):
            print(f"画像ファイルが見つかりません: {image_path}")
            continue

        # キーポイントを取得
        if 'people' in data and len(data['people']) > 0:
            # OpenPoseの出力形式の場合
            keypoints = data['people'][0]['pose_keypoints_2d']
        elif 'keypoints' in data:
            # 単純な形式の場合
            keypoints = data['keypoints']
        else:
            print(f"キーポイントが見つかりません: {json_file}")
            continue

        # キーポイントを (x, y, v) のリストに変換
        if isinstance(keypoints[0], (int, float)):
            keypoints = [keypoints[i:i+3] for i in range(0, len(keypoints), 3)]
        elif isinstance(keypoints[0], list) or isinstance(keypoints[0], tuple):
            pass  # すでに (x, y, v) のリスト
        else:
            print(f"キーポイントの形式が不正です: {keypoints[0]}")
            continue

        is_forward = is_facing_forward(keypoints)
        print(f"画像 {image_name} の正面判定結果: {'正面' if is_forward else '非正面'}")

        # 画像を対応するフォルダにコピー
        if is_forward:
            dest_path = os.path.join(output_folder_forward, image_name)
            print(f"画像 {image_name} を正面フォルダにコピーします。")
        else:
            dest_path = os.path.join(output_folder_not_forward, image_name)
            print(f"画像 {image_name} を非正面フォルダにコピーします。")
        shutil.copy2(image_path, dest_path)

    print("処理が完了しました。")

if __name__ == '__main__':
    main()
