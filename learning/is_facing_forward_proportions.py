#スケルトン全体の形状に基づく判定
import os
import json
import shutil
def is_facing_forward_proportions(keypoints):
    required_parts = {
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_hip': 11,
        'right_hip': 12,
        'left_ankle': 15,
        'right_ankle': 16
    }

    kp = keypoints
    parts = {}
    for part, idx in required_parts.items():
        if idx >= len(kp):
            return False
        x, y, v = kp[idx]
        if v > 0:
            parts[part] = (x, y)
        else:
            return False

    # 身体の高さを計算（肩から足首まで）
    top_y = (parts['left_shoulder'][1] + parts['right_shoulder'][1]) / 2
    bottom_y = (parts['left_ankle'][1] + parts['right_ankle'][1]) / 2
    height = bottom_y - top_y

    # 肩幅と腰幅を計算
    shoulder_width = abs(parts['left_shoulder'][0] - parts['right_shoulder'][0])
    hip_width = abs(parts['left_hip'][0] - parts['right_hip'][0])

    # 比率を評価
    shoulder_ratio = shoulder_width / height
    hip_ratio = hip_width / height

    # 比率の閾値を設定
    min_ratio = 0.15
    max_ratio = 0.4

    if min_ratio < shoulder_ratio < max_ratio and min_ratio < hip_ratio < max_ratio:
        return True
    else:
        return False
def main():
    # フォルダのパスを指定
    json_folder_path = './test/output/json'  # JSONファイルが保存されているフォルダ
    image_folder_path = './test/output/back_ground'  # 画像ファイルが保存されているフォルダ
    output_folder_forward = './test/output/forward'  # 正面を向いている画像を格納するフォルダ
    output_folder_not_forward = './test/output/forward_not'  # 正面を向いていない画像を格納するフォルダ
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
        # 例: '028254_keypoints.json' -> '028254.jpg'
        base_name = os.path.splitext(json_file)[0]  # '028254_keypoints'
        image_name = base_name.replace('_keypoints', '') + '.jpg'
        image_path = os.path.join(image_folder_path, image_name)

        if not os.path.exists(image_path):
            print(f"画像ファイルが見つかりません: {image_path}")
            continue

        # キーポイントを取得
        # JSONファイルの構造に応じて変更が必要
        if 'people' in data and len(data['people']) > 0:
            # OpenPoseの出力形式の場合
            keypoints = data['people'][0]['pose_keypoints_2d']
        elif 'keypoints' in data:
            # 単純な形式の場合
            keypoints = data['keypoints']
        else:
            print(f"キーポイントが見つかりません: {json_file}")
            continue

        # キーポイントの内容を表示（デバッグ用）
        print(f"キーポイントの内容: {keypoints}")

        # キーポイントを (x, y, v) のリストに変換
        # もし keypoints がすでに (x, y, v) のリストであれば、再処理は不要
        if isinstance(keypoints[0], (int, float)):
            keypoints = [keypoints[i:i+3] for i in range(0, len(keypoints), 3)]

        is_forward = is_facing_forward_proportions(keypoints)
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
