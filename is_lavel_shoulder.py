import os
import json
import math
import shutil

def calculate_angle(a, b, c):
    """3点a, b, cのうち、点bにおける角度を計算"""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot_product = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)
    if mag_ab * mag_cb == 0:
        return 0
    angle = math.acos(dot_product / (mag_ab * mag_cb))
    return math.degrees(angle)

def calculate_centroid(keypoints):
    """キーポイントの重心を計算"""
    x_coords = [kp[0] for kp in keypoints if kp[2] > 0]
    y_coords = [kp[1] for kp in keypoints if kp[2] > 0]
    if not x_coords or not y_coords:
        return (0, 0)
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return (centroid_x, centroid_y)

def is_front_facing(keypoints, image_width, image_height, 
                   symmetry_threshold=0.2, angle_threshold=160,
                   centroid_threshold=0.3, tilt_threshold=30):
    """
    人物が正面を向いており、直立していて、中央に配置されていて、傾いていないかを判定する。
    
    Parameters:
    - keypoints: [(x, y, v), ...] COCO形式のキーポイントリスト
    - image_width: 画像の幅
    - image_height: 画像の高さ
    - symmetry_threshold: 対称性の閾値
    - angle_threshold: 腕の伸び具合の閾値
    - centroid_threshold: 中央配置の閾値（画像の最大距離に対する割合）
    - tilt_threshold: 傾きの閾値（ピクセル単位）
    
    Returns:
    - True if正面向きで条件を満たしている、False otherwise
    """
    # 必要なキーポイントを辞書に格納
    key_dict = {}
    for idx, (x, y, v) in enumerate(keypoints):
        key_dict[idx+1] = (x, y, v)  # COCOキーポイントは1から始まる

    # 可視性の確認
    required_keys = [1,2,3,6,7,8,9,10,11,12,13]  # 鼻、目、肩、肘、手首、股関節
    for key in required_keys:
        if key_dict.get(key, (0,0,0))[2] < 0.5:
            return False  # 必要なキーポイントが見えていない

    # 対称性の確認（肩の例）
    left_shoulder = key_dict[6]
    right_shoulder = key_dict[7]
    left_hip = key_dict[12]
    right_hip = key_dict[13]

    shoulder_distance = math.hypot(left_shoulder[0] - right_shoulder[0],
                                   left_shoulder[1] - right_shoulder[1])
    hip_distance = math.hypot(left_hip[0] - right_hip[0],
                              left_hip[1] - right_hip[1])
    if shoulder_distance < symmetry_threshold * hip_distance:
        return False  # 肩が十分に広がっていない

    # 腕の位置の確認
    left_elbow = key_dict[8]
    left_wrist = key_dict[10]
    right_elbow = key_dict[9]
    right_wrist = key_dict[11]

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    if left_arm_angle < angle_threshold or right_arm_angle < angle_threshold:
        return False  # 腕が十分に伸びていない

    # 直立性の確認
    # 肩と腰のY座標の差を確認（水平性）
    shoulder_y_diff = abs(left_shoulder[1] - right_shoulder[1])
    hip_y_diff = abs(left_hip[1] - right_hip[1])
    if shoulder_y_diff > tilt_threshold or hip_y_diff > tilt_threshold:
        return False  # 肩または腰が水平でない

    # 背骨の直線性を確認
    # 背骨の角度を計算（垂直線との角度）
    spine_mid_x = (left_shoulder[0] + right_shoulder[0])/2
    spine_mid_y = (left_shoulder[1] + right_shoulder[1])/2
    hip_mid_x = (left_hip[0] + right_hip[0])/2
    hip_mid_y = (left_hip[1] + right_hip[1])/2

    delta_x = hip_mid_x - spine_mid_x
    delta_y = hip_mid_y - spine_mid_y
    spine_angle = math.degrees(math.atan2(delta_y, delta_x))
    spine_angle = abs(spine_angle - 90)  # 90度が垂直
    if spine_angle > tilt_threshold:
        return False  # 背骨が垂直でない

    # 中央配置の確認
    centroid = calculate_centroid(keypoints)
    image_center = (image_width / 2, image_height / 2)
    centroid_distance = math.hypot(centroid[0] - image_center[0], centroid[1] - image_center[1])
    max_distance = math.sqrt((image_width/2)**2 + (image_height/2)**2)
    if centroid_distance > centroid_threshold * max_distance:
        return False  # 人物の重心が画像の中心から遠い

    # 肩の水平性を確認
    if shoulder_y_diff > tilt_threshold:
        return False  # 肩の高さに差がある

    return True

def main():
    # フォルダのパスを指定
    json_folder_path = './test/output/json'  # JSONファイルが保存されているフォルダ
    image_folder_path = './test/output/back_ground'  # 画像ファイルが保存されているフォルダ
    output_folder_forward = './test/output/判別結果/肩/forward'  # 正面を向いている画像を格納するフォルダ
    output_folder_not_forward = './test/output/判別結果/肩/forward_not'  # 正面を向いていない画像を格納するフォルダ

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
            # COCO形式の場合
            keypoints = data['people'][0]['keypoints']
        elif 'keypoints' in data:
            # 他の形式の場合
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

        # 画像のサイズを取得
        # PILを使用して画像のサイズを取得
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except ImportError:
            print("PILライブラリがインストールされていません。画像サイズの取得に失敗しました。")
            image_width, image_height = 640, 480  # デフォルト値を設定
        except Exception as e:
            print(f"画像サイズの取得中にエラーが発生しました: {e}")
            image_width, image_height = 640, 480  # デフォルト値を設定

        # 正面判定
        is_forward = is_front_facing(keypoints, image_width, image_height)
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
