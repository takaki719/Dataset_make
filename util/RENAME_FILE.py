# ファイル名の範囲
start_number = 1
end_number = 25857

# 出力先のファイル名
output_file = "file_list.txt"

# ファイル名を生成し、txtファイルに書き込む
with open(output_file, "w") as file:
    for i in range(start_number, end_number + 1):
        file.write(f"{i:06d}_0.jpg {i:06d}_1.jpg\n")

print(f"ファイルリストが {output_file} に保存されました。")
