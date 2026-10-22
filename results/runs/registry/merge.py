from pathlib import Path
import pandas as pd

# Thư mục hiện tại
ROOT = Path.cwd()

# File tổng hợp đầu ra
OUTPUT = ROOT / "registry.csv"

# Tìm tất cả file CSV trong các thư mục con
csv_files = [
    f for f in ROOT.rglob("*.csv")
    if f != OUTPUT
]

if not csv_files:
    print("Không tìm thấy file CSV nào.")
    raise SystemExit

print(f"Tìm thấy {len(csv_files)} file CSV.")

# Gộp tất cả file
dfs = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"Đã đọc: {file}")
    except Exception as e:
        print(f"Lỗi khi đọc {file}: {e}")

if not dfs:
    print("Không có dữ liệu để tổng hợp. Không xóa file nào.")
    raise SystemExit

# Gộp dữ liệu, giữ nguyên các cột
result = pd.concat(dfs, ignore_index=True, sort=False)

# Lưu file tổng hợp
result.to_csv(OUTPUT, index=False)

print(f"\nĐã tổng hợp {len(result)} dòng vào: {OUTPUT}")

# Chỉ xóa file nguồn sau khi lưu file tổng hợp thành công
for file in csv_files:
    file.unlink()

print(f"Đã xóa {len(csv_files)} file CSV nguồn.")
print(f"File kết quả: {OUTPUT}")