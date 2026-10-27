# merge_registry.py – gộp registry.csv của các phiên Kaggle (thay cho merge.py)
#   python merge_registry.py results/runs/registry/registry.csv dl/s1/registry.csv ...
# Không xóa file nào; luôn đọc lại registry chung; run_id trùng phải giống hệt nhau.
import sys
from pathlib import Path
import pandas as pd
 
INT_COLS = ["fold", "seed", "n_epochs", "n_params"]
 
 
def merge(main, inputs):
    main = Path(main)
    old = pd.read_csv(main) if main.exists() else pd.DataFrame()
    allr = pd.concat([old] + [pd.read_csv(p) for p in inputs], ignore_index=True)
    for c in INT_COLS:
        if c in allr:
            allr[c] = pd.to_numeric(allr[c]).round().astype("Int64")
    dup = allr[allr.duplicated("run_id", keep=False)]
    bad = [k for k, g in dup.groupby("run_id") if len(g.drop_duplicates()) > 1]
    if bad:
        raise SystemExit(f"Dừng: run_id trùng nhưng nội dung khác: {bad}")
    out = allr.drop_duplicates().sort_values("run_id").reset_index(drop=True)
    tmp = main.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(main)                     # thay file một lần, không mất dữ liệu nếu lỗi
    print(f"{len(out)} dòng ({len(out) - len(old)} dòng mới)")
 
 
if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2:])
