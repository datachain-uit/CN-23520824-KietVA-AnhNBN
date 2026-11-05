# approve_runs.py – ghi các lần chạy đã được review vào approved.csv (không ghi tay).
# Đặt cạnh registry.csv (results/runs/registry/). Chạy ở gốc repo, NGƯỜI REVIEW chạy,
# sau khi pull request chứa các lần chạy đó đã được gộp:
#   python results/runs/registry/approve_runs.py --reviewer Kiet --commit <hash> \
#       --checks "Bảng 6.7; summarize_boot khớp" --model DKT --variant A
#   python results/runs/registry/approve_runs.py --reviewer NhatAnh --commit <hash> \
#       --checks "PR #6; test_simplekt_qmask 5/5" --run_id SIMPLEKT_C_f0_s42_...
# Chọn theo --run_id, hoặc lọc theo --model/--variant/--fold/--seed. Không ghi trùng.
import argparse, datetime, os
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
KEY = ["run_id", "model", "variant", "fold", "seed"]
COLS = KEY + ["reviewer", "ngay", "commit", "kiem_tra", "ghi_chu"]


def main(a):
    reg = pd.read_csv(os.path.join(D, "registry.csv"))
    path = os.path.join(D, "approved.csv")
    ok = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=COLS)
    if a.run_id:
        miss = sorted(set(a.run_id) - set(reg["run_id"]))
        if miss:
            raise SystemExit(f"không có trong registry.csv: {miss}")
        sel = reg[reg["run_id"].isin(a.run_id)]
    else:
        sel = reg
        for col in ("model", "variant", "fold", "seed"):
            v = getattr(a, col)
            if v is not None:
                sel = sel[sel[col].astype(str) == str(v)]
    sel = sel[~sel["run_id"].isin(ok["run_id"])]
    if sel.empty:
        print("không có lần chạy mới cần ghi")
        return
    new = sel[KEY].assign(reviewer=a.reviewer, ngay=datetime.date.today().isoformat(),
                          commit=a.commit, kiem_tra=a.checks, ghi_chu=a.note)
    out = pd.concat([ok, new], ignore_index=True)[COLS]
    out.to_csv(path, index=False)
    print(f"thêm {len(new)} dòng; approved.csv có {len(out)} dòng")
    print(new.groupby(["model", "variant"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--commit", required=True)        # commit gộp PR trên main
    ap.add_argument("--checks", required=True)        # đã kiểm gì (Bảng 6.7 ngày 31/10)
    ap.add_argument("--note", default="")
    ap.add_argument("--run_id", nargs="*")
    ap.add_argument("--model")
    ap.add_argument("--variant")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    main(ap.parse_args())
