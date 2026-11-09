# repeat_diag.py – chẩn đoán cho sai lệch G1 của simpleKT-PQC (quyết định 7.1, 07/11).
# Quy ước "một câu = đoạn liên tiếp cùng ID câu" gộp hai lần trả lời liền nhau của cùng
# một câu, nên khi dự đoán lần lặp, simpleKT-PQC không nhìn được lần trả lời ngay trước.
# Script so b − a (mặc định C − pyKT của SIMPLEKT) trên nhóm câu "lặp liền kề" và phần
# còn lại, từng fold, bằng paired bootstrap theo học sinh. CPU, vài phút mỗi fold.
# Đặt trong tests/bootstrap/, cạnh boot_metrics.py. Chạy ở gốc repo:
#   python tests/bootstrap/repeat_diag.py --runs_root tests/bootstrap/runs \
#       --approved results/runs/registry/approved.csv --out tests/bootstrap/out_skt/repeat
import argparse, os
import numpy as np
import pandas as pd
from boot_metrics import paired_bootstrap


def repeat_groups(q):
    """lặp: câu được chấm có cùng ID câu với câu được chấm ngay trước của học sinh đó."""
    q = q.sort_values("qidx")
    prev = q.groupby("uid")["question"].shift(1)
    return pd.DataFrame({"qidx": q["qidx"].to_numpy(),
                         "nhom": np.where(q["question"].to_numpy() == prev.to_numpy(),
                                          "lặp liền kề", "không lặp")})


def main(a):
    reg = pd.read_csv(os.path.join(a.runs_root, "registry", "registry.csv"))
    if a.approved:
        reg = reg[reg["run_id"].isin(pd.read_csv(a.approved)["run_id"])]
    reg = reg[(reg["model"] == a.model) & (reg["seed"] == a.seed)
              & reg["variant"].isin([a.a, a.b])]
    rows = []
    for fold, r in reg.groupby("fold"):
        r = r.set_index("variant")
        if a.a not in r.index or a.b not in r.index:
            print(f"Bỏ qua fold {fold}: thiếu {a.a} hoặc {a.b}")
            continue
        qa = pd.read_parquet(os.path.join(a.runs_root, r.loc[a.a, "preds_path"]))
        qb = pd.read_parquet(os.path.join(a.runs_root, r.loc[a.b, "preds_path"]))
        g = repeat_groups(qa)
        for nhom, part in [("tất cả", g)] + list(g.groupby("nhom")):
            res = paired_bootstrap(qa, qb, a.n_boot, seed=int(fold), subset=part["qidx"])
            res.insert(0, "nhom", nhom)
            res.insert(0, "fold", fold)
            res["ti_le"] = len(part) / len(g)
            rows.append(res)
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(f"{a.out}_folds.csv", index=False)
    out["ci_tot"] = np.where(out["metric"] == "ΔAUC", out["lo"] > 0, out["hi"] < 0)
    out["ci_kem"] = np.where(out["metric"] == "ΔAUC", out["hi"] < 0, out["lo"] > 0)
    summ = out.groupby(["nhom", "metric"]).agg(
        hieu_tb=("delta", "mean"), nho_nhat=("delta", "min"), lon_nhat=("delta", "max"),
        ci_tot=("ci_tot", "sum"), ci_kem=("ci_kem", "sum"), so_fold=("delta", "size"),
        so_cau_tb=("n_cau", "mean"), ti_le=("ti_le", "mean"))
    summ.round(5).to_csv(f"{a.out}.csv")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--model", default="SIMPLEKT")
    ap.add_argument("--a", default="pyKT")              # mốc
    ap.add_argument("--b", default="C")                 # simpleKT-PQC
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--approved", default="")
    ap.add_argument("--out", default="tests/bootstrap/out_skt/repeat")
    main(ap.parse_args())
