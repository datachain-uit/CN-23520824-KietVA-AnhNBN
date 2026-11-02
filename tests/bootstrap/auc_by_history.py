# auc_by_history.py – AUC mức câu theo vị trí của câu trong chuỗi test của học sinh
# Mục đích: xem C kém A ở câu đầu tiên được chấm (câu 1 KC, ΔAUC khoảng −0.007) chỉ xảy ra
# ở vị trí 0 hay kéo dài ở các câu đầu (lịch sử ngắn). Không có bootstrap: vài phút CPU.
# Đặt trong tests/bootstrap/, dùng cùng --runs_root với run_boot.py. Chạy ở gốc repo:
#   python tests/bootstrap/auc_by_history.py --runs_root <bản sao runs> \
#       --seeds 42 43 44 --out tests/bootstrap/summary/history
# Ghi <out>_runs.csv (mỗi lần chạy, mỗi nhóm) và <out>.csv (tổng hợp, Δ so với --ref).
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
 
EDGES = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, np.inf]
LABELS = ["0", "1", "2–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200–499", "≥500"]
 
 
def history_groups(q):
    """vi_tri = số câu được chấm của cùng học sinh đứng trước câu này (0: câu đầu)."""
    q = q.sort_values("qidx")
    pos = q.groupby("uid").cumcount()
    return pd.DataFrame({"qidx": q["qidx"].to_numpy(),
                         "cau": np.where(q["n_kc"].to_numpy() > 1, "nhiều KC", "1 KC"),
                         "vi_tri": pd.cut(pos, EDGES, right=False, labels=LABELS)})
 
 
def auc_by_group(q, groups):
    m = q[["qidx", "y_true", "p_late_mean"]].merge(groups, on="qidx")
    rows = []
    for (cau, vt), part in m.groupby(["cau", "vi_tri"], observed=True):
        ok = part["y_true"].nunique() == 2              # AUC cần cả nhãn đúng và sai
        rows.append({"cau": cau, "vi_tri": vt, "so_cau": len(part),
                     "auc": roc_auc_score(part["y_true"], part["p_late_mean"])
                     if ok else np.nan})
    return pd.DataFrame(rows)
 
 
def main(a):
    reg = pd.read_csv(os.path.join(a.runs_root, "registry", "registry.csv"))
    reg = reg[(reg["model"] == a.model) & reg["seed"].isin(a.seeds)
              & reg["variant"].isin(a.variants)]
    assert not reg.duplicated(["variant", "fold", "seed"]).any(), "lần chạy bị trùng"
    out = []
    for (fold, seed), r in reg.groupby(["fold", "seed"]):
        r = r.set_index("variant")
        if a.ref not in r.index:
            print(f"Bỏ qua fold {fold}, seed {seed}: không có {a.ref}")
            continue
        path = lambda v: os.path.join(a.runs_root, r.loc[v, "preds_path"])
        ref = pd.read_parquet(path(a.ref))
        groups = history_groups(ref)                    # nhóm lấy từ lần chạy tham chiếu
        for v in r.index:
            q = ref if v == a.ref else pd.read_parquet(path(v))
            assert set(q["qidx"]) == set(ref["qidx"]), f"{v}: khác tập câu với {a.ref}"
            res = auc_by_group(q, groups)
            res["variant"], res["fold"], res["seed"] = v, fold, seed
            out.append(res)
    runs = pd.concat(out, ignore_index=True)
    keys = ["fold", "seed", "cau", "vi_tri"]
    runs["vi_tri"] = pd.Categorical(runs["vi_tri"], LABELS, ordered=True)
    base = runs.loc[runs["variant"] == a.ref, keys + ["auc"]]
    runs = runs.merge(base.rename(columns={"auc": "auc_ref"}), on=keys)
    runs["delta"] = runs["auc"] - runs["auc_ref"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    runs.to_csv(f"{a.out}_runs.csv", index=False)
    summ = runs.groupby(["variant", "cau", "vi_tri"], observed=True).agg(
        so_cau=("so_cau", "first"), auc_tb=("auc", "mean"), delta_tb=("delta", "mean"),
        nho_nhat=("delta", "min"), lon_nhat=("delta", "max"),
        so_lan_am=("delta", lambda x: int((x < 0).sum())), so_lan_chay=("delta", "size"))
    summ.round(5).to_csv(f"{a.out}.csv")
    print(summ.round(4).to_string())
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--model", default="DKT")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--variants", nargs="+", default=["A", "C", "ML_cus", "Fuse"])
    ap.add_argument("--ref", default="A")
    ap.add_argument("--out", default="tests/bootstrap/summary/history")
    main(ap.parse_args())
