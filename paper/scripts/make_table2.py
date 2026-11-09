# make_table2.py – sinh Table 2 của bài (kết quả chính) từ registry.csv, không gõ tay số.
# Tên biến thể trong bài theo quyết định 7.2 ngày 07/11; registry giữ mã cũ (A, C, ...).
# Đặt trong paper/scripts/. Chạy ở gốc repo:
#   python paper/scripts/make_table2.py --approved results/runs/registry/approved.csv \
#       --runs_root tests/bootstrap/runs --out paper/tables/table2
# --runs_root: tính lại ACC từ preds.parquet với ngưỡng >= 0.5 (m14); bỏ trống thì lấy
# cột acc của registry và in cảnh báo. Ghi <out>.tex (bảng LaTeX) và <out>.csv.
# In đậm KHÔNG làm ở đây: chỉ in đậm tay khi summarize_boot.py cho mức "rõ".
import argparse, os
import pandas as pd

NAMES = {  # (model, variant) -> (nhóm, thứ tự, tên trong bài)
    ("DKT", "A"): (1, 1, "DKT"),
    ("DKT", "B"): (1, 2, "DKT, leak-free selection"),
    ("DKT", "ML"): (1, 3, "DKT-ML"),
    ("DKT", "ML_cus"): (1, 4, "DKT-ML-PQC"),
    ("DKT", "C"): (1, 5, "DKT-PQC"),
    ("DKT", "Fuse"): (1, 6, "DKT-Fuse"),
    ("DKT", "pyKT"): (2, 1, "DKT (pyKT)"),
    ("AKT", "pyKT"): (2, 2, "AKT (pyKT)"),
    ("QIKT", "pyKT"): (2, 3, "QIKT (pyKT)"),
    ("SIMPLEKT", "pyKT"): (3, 1, "simpleKT (pyKT)"),
    ("SIMPLEKT", "C"): (3, 2, "simpleKT-PQC"),
}
# AUC trong Table 2 của bài XES3G5M (đối chiếu)
REF = {"DKT": "0.7852", "AKT": "0.8207", "SIMPLEKT": "0.8163", "QIKT": "0.8222"}


def acc_from_preds(root, path):
    q = pd.read_parquet(os.path.join(root, path), columns=["y_true", "p_late_mean"])
    return float(((q["p_late_mean"] >= 0.5).astype(int) == q["y_true"]).mean())


def main(a):
    reg = pd.read_csv(a.registry)
    if a.approved:
        ok = set(pd.read_csv(a.approved)["run_id"])
        print(f"bỏ {(~reg['run_id'].isin(ok)).sum()} lần chạy chưa duyệt")
        reg = reg[reg["run_id"].isin(ok)]
    reg = reg[[(m, v) in NAMES for m, v in zip(reg["model"], reg["variant"])]].copy()
    if a.runs_root:
        reg["acc"] = [acc_from_preds(a.runs_root, p) for p in reg["preds_path"]]
    else:
        print("CẢNH BÁO: ACC lấy từ registry (chưa tính lại với ngưỡng >= 0.5)")
    rows = []
    for (m, v), g in reg.groupby(["model", "variant"]):
        grp, order, name = NAMES[(m, v)]
        sd = lambda c: g[c].std(ddof=1) if len(g) > 1 else float("nan")
        rows.append(dict(nhom=grp, thu_tu=order, ten=name, n=len(g),
                         auc=g["auc"].mean(), auc_sd=sd("auc"), acc=g["acc"].mean(),
                         brier=g["brier"].mean(), ece=g["ece"].mean(), ece_sd=sd("ece"),
                         params=int(g["n_params"].iloc[0]),
                         ref=REF.get(m, "--") if v == "pyKT" else "--"))
    t = pd.DataFrame(rows).sort_values(["nhom", "thu_tu"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(f"{a.out}.csv", index=False)
    pm = lambda x, s: f"{x:.4f}" + ("" if pd.isna(s) else f" $\\pm$ {s:.4f}")
    lines = [r"\begin{tabular}{@{}lrllllrl@{}}", r"\toprule",
             r"Model & Runs & AUC & ACC & Brier & ECE & Params & Ref.\ AUC \\",
             r"\midrule"]
    last = None
    for r in t.itertuples():
        if last is not None and r.nhom != last:
            lines.append(r"\midrule")
        last = r.nhom
        lines.append(f"{r.ten} & {r.n} & {pm(r.auc, r.auc_sd)} & {r.acc:.4f} & "
                     f"{r.brier:.4f} & {pm(r.ece, r.ece_sd)} & {r.params:,} & "
                     f"{r.ref} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{a.out}.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="results/runs/registry/registry.csv")
    ap.add_argument("--approved", default="")
    ap.add_argument("--runs_root", default="")
    ap.add_argument("--out", default="paper/tables/table2")
    main(ap.parse_args())
