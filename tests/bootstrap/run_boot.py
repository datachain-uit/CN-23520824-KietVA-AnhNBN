# run_boot.py – bootstrap cho N1, N2, đọc preds theo registry (thay run_boot.ipynb rỗng)
# Đặt trong tests/bootstrap/, cạnh boot_metrics.py (Khung 4.8 ngày 25/10). --runs_root
# là bản sao thư mục runs/ trên Drive (có preds.parquet); chép registry.csv của repo vào
# <runs_root>/registry/. Chạy ở gốc repo (bản 28/10):
#   python tests/bootstrap/run_boot.py --runs_root <bản sao runs> --seed 42 \
#       --out tests/bootstrap/out/boot_s42
# Ghi 3 file: <out>_pairs.csv (mọi cặp, mọi fold), <out>_strata.csv (theo nhóm câu, cả
# nhóm chéo câu × câu trước), <out>_kcpos.csv (AUC mức KC theo vị trí KC, câu nhiều KC).
import argparse, os
import pandas as pd
from sklearn.metrics import roc_auc_score
from boot_metrics import paired_bootstrap, strata

PAIRS = [("B", "A"), ("C", "A"), ("C", "B"),                      # N2
         ("ML", "A"), ("ML_cus", "ML"), ("C", "ML_cus"), ("Fuse", "C"),  # N1: phân rã
         ("ML_cus", "A"), ("Fuse", "A"), ("Fuse", "ML_cus")]  # 3 cách hiện thực
STRATA_PAIRS = [("C", "A"), ("ML_cus", "A"), ("ML_cus", "C"), ("Fuse", "A")]


def kc_position_auc(raw):
    """Câu nhiều KC: AUC mức KC của KC đầu và của KC thứ 2 trở đi (cần cột rest)."""
    assert "rest" in raw, "preds_raw thiếu cột rest: chạy fix_pykt_raw.py (28/10)"
    n = raw.groupby("qidx")["qidx"].transform("size")
    pos = n - 1 - raw["rest"]                  # không dùng thứ tự dòng: pyKT ghi ngược
    m = raw[n > 1]
    pos = pos[n > 1]
    return {g: roc_auc_score(m.loc[pos == 0 if g == "KC đầu" else pos > 0, "y_true"],
                             m.loc[pos == 0 if g == "KC đầu" else pos > 0, "p_kc"])
            for g in ("KC đầu", "KC 2+")}


def main(a):
    reg = pd.read_csv(os.path.join(a.runs_root, "registry", "registry.csv"))
    reg = reg[(reg["model"] == a.model) & (reg["seed"] == a.seed)].copy()
    reg["fold"] = reg["fold"].astype(int)
    assert not reg.duplicated(["variant", "fold"]).any(), "(biến thể, fold) bị trùng"
    pairs, strat, kcpos = [], [], []
    for fold, r in reg.groupby("fold"):
        r = r.set_index("variant")
        load = lambda v, col="preds_path": pd.read_parquet(
            os.path.join(a.runs_root, r.loc[v, col]))
        for b, c in PAIRS + [x for x in STRATA_PAIRS if x not in PAIRS]:
            if b not in r.index or c not in r.index:
                continue
            qa, qb = load(c), load(b)
            la = qa.set_index("qidx")["y_true"].astype("int64")
            lb = qb.set_index("qidx")["y_true"].astype("int64")
            assert len(la) == len(lb) and la.equals(lb.reindex(la.index)), \
                f"{b}, {c}: khác nhãn hoặc khác tập câu – kiểm tra Dataset (Khung 4.6)"
            if (b, c) in PAIRS:
                res = paired_bootstrap(qa, qb, a.n_boot, seed=fold)
                res.insert(0, "so_sanh", f"{b} − {c}")
                res.insert(0, "fold", fold)
                res["run_a"], res["run_b"] = r.loc[c, "run_id"], r.loc[b, "run_id"]
                pairs.append(res)
            if (b, c) in STRATA_PAIRS:
                s = strata(qa)
                s["cheo"] = s["cau"] + " / trước " + s["cau_truoc"]   # tách chéo, mục 2.3
                for col in ("cau", "cau_truoc", "cheo"):
                    for g, part in s.groupby(col):
                        x = paired_bootstrap(qa, qb, a.n_boot, seed=fold,
                                             subset=part["qidx"])
                        x.insert(0, "nhom", f"{col}={g}")
                        x.insert(0, "so_sanh", f"{b} − {c}")
                        x.insert(0, "fold", fold)
                        strat.append(x)
        for v in r.index:
            raw = r.loc[v, "preds_raw_path"]
            if isinstance(raw, str) and raw:
                kcpos.append({"fold": fold, "variant": v,
                              **kc_position_auc(load(v, "preds_raw_path"))})
    seed = {"seed": a.seed}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.concat(pairs).assign(**seed).to_csv(f"{a.out}_pairs.csv", index=False)
    if strat:
        pd.concat(strat).assign(**seed).to_csv(f"{a.out}_strata.csv", index=False)
    pd.DataFrame(kcpos).assign(**seed).to_csv(f"{a.out}_kcpos.csv", index=False)
    summary = pd.concat(pairs).groupby(["so_sanh", "metric"])["delta"]
    print(summary.agg(["mean", "min", "max"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="results/runs")
    ap.add_argument("--model", default="DKT")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default="tests/bootstrap/out/boot_s42")
    main(ap.parse_args())
