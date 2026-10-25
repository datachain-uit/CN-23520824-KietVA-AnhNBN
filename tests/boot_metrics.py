# boot_metrics.py – paired bootstrap theo học sinh cho ΔAUC, ΔBrier, ΔECE
# (mở rộng paired_bootstrap_auc_fast trong tests/bootstrap.ipynb)
import numpy as np
import pandas as pd
 
 
def _wauc(inv, g, y, w):                   # AUC có trọng số, xử lý điểm trùng như sklearn
    wp = np.bincount(inv, weights=w * y, minlength=g)
    wn = np.bincount(inv, weights=w * (1 - y), minlength=g)
    return (wp * (np.cumsum(wn) - 0.5 * wn)).sum() / (wp.sum() * wn.sum())
 
 
def _wece(bins, p, y, w, n_bins=10):       # cùng cách chia bin với hàm ECE của nhóm
    sy = np.bincount(bins, weights=w * y, minlength=n_bins)
    sp = np.bincount(bins, weights=w * p, minlength=n_bins)
    return np.abs(sy - sp).sum() / w.sum()
 
 
def paired_bootstrap(qa, qb, n_boot=1000, seed=0, subset=None):
    """qa, qb: preds.parquet của hai lần chạy (b: phương pháp mới). subset: tập qidx con
    để phân tích theo nhóm câu hỏi. Trả về Δ = b − a và khoảng tin cậy 95%."""
    m = qa[["qidx", "uid", "y_true", "p_late_mean"]].merge(
        qb[["qidx", "p_late_mean"]], on="qidx", suffixes=("_a", "_b"))
    assert len(m) == len(qa) == len(qb), "hai lần chạy phải có cùng tập câu hỏi"
    if subset is not None:
        m = m[m["qidx"].isin(subset)]
    y = m["y_true"].to_numpy(float)
    u, _ = pd.factorize(m["uid"])
    edges = np.linspace(0, 1, 11)[1:-1]
    prep = []
    for col in ("p_late_mean_a", "p_late_mean_b"):
        p = m[col].to_numpy(float)
        _, inv = np.unique(p, return_inverse=True)
        bins = np.clip(np.digitize(p, edges, right=True), 0, 9)
        prep.append((p, inv, inv.max() + 1, bins))
 
    def stats(w):
        out = [[_wauc(inv, g, y, w), np.sum(w * (p - y) ** 2) / w.sum(),
                _wece(bins, p, y, w)] for p, inv, g, bins in prep]
        return np.array(out[1]) - np.array(out[0])
 
    d0 = stats(np.ones(len(y)))
    rng = np.random.default_rng(seed)
    n_u = u.max() + 1
    ds = np.array([stats(np.bincount(rng.integers(0, n_u, n_u), minlength=n_u)[u]
                         .astype(float)) for _ in range(n_boot)])
    lo, hi = np.percentile(ds, [2.5, 97.5], axis=0)
    return pd.DataFrame({"metric": ["ΔAUC", "ΔBrier", "ΔECE"], "delta": d0,
                         "lo": lo, "hi": hi, "n_cau": len(m)})
 
 
def strata(q):
    """Nhóm câu hỏi để kiểm tra cơ chế: số KC của câu được dự đoán và của câu trước."""
    s = q.set_index("qidx")
    prev = s["n_kc"].reindex(q["qidx"] - 1).to_numpy()        # qidx tăng 1 mỗi câu
    same = s["uid"].reindex(q["qidx"] - 1).to_numpy() == q["uid"].to_numpy()
    return pd.DataFrame({"qidx": q["qidx"].to_numpy(),
                         "cau": np.where(q["n_kc"].to_numpy() > 1, "nhiều KC", "1 KC"),
                         "cau_truoc": np.where(~same, "không rõ",
                                               np.where(prev > 1, "nhiều KC", "1 KC"))})
 
# Ví dụ: s = strata(qa); for v, part in s.groupby("cau"):
#            print(v, paired_bootstrap(qa, qc, subset=part["qidx"]))
