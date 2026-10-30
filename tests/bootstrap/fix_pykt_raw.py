# fix_pykt_raw.py – thêm cột rest (vị trí KC trong câu) vào preds_raw.parquet của pyKT
# pyKT ghi các KC của một câu theo thứ tự NGƯỢC với test.csv (evaluate_model.py, hàm
# group_fusion duyệt các dòng từ cuối lên), nên không suy vị trí KC từ thứ tự dòng được.
# Sửa các lần chạy cũ (một lần, chạy ở máy có bản sao runs/ và test.csv):
#   python fix_pykt_raw.py --test_csv <dữ liệu>/test.csv --runs <runs>/DKT_pyKT_f0_s42_...
# pykt_to_registry.py bản 28/10 gọi hai hàm dưới đây, nên lần chạy mới đã có cột rest.
import argparse, os
import numpy as np
import pandas as pd
 
 
def kc_order(test_csv):
    """Mỗi KC của test.csv: qidx (đánh số như pyKT), mã KC, rest = số KC còn sau nó."""
    t = pd.read_csv(test_csv, usecols=["concepts", "is_repeat"])
    parts, nxt = [], 0
    for cs, rep in zip(t["concepts"], t["is_repeat"]):
        cs = np.array(cs.split(","), dtype=int)
        rep = np.array(rep.split(","), dtype=int)
        st = np.flatnonzero(rep == 0)                   # KC đầu của mỗi câu
        g = np.cumsum(rep == 0) - 1                     # câu thứ mấy trong dòng
        n = np.bincount(g)                              # số KC của mỗi câu
        pos = np.arange(len(cs)) - st[g]
        parts.append(pd.DataFrame({"qidx": nxt + g, "concept": cs,
                                   "rest": n[g] - 1 - pos}))
        nxt += len(st)
    ref = pd.concat(parts, ignore_index=True)
    ref["k"] = ref.groupby(["qidx", "concept"]).cumcount()   # KC lặp lại trong một câu
    return ref
 
 
def kc_rest(kc, ref):
    """Gắn rest cho từng dòng (qidx, concept) của preds_raw pyKT; kiểm tra đủ."""
    kc = kc.drop(columns="rest", errors="ignore")
    k = kc.groupby(["qidx", "concept"]).cumcount(ascending=False)   # pyKT ghi ngược
    m = kc.assign(k=k).merge(ref, on=["qidx", "concept", "k"], how="left",
                             validate="one_to_one")
    assert m["rest"].notna().all(), "có KC trong preds_raw không có trong test.csv"
    s = m.groupby("qidx")["rest"].agg(["nunique", "size", "max"])
    n_ref = ref.groupby("qidx").size().reindex(s.index)
    ok = (s["nunique"] == s["size"]) & (s["max"] == s["size"] - 1) & (s["size"] == n_ref)
    assert ok.all(), "rest không phải 0..n-1, hoặc câu thiếu KC so với test.csv"
    return m.drop(columns="k").astype({"rest": "int16"})
 
 
def main(a):
    ref = kc_order(a.test_csv)
    for d in a.runs:
        p = os.path.join(d, "preds_raw.parquet")
        raw = pd.read_parquet(p)
        out = kc_rest(raw, ref)
        n = out.groupby("qidx")["qidx"].transform("size")
        row = out.groupby("qidx").cumcount()            # thứ tự dòng trong file gốc
        multi = n > 1
        rev = (out.loc[multi, "rest"] == row[multi]).groupby(out.loc[multi, "qidx"]).all()
        out.to_parquet(p, index=False)
        print(f"{d}: {len(out)} dòng; câu nhiều KC có thứ tự dòng ngược: "
              f"{rev.mean():.1%} ({rev.sum()}/{len(rev)})")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--runs", nargs="+", required=True)   # thư mục run của pyKT
    main(ap.parse_args())
