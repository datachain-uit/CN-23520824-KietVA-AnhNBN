#!/usr/bin/env python3
"""Sinh Table 4 (6.3): dAUC theo loai cau, tu strata.csv.
Dung:  python make_table4.py strata.csv tables/table4.tex [strata_skt.csv]
File thu 2 (tuy chon, cho simpleKT) cung dinh dang; lay so_sanh dau tien trong file do."""
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
s = pd.read_csv(src); s2 = pd.read_csv(sys.argv[3]) if len(sys.argv) > 3 else None
ROWS = [("cau=1 KC", "1 KC / all"), ("cau=nhiều KC", "multi-KC / all"), None,
        ("cheo=1 KC / trước 1 KC", "1 KC / 1 KC"), ("cheo=1 KC / trước nhiều KC", "1 KC / multi-KC"),
        ("cheo=nhiều KC / trước 1 KC", "multi-KC / 1 KC"), ("cheo=nhiều KC / trước nhiều KC", "multi-KC / multi-KC"), None,
        ("cheo=1 KC / trước không rõ", "1 KC / first graded question"),
        ("cheo=nhiều KC / trước không rõ", "multi-KC / first graded question"),
        ("cau_truoc=không rõ", "all first graded questions")]
def sub(d, cmp): return d[d.so_sanh == cmp].set_index("nhom")
a = sub(s, "C − A"); b = sub(s2, s2.so_sanh.iloc[0]) if s2 is not None else None
def sg(x): return ("$-$" if x < 0 else "+") + f"{abs(x):.4f}"
def cell(r):
    t = f"{sg(r.hieu_tb)} ({int(r.tot_hon)}; {int(r.ci_tot)} / {int(r.ci_kem)})"
    return f"\\textbf{{{t}}}" if r.muc == "rõ" else t
n = int(a.so_lan_chay.iloc[0])
head = f"DKT-PQC $-$ DKT (of {n})" + (f" & simpleKT-PQC $-$ simpleKT (of {int(b.so_lan_chay.iloc[0])})" if b is not None else "")
L = [r"\begin{tabular}{@{}l r l" + ("l" if b is not None else "") + "@{}}", r"\toprule",
     f"Target / previous question & \\#questions & {head} \\\\", r"\midrule"]
for it in ROWS:
    if it is None: L.append(r"\midrule"); continue
    k, lab = it; r = a.loc[k]
    nq = f"{int(round(r.so_cau_tb)):,}".replace(",", r"\,")
    row = f"{lab} & {nq} & {cell(r)}"
    if b is not None: row += " & " + (cell(b.loc[k]) if k in b.index else "--")
    L.append(row + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(dst, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("wrote", dst)
