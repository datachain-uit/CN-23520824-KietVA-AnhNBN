#!/usr/bin/env python3
"""Sinh Table 3 (6.2) hoac bang simpleKT (6.6) tu pairs.csv.
Dung:  python make_table3.py pairs.csv tables/table3.tex
       python make_table3.py summary_skt/pairs.csv tables/table_skt.tex   (hang lay tu file)
Chi ghi phan tabular; chen vao bai bang \\input. In dam khi cot 'muc' == 'rõ'."""
import sys, pandas as pd

src, dst = sys.argv[1], sys.argv[2]
p = pd.read_csv(src)
NAME = {"A": "DKT", "B": "DKT, leak-free selection", "C": "DKT-PQC", "ML": "DKT-ML",
        "ML_cus": "DKT-ML-PQC", "Fuse": "DKT-Fuse"}
# (so_sanh trong file, nhan hien thi) theo nhom; nhom ngan cach bang \midrule
GROUPS = [
    [("ML − A", "DKT-ML $-$ DKT (i)"), ("ML_cus − ML", "DKT-ML-PQC $-$ DKT-ML (ii)"),
     ("C − ML_cus", "DKT-PQC $-$ DKT-ML-PQC (label)"), ("C − A", "DKT-PQC $-$ DKT (total)")],
    [("ML_cus − A", "DKT-ML-PQC $-$ DKT"), ("Fuse − A", "DKT-Fuse $-$ DKT"),
     ("Fuse − C", "DKT-Fuse $-$ DKT-PQC"), ("Fuse − ML_cus", "DKT-Fuse $-$ DKT-ML-PQC")],
    [("C − B", "DKT-PQC $-$ DKT, leak-free selection"), ("B − A", "DKT, leak-free selection $-$ DKT")],
]
if not set(p.so_sanh) & {g[0] for grp in GROUPS for g in grp}:   # file khac (vd. simpleKT)
    GROUPS = [[(s, s.replace("−", "$-$")) for s in p.so_sanh.drop_duplicates()]]

def f(x, d):                       # dau + so thap phan, dau tru dang math
    s = f"{abs(x):.{d}f}"
    return ("$-$" if x < 0 and float(s) != 0 else "+" if float(s) != 0 else "") + s if x else s
def bold(s, on): return f"\\textbf{{{s}}}" if on else s

L = [r"\begin{tabular}{@{}l l c c c@{}}", r"\toprule",
     r"Comparison ($b-a$) & $\Delta$AUC [min; max] & CI & $\Delta$Brier & $\Delta$ECE \\", r"\midrule"]
for gi, grp in enumerate(GROUPS):
    for key, lab in grp:
        r = p[p.so_sanh == key].set_index("metric")
        if r.empty: print("THIEU:", key, file=sys.stderr); continue
        a, b, e = r.loc["ΔAUC"], r.loc["ΔBrier"], r.loc["ΔECE"]
        auc = bold(f(a.hieu_tb, 5), a.muc == "rõ") + f" [{f(a.nho_nhat,5)}; {f(a.lon_nhat,5)}]"
        L.append(f"{lab} & {auc} & {int(a.ci_tot)} / {int(a.ci_kem)} & "
                 f"{bold(f(b.hieu_tb,5), b.muc=='rõ')} & {bold(f(e.hieu_tb,5), e.muc=='rõ')} \\\\")
    if gi < len(GROUPS) - 1: L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open(dst, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("wrote", dst)
