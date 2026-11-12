# make_table4.py – sinh Table 4 (ΔAUC theo nhóm câu) cho cả hai backbone (12/11).
# Cột DKT-PQC − DKT lấy từ summary/strata.csv (cặp "C − A", 15 lần chạy); cột
# simpleKT-PQC − simpleKT lấy từ summary_skt/strata.csv (cặp "C − pyKT", 5 lần chạy).
# --repeat_folds (tùy chọn): file *_folds.csv của repeat_diag.py; thêm khối "lặp liền kề"
# chỉ cho simpleKT (DKT dùng is_repeat nên không có quy ước gộp). Ô: ΔAUC (số lần dương;
# số lần CI > 0 / CI < 0); in đậm khi "rõ" (cùng dấu ở mọi lần, CI loại trừ 0 ở hơn
# một nửa số lần). Chạy ở gốc repo:
#   python paper/scripts/make_table4.py --dkt tests/bootstrap/summary/strata.csv \
#       --skt tests/bootstrap/summary_skt/strata.csv \
#       --repeat_folds tests/bootstrap/out_skt/repeat_kc_folds.csv \
#       --out paper/tables/table4.tex
import argparse
import pandas as pd

ROWS = [  # (nhãn trong bài, nhóm trong strata.csv)
    ("1 KC / all", "cau=1 KC"),
    ("multi-KC / all", "cau=nhiều KC"),
    None,
    ("1 KC / 1 KC", "cheo=1 KC / trước 1 KC"),
    ("1 KC / multi-KC", "cheo=1 KC / trước nhiều KC"),
    ("multi-KC / 1 KC", "cheo=nhiều KC / trước 1 KC"),
    ("multi-KC / multi-KC", "cheo=nhiều KC / trước nhiều KC"),
    None,
    ("1 KC / first graded question", "cheo=1 KC / trước không rõ"),
    ("multi-KC / first graded question", "cheo=nhiều KC / trước không rõ"),
    ("all first graded questions", "cau_truoc=không rõ"),
]
REPEAT = [("immediate repeat / all", "lặp liền kề"), ("not a repeat / all", "không lặp"),
          ("1 KC, immediate repeat", "1 KC, lặp liền kề"),
          ("1 KC, not a repeat", "1 KC, không lặp")]


def num(v):
    s = f"{v:+.4f}"
    return s.replace("-", "$-$")


def cell(v, pos, ci_hi, ci_lo, n):
    k = n // 2 + 1                                   # 8 của 15, 3 của 5
    clear = (pos == n and ci_hi >= k) or (pos == 0 and ci_lo >= k)
    s = f"{num(v)} ({pos}; {ci_hi} / {ci_lo})"
    return f"\\textbf{{{s}}}" if clear else s


def from_strata(path, pair):
    d = pd.read_csv(path)
    d = d[d["so_sanh"] == pair].set_index("nhom")
    if d.empty:
        raise SystemExit(f"{path}: không có cặp '{pair}'")
    return d


def strata_cell(d, nhom):
    r = d.loc[nhom]
    return cell(r["hieu_tb"], int(r["tot_hon"]), int(r["ci_tot"]), int(r["ci_kem"]),
                int(r["so_lan_chay"])), int(r["so_cau_tb"])


def repeat_cells(path):
    f = pd.read_csv(path)
    f = f[f["metric"] == "ΔAUC"]
    out = {}
    for nhom, g in f.groupby("nhom"):
        out[nhom] = (cell(g["delta"].mean(), int((g["delta"] > 0).sum()),
                          int((g["lo"] > 0).sum()), int((g["hi"] < 0).sum()), len(g)),
                     int(round(g["n_cau"].mean())))
    return out


def main(a):
    dkt, skt = from_strata(a.dkt, a.dkt_pair), from_strata(a.skt, a.skt_pair)
    lines = [r"\begin{tabular}{@{}l r l l@{}}", r"\toprule",
             r"Target / previous question & \#questions & DKT-PQC $-$ DKT (of 15)"
             r" & simpleKT-PQC $-$ simpleKT (of 5) \\", r"\midrule"]
    for row in ROWS:
        if row is None:
            lines.append(r"\midrule")
            continue
        label, nhom = row
        c1, n1 = strata_cell(dkt, nhom)
        c2, n2 = strata_cell(skt, nhom)
        if n1 != n2:
            raise SystemExit(f"{nhom}: số câu khác nhau ({n1} và {n2})")
        lines.append(f"{label} & {n1:,}".replace(",", r"\,") + f" & {c1} & {c2} \\\\")
    if a.repeat_folds:
        rep = repeat_cells(a.repeat_folds)
        lines.append(r"\midrule")
        for label, nhom in REPEAT:
            if nhom not in rep:
                print(f"bỏ dòng '{label}': {a.repeat_folds} không có nhóm '{nhom}'")
                continue
            c, n = rep[nhom]
            lines.append(f"{label} & {n:,}".replace(",", r"\,") + f" & -- & {c} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dkt", default="tests/bootstrap/summary/strata.csv")
    ap.add_argument("--skt", default="tests/bootstrap/summary_skt/strata.csv")
    ap.add_argument("--dkt_pair", default="C − A")
    ap.add_argument("--skt_pair", default="C − pyKT")
    ap.add_argument("--repeat_folds", default="")
    ap.add_argument("--out", default="paper/tables/table4.tex")
    main(ap.parse_args())
