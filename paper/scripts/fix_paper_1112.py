# fix_paper_1112.py – các sửa cơ học cho bản thảo 11/11 (nhận xét 12/11). Chạy một lần:
#   python paper/scripts/fix_paper_1112.py paper
# Mỗi sửa thay đúng một chuỗi; nếu không thấy chuỗi (đã sửa tay trên Overleaf) thì in
# "không thấy" và bỏ qua. Không đụng tới phần nội dung nhóm phải tự viết (mục 6.6, 7, 8).
import sys
from pathlib import Path

BP = r"\cite{DBLP:conf/csedu/BadranP25a}"
FIXES = [
    # (file, chuỗi cũ, chuỗi mới, mô tả)
    ("sections/04_rule.tex", r"\cite{piech2015dkt}", r"\cite{piech2015deep}", "khóa DKT"),
    ("sections/04_rule.tex", r"\cite{badran2025}", BP, "khóa Badran 2025 (3 chỗ)"),
    ("sections/04_rule.tex", r"\cite{liu2023simplekt,liu2022pykt}",
     r"\cite{DBLP:conf/iclr/0001L0H023,liu2022pykt}", "khóa simpleKT"),
    ("sections/04_rule.tex", r"Sect.~\ref{sec:rule-def}", r"Sect.~\ref{subsec:rule-def}",
     "nhãn mục 4.1"),
    ("sections/04_rule.tex", r"\paragraph{DKT-Fuse.} DKT-Fuse. This variant",
     r"\paragraph{DKT-Fuse.} This variant", "chữ lặp"),
    ("sections/04_rule.tex", r"\paragraph{DKT-ML-PQC.} this use MASK labels",
     r"\paragraph{DKT-ML-PQC.} This variant uses MASK labels", "ngữ pháp"),
    ("sections/05_setup.tex", r"3\,163 students", r"3\,613 students", "số học sinh test"),
    ("sections/05_setup.tex",
     "do not leak, they are run on the five DKT variants (fold 0, seed 42).",
     "do not leak (fold 0, seed 42); the intervention test is run on DKT and DKT-PQC,"
     " the other checks on the five DKT variants.", "phạm vi kiểm thử can thiệp"),
    ("sections/06_results.tex",
     "Bold: clear difference to the standard model of the same backbone"
     " (DKT or simpleKT). ",
     r"Paired tests: Table~\ref{tab:decomp} (DKT) and"
     r" Sect.~\ref{subsec:res-skt} (simpleKT). ",
     "chú thích Table 2 (bảng không in đậm)"),
    ("sections/06_results.tex", r"(Tables~\ref{tab:main})", r"(Table~\ref{tab:main})",
     "số ít"),
    ("sections/06_results.tex", "Overfit 4 sequences, loss after 300 steps & 0.0013 &",
     "Overfit 4 sequences, loss after 300 steps & 0.0014 &",
     "overfit DKT chạy lại (m19)"),
    ("sections/06_results.tex",
     "(target / previous question), DKT-PQC $-$ DKT, mean over 15 runs.",
     "(target / previous question): DKT-PQC $-$ DKT (mean over 15 runs) and simpleKT-PQC"
     r" $-$ simpleKT (5 runs; last block: immediate repeats,"
     r" Sect.~\ref{subsec:res-skt}).",
     "chú thích Table 4 hai backbone"),
    ("sections/06_results.tex", "  \\input{tables/table4.tex}%",
     "  \\resizebox{\\textwidth}{!}{\\input{tables/table4.tex}}%", "Table 4 vừa khổ"),
    ("sections/02_related.tex",
     "They evaluate with KTbench, which scores the expanded sequence with the mask"
     " and then averages the KC predictions, not with the all-in-one protocol of pyKT. ",
     "They also propose an autoregressive variant (DKT-AD) that feeds the model's own"
     " predictions instead of the labels of the earlier KCs, and report that leakage"
     " grows with the number of KCs per question. With KTbench they test the leaking"
     " baselines (DKT, AKT) all-in-one and evaluate their leak-free variants one-by-one,"
     " both with the mean over KCs, on ASSISTments2009, Algebra2005, Riiid2020 and"
     " Duolingo2018. ",
     "mô tả đúng cách đánh giá của Badran & Preisach"),
    ("sections/02_related.tex",
     "Second, evaluation follows the pyKT all-in-one protocol with 3 seeds, 5 folds and a"
     " bootstrap over students, instead of KTbench.",
     "Second, every model, leak-free variants included, is evaluated with the same pyKT"
     " all-in-one protocol on XES3G5M, with 3 seeds, 5 folds and a bootstrap over"
     " students.",
     "điểm khác thứ hai"),
]
DROP = [  # dòng chú thích cần xóa (khớp đầu dòng)
    ("sections/04_rule.tex", "% Table 1 – thay bảng tab:design"),
    ("sections/04_rule.tex", "% Cột Params lấy từ registry.csv"),
    ("sections/04_rule.tex", "% DKT-AD của Badran & Preisach"),
    ("sections/06_results.tex", "% TODO (review m19)"),
    ("sections/06_results.tex", "% TODO (6.6)"),
    ("main.tex", "% TODO: xác nhận danh sách"),
]


def main(root):
    root = Path(root)
    for f, old, new, why in FIXES:
        p = root / f
        s = p.read_text(encoding="utf-8")
        n = s.count(old)
        if n == 0:
            print(f"không thấy  {f}: {why}")
            continue
        p.write_text(s.replace(old, new), encoding="utf-8")
        print(f"đã sửa ({n}) {f}: {why}")
    for f, start in DROP:
        p = root / f
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        keep = [ln for ln in lines if not ln.startswith(start)]
        if len(keep) < len(lines):
            p.write_text("".join(keep), encoding="utf-8")
            print(f"đã xóa      {f}: {start}")


if __name__ == "__main__":
    main(sys.argv[1])
