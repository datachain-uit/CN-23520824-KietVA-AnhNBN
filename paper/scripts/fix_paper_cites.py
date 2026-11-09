# fix_paper_cites.py – sửa các lỗi làm bản thảo biên dịch sai (nhận xét 07/11, mục 6.4):
# khóa trích dẫn không có trong ref.bib, chỗ "[cite]" còn để trống, thiếu amssymb.
# Đổi khóa trong ref.bib sang tên ngắn mà văn bản đang dùng. Chạy một lần ở gốc repo:
#   python paper/scripts/fix_paper_cites.py paper
# Sau đó biên dịch lại (pdflatex, bibtex, pdflatex x2): log không còn "undefined".
import re, sys
from pathlib import Path

BIB_KEYS = {"DBLP:conf/csedu/BadranP25a": "badran2025",
            "badran2026enhancing": "badran2026",
            "DBLP:conf/iclr/0001L0H023": "liu2023simplekt",
            "piech2015deep": "piech2015dkt"}
CITE_KEYS = dict(BIB_KEYS, badran="badran2025", pykt="liu2022pykt")
PLACEHOLDERS = {"(DKT) [cite]": "(DKT)~\\cite{piech2015dkt}",
                "\\paragraph{DKT-ML [cite].}": "\\paragraph{DKT-ML~\\cite{badran2025}.}"}


def fix_cites(text):
    def repl(m):
        keys = [CITE_KEYS.get(k.strip(), k.strip()) for k in m.group(2).split(",")]
        return f"{m.group(1)}{{{','.join(keys)}}}"
    return re.sub(r"(\\cite[tp]?)\{([^}]*)\}", repl, text)


def main(root):
    root = Path(root)
    bib = root / "ref.bib"
    s = bib.read_text(encoding="utf-8")
    for old, new in BIB_KEYS.items():
        s = s.replace("{" + old + ",", "{" + new + ",")
    bib.write_text(s, encoding="utf-8")
    for tex in [root / "main.tex", *sorted((root / "sections").glob("*.tex"))]:
        s = tex.read_text(encoding="utf-8")
        new = fix_cites(s)
        for old, rep in PLACEHOLDERS.items():
            new = new.replace(old, rep)
        if tex.name == "main.tex" and "{amssymb}" not in new:
            new = new.replace("\\usepackage{amsmath}",
                              "\\usepackage{amsmath}\n\\usepackage{amssymb}")
        if new != s:
            tex.write_text(new, encoding="utf-8")
            print("đã sửa", tex)
    left = [t.name for t in (root / "sections").glob("*.tex")
            if "[cite]" in t.read_text(encoding="utf-8")]
    print("còn [cite] trong:", left or "không còn")


if __name__ == "__main__":
    main(sys.argv[1])
