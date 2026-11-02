# kcpos_models.py – AUC mức KC ở KC đầu và ở KC thứ 2 trở đi (câu nhiều KC), cho mọi lần
# chạy có preds_raw: DKT, AKT, simpleKT của pyKT và các biến thể của nhóm. CPU, vài phút.
# Đặt trong tests/bootstrap/, cạnh run_boot.py. Chạy ở gốc repo:
#   python tests/bootstrap/kcpos_models.py --runs_root <bản sao runs> \
#       --models DKT AKT SIMPLEKT --out tests/bootstrap/out/kcpos_models.csv
# Mặc định chỉ lấy các lần chạy pyKT; "--variants" không kèm giá trị thì lấy mọi biến thể.
# preds_raw phải có cột rest (fix_pykt_raw.py hoặc converter bản 28/10); QIKT, IEKT không
# có preds_raw nên không có trong bảng này.
import argparse, os
import pandas as pd
from run_boot import kc_position_auc
 
 
def main(a):
    reg = pd.read_csv(os.path.join(a.runs_root, "registry", "registry.csv"))
    reg = reg[reg["model"].isin(a.models) & reg["preds_raw_path"].notna()]
    if a.variants:
        reg = reg[reg["variant"].isin(a.variants)]
    rows = []
    for r in reg.itertuples():
        raw = pd.read_parquet(os.path.join(a.runs_root, r.preds_raw_path))
        if "rest" not in raw:
            raise SystemExit(f"Dừng: {r.run_id} chưa có cột rest – chạy fix_pykt_raw.py")
        rows.append({"run_id": r.run_id, "model": r.model, "variant": r.variant,
                     "fold": r.fold, "seed": r.seed, **kc_position_auc(raw)})
    out = pd.DataFrame(rows)
    out["chenh"] = out["KC 2+"] - out["KC đầu"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.groupby(["model", "variant"])[["KC đầu", "KC 2+", "chenh"]]
          .agg(["mean", "std", "size"]).round(4).to_string())
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--models", nargs="+", default=["DKT", "AKT", "SIMPLEKT"])
    ap.add_argument("--variants", nargs="*", default=["pyKT"])
    ap.add_argument("--out", default="tests/bootstrap/out/kcpos_models.csv")
    main(ap.parse_args())
