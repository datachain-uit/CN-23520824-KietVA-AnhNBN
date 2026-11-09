# summarize_boot.py – gộp registry và kết quả run_boot.py thành 4 bảng để đọc và viết bài
# Chạy ở gốc repo (cần pandas), sau khi đã gộp mọi registry phiên vào registry.csv:
#   python tests/bootstrap/summarize_boot.py --boot_dir tests/bootstrap/out \
#       --seeds 42 43 44 --out tests/bootstrap/summary
# Ghi 4 file CSV vào --out và in ra màn hình. Cách đọc: mục 5 của nhận xét 31/10.
# Bản 02/11: thêm --approved results/runs/registry/approved.csv để main.csv chỉ gồm
# các lần chạy đã duyệt (chạy run_boot.py với cùng --approved để các file boot khớp).
#   main.csv   mỗi mô hình, biến thể: số lần chạy, AUC (trung bình ± độ lệch chuẩn mẫu
#              trên mọi lần chạy), AUC trung bình theo seed, ECE, Brier, AUPRC lớp sai
#   pairs.csv  mỗi cặp "b − a" và mỗi thước đo: hiệu trung bình, nhỏ nhất, lớn nhất;
#              số lần b tốt hơn; số lần CI 95% loại trừ 0 về phía b tốt hơn / b kém hơn
#   kcpos.csv  câu nhiều KC: AUC mức KC ở KC đầu, ở KC thứ 2 trở đi, và chênh lệch
#   strata.csv ΔAUC theo nhóm câu (cau, cau_truoc) và nhóm chéo (cheo)
# Quy ước hướng: AUC cao là tốt; Brier, ECE thấp là tốt. Cột "tot_hon", "ci_tot", "ci_kem"
# đã tính theo hướng này, nên với ΔBrier, ΔECE thì hiệu âm được đếm là "b tốt hơn".
import argparse, os
import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"ΔBrier", "ΔECE"}


def mean_sd(s):
    if len(s) == 1:                            # ví dụ DKT pyKT chỉ có một dòng kcpos
        return f"{s.iloc[0]:.4f} (1 lần)"
    return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"


def read_boot(boot_dir, kind, seeds):
    parts = [pd.read_csv(os.path.join(boot_dir, f"boot_s{s}_{kind}.csv")) for s in seeds]
    return pd.concat(parts, ignore_index=True)


def add_direction(d):
    """Thêm 3 cột 0/1 theo hướng 'b tốt hơn a' cho từng dòng của pairs hoặc strata."""
    low = d["metric"].isin(LOWER_IS_BETTER)
    d["tot_hon"] = np.where(low, d["delta"] < 0, d["delta"] > 0).astype(int)
    d["ci_tot"] = np.where(low, d["hi"] < 0, d["lo"] > 0).astype(int)
    d["ci_kem"] = np.where(low, d["lo"] > 0, d["hi"] < 0).astype(int)
    return d


def count_table(d, keys):
    t = d.groupby(keys).agg(hieu_tb=("delta", "mean"), nho_nhat=("delta", "min"),
                            lon_nhat=("delta", "max"), tot_hon=("tot_hon", "sum"),
                            ci_tot=("ci_tot", "sum"), ci_kem=("ci_kem", "sum"),
                            so_lan_chay=("delta", "size"), so_cau_tb=("n_cau", "mean"))
    t["so_cau_tb"] = t["so_cau_tb"].round().astype(int)
    t["muc"] = [level(r) for r in t.itertuples()]
    t["nhan_xet"] = [describe(r) for r in t.itertuples()]
    return t


def level(r):
    """Mức dùng khi viết (mục 5.3): "rõ" nếu cùng dấu ở mọi lần chạy và CI loại trừ 0
    ở quá nửa số lần chạy; "xu hướng" nếu cùng dấu nhưng chưa đủ; còn lại "chưa phân
    biệt". Không thay tiêu chí của giả thuyết (H1a: ΔAUC ≥ 0.003 và CI không chứa 0)."""
    n = r.so_lan_chay
    if r.tot_hon not in (0, n):
        return "chưa phân biệt"
    k = r.ci_tot if r.tot_hon == n else r.ci_kem
    return "rõ" if k > n / 2 else "xu hướng"


def describe(r):
    n = r.so_lan_chay
    if r.tot_hon == n:
        return f"b tốt hơn ở {n}/{n} lần; CI loại trừ 0 ở {r.ci_tot}/{n}"
    if r.tot_hon == 0:
        return f"b kém hơn ở {n}/{n} lần; CI loại trừ 0 ở {r.ci_kem}/{n}"
    return f"đổi dấu: b tốt hơn {r.tot_hon}/{n}; CI tốt {r.ci_tot}, CI kém {r.ci_kem}"


def main(a):
    os.makedirs(a.out, exist_ok=True)
    reg = pd.read_csv(a.registry)
    reg = reg[reg["seed"].isin(a.seeds)]
    if a.approved:
        ok = set(pd.read_csv(a.approved)["run_id"])
        print(f"bỏ {(~reg['run_id'].isin(ok)).sum()} lần chạy chưa có trong approved.csv")
        reg = reg[reg["run_id"].isin(ok)]
    dup = reg[reg.duplicated(["model", "variant", "fold", "seed"], keep=False)]
    if len(dup):
        raise SystemExit(f"Dừng: một (mô hình, biến thể, fold, seed) có nhiều dòng:\n"
                         f"{dup[['run_id', 'auc']]}")
    g = reg.groupby(["model", "variant"])
    main_t = pd.DataFrame({"so_lan_chay": g.size(), "auc": g["auc"].agg(mean_sd),
                           "ece": g["ece"].agg(mean_sd), "brier": g["brier"].mean(),
                           "auprc_sai": g["auprc_wrong"].mean(),
                           "so_tham_so": g["n_params"].first()})
    by_seed = reg.pivot_table(index=["model", "variant"], columns="seed", values="auc",
                              aggfunc="mean")
    by_seed.columns = [f"auc_s{c}" for c in by_seed.columns]
    main_t = main_t.join(by_seed).loc[g["auc"].mean().sort_values().index]

    pairs = count_table(add_direction(read_boot(a.boot_dir, "pairs", a.seeds)),
                        ["so_sanh", "metric"])

    k = read_boot(a.boot_dir, "kcpos", a.seeds)
    k["chenh"] = k["KC 2+"] - k["KC đầu"]
    kcpos = k.groupby("variant").agg(kc_dau=("KC đầu", mean_sd),
                                     kc_2=("KC 2+", mean_sd), chenh=("chenh", mean_sd),
                                     so_lan_chay=("chenh", "size"))

    s = read_boot(a.boot_dir, "strata", a.seeds)
    strata = count_table(add_direction(s[s["metric"] == a.strata_metric]),
                         ["so_sanh", "nhom"])

    for name, t in [("main", main_t), ("pairs", pairs), ("kcpos", kcpos),
                    ("strata", strata)]:
        t.round(5).to_csv(os.path.join(a.out, f"{name}.csv"))
        print(f"\n=== {name} ===\n{t.round(5).to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="results/runs/registry/registry.csv")
    ap.add_argument("--boot_dir", default="tests/bootstrap/out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--strata_metric", default="ΔAUC", choices=["ΔAUC", "ΔBrier", "ΔECE"])
    ap.add_argument("--out", default="tests/bootstrap/summary")
    ap.add_argument("--approved", default="")      # rỗng: dùng mọi dòng của registry
    main(ap.parse_args())
