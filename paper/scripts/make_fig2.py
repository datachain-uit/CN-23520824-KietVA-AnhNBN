# Fig. 2: KC-level AUC, first KC vs KC 2+ (mean, error bar = sd over runs)
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
D = "."   # thư mục chứa các csv (summary/)
def pm(s):                       # "0.8008 ± 0.0006" -> (mean, sd)
    a = s.split("±"); return float(a[0]), float(a[1]) if len(a) > 1 else 0.0
k = pd.read_csv(f"{D}/kcpos.csv")
names = {"A":"DKT","ML":"DKT-ML","ML_cus":"DKT-ML-PQC","C":"DKT-PQC","Fuse":"DKT-Fuse"}
rows = []
for v, n in names.items():
    r = k[k.variant == v].iloc[0]; rows.append((n, *pm(r.kc_dau), *pm(r.kc_2)))
m = pd.read_csv(f"{D}/kcpos_models.csv"); s = pd.read_csv(f"{D}/kcpos_simplekt_pqc.csv")
def ms(d, lab): return (lab, d["KC đầu"].mean(), d["KC đầu"].std(), d["KC 2+"].mean(), d["KC 2+"].std())
g = lambda model: m[m.model == model]
rows += [ms(g("DKT"), "DKT (pyKT)"), ms(g("SIMPLEKT"), "simpleKT"),
         ms(s[s.variant == "C"], "simpleKT-PQC"), ms(g("AKT"), "AKT")]
df = pd.DataFrame(rows, columns=["model", "f", "fsd", "l", "lsd"])
x = np.arange(len(df)); w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.bar(x - w/2, df.f, w, yerr=df.fsd, capsize=2, label="first KC")
ax.bar(x + w/2, df.l, w, yerr=df.lsd, capsize=2, label="KC 2+")
ax.set_xticks(x); ax.set_xticklabels(df.model, rotation=30, ha="right", fontsize=7)
ax.set_ylim(0.65, 0.85); ax.set_ylabel("KC-level AUC", fontsize=8)
ax.legend(fontsize=7, frameon=False); ax.tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig("fig2_kcpos.pdf"); print(df.round(4).to_string())
