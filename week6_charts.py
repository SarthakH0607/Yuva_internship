"""Week 6: plain-language charts for a non-technical reader. Reuses the data prep and models from model_run.py."""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, ORANGE = "#2b6cb0", "#a0aec0", "#dd6b20"
exec(open("model_run.py").read().split("rows=[]")[0])
os.makedirs("figures", exist_ok=True)
yt = te["mismatch_any"].values
gb = models["Gradient boosting"].fit(X(tr), tr["mismatch_any"])
p = gb.predict_proba(X(te))[:, 1]
base = yt.mean() * 100

# Chart 1: reviews grouped by the model's risk rating
q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=["Lowest\nrisk 20%", "Low", "Middle", "High", "Highest\nrisk 20%"])
band = pd.Series(yt).groupby(q.values).mean() * 100
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(band.index.astype(str), band.values, color=[GREY]*4 + [ORANGE])
for b, v in zip(bars, band.values): ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.0f}%", ha="center", fontweight="bold")
ax.axhline(base, color="black", ls="--", lw=1); ax.text(-0.45, base + 0.4, f"Average: {base:.0f}%", ha="left", fontsize=11)
ax.set_ylabel("Reviews that were a poor fit (%)"); ax.set_xlabel("The model's risk rating for the review")
ax.set_title("Reviews rated highest risk are about twice as likely\nto be a poor fit as those rated lowest", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/1_risk_groups.png", dpi=160); plt.close()

# Chart 2: what flagging buys, per 1,000 reviews
n = 1000; bad = base / 100 * n; flag = 200
top = yt[np.argsort(-p)][: int(len(yt) * flag / n)].mean() * flag
fig, ax = plt.subplots(figsize=(8, 4.5))
vals = [bad * flag / n, top]; labs = ["Check 200 reviews\nchosen at random", "Check the 200 reviews\nthe model rates riskiest"]
bars = ax.bar(labs, vals, color=[GREY, ORANGE], width=0.5)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.0f}", ha="center", fontweight="bold", fontsize=16)
ax.set_ylabel("Poor-fit reviews found"); ax.set_ylim(0, max(vals) * 1.3)
ax.set_title(f"Out of every 1,000 reviews, about {bad:.0f} are a poor fit.\nThe model finds about {top - vals[0]:.0f} more than chance", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/2_what_flagging_finds.png", dpi=160); plt.close()
print("per 1000: random", round(vals[0], 1), "model", round(top, 1), "bad", round(bad, 1))

# Chart 3: product type
r2 = pd.concat([tr, te]); c = r2.groupby("secondary_category")["mismatch_any"].agg(["mean", "size"]); c = c[c["size"] >= 500].sort_values("mean")
avg = r2.mismatch_any.mean() * 100
fig, ax = plt.subplots(figsize=(8.5, 5.5))
cols = [ORANGE if v * 100 >= 22 else GREY for v in c["mean"]]
ax.barh(c.index, c["mean"] * 100, color=cols)
for i, v in enumerate(c["mean"] * 100): ax.text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=11)
ax.axvline(avg, color="black", ls="--", lw=1); ax.text(avg + 0.3, -0.9, f"Average {avg:.0f}%", fontsize=11)
ax.set_xlabel("Reviews that were a poor fit (%)")
ax.set_title("Self tanners, minis and lip care get the most poor-fit\nreviews; treatments and cleansers the fewest", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/3_product_type.png", dpi=160); plt.close()

# Chart 4: poor-fit reviews over the years
r2["year"] = pd.to_datetime(r2["submission_time"]).dt.year
y = r2.groupby("year")["mismatch_any"].agg(["mean", "size"]); y = y[y["size"] >= 300]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(y.index, y["mean"] * 100, marker="o", color=BLUE, lw=2.5)
pk = y["mean"].idxmax(); ax.annotate(f"Peak {y['mean'].max()*100:.0f}% in {pk}", (pk, y["mean"].max()*100), xytext=(pk - 6, y["mean"].max()*100 - 1), fontsize=11, arrowprops=dict(arrowstyle="->"))
ax.set_ylim(0, 30); ax.set_ylabel("Reviews that were a poor fit (%)"); ax.set_xlabel("Year of review")
ax.set_title("The share of poor-fit reviews rose until 2020, then fell", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/4_over_time.png", dpi=160); plt.close()
print(band.round(1).to_dict(), y.round(3).to_dict()["mean"])