"""A/B test analysis: paywall-offer experiment (SYNTHETIC data).

Covers: power / sample size, sample-ratio-mismatch (SRM) check, conversion lift with a confidence
interval, variance reduction with CUPED, and segment analysis with multiple-comparison control.
Run: python analysis.py -> results/*.png, results/summary.csv, README.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

OUT = Path(__file__).parent
RES = OUT / "results"
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# ------------------------------------------------------------ design: how many users do we need?
baseline, mde_abs = 0.060, 0.006          # 6.0% baseline conversion, detect +0.6pp (10% relative)
es = proportion_effectsize(baseline + mde_abs, baseline)
n_per_arm = int(np.ceil(NormalIndPower().solve_power(es, alpha=0.05, power=0.8, ratio=1.0)))

# ------------------------------------------------------------ synthetic experiment
N = 2 * 21_000
df = pd.DataFrame({
    "user": np.arange(N),
    "arm": rng.choice(["control", "variant"], N),          # intended 50/50 split
    "device": rng.choice(["mobile", "desktop", "tablet"], N, p=[0.62, 0.33, 0.05]),
    "new_user": rng.random(N) < 0.45,
})
df["pre_engagement"] = rng.gamma(2.0, 2.0, N)               # pre-experiment covariate (sessions)
logit_base = np.log(baseline / (1 - baseline)) + 0.10 * (df.pre_engagement - df.pre_engagement.mean())
true_lift_logit = np.where(df.arm == "variant", 0.07, 0.0)
true_lift_logit = true_lift_logit + np.where((df.arm == "variant") & (df.device == "mobile"), 0.03, 0.0)
p = 1 / (1 + np.exp(-(logit_base + true_lift_logit)))
df["converted"] = (rng.random(N) < p).astype(int)
df["revenue"] = (0.85 * df.pre_engagement + rng.normal(0, 1.5, N) + 0.25 * (df.arm == "variant"))  # sessions per user in the test window (secondary metric)

# ------------------------------------------------------------ 1. SRM check
counts = df.arm.value_counts()
chi2, srm_p = stats.chisquare([counts["control"], counts["variant"]])

# ------------------------------------------------------------ 2. primary metric: conversion
g = df.groupby("arm").converted.agg(["sum", "count", "mean"])
c, v = g.loc["control"], g.loc["variant"]
z, pval = proportions_ztest([v["sum"], c["sum"]], [v["count"], c["count"]])
diff = v["mean"] - c["mean"]
se = np.sqrt(v["mean"] * (1 - v["mean"]) / v["count"] + c["mean"] * (1 - c["mean"]) / c["count"])
ci = (diff - 1.96 * se, diff + 1.96 * se)
rel = diff / c["mean"]

# ------------------------------------------------------------ 3. CUPED on sessions per user
def cuped(y, x):
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())
df["rev_cuped"] = cuped(df.revenue.to_numpy(), df.pre_engagement.to_numpy())
def welch(col):
    a = df.loc[df.arm == "variant", col]; b = df.loc[df.arm == "control", col]
    t = stats.ttest_ind(a, b, equal_var=False)
    return a.mean() - b.mean(), (a.var() / len(a) + b.var() / len(b)) ** 0.5, t.pvalue
d_raw, se_raw, p_raw = welch("revenue")
d_cu, se_cu, p_cu = welch("rev_cuped")
var_reduction = 1 - (se_cu / se_raw) ** 2

# ------------------------------------------------------------ 4. segments with multiple-testing control
seg_rows = []
for col in ["device", "new_user"]:
    for val, s in df.groupby(col):
        gg = s.groupby("arm").converted.agg(["sum", "count"])
        zz, pp = proportions_ztest([gg.loc["variant", "sum"], gg.loc["control", "sum"]],
                                   [gg.loc["variant", "count"], gg.loc["control", "count"]])
        lift = gg.loc["variant", "sum"] / gg.loc["variant", "count"] - gg.loc["control", "sum"] / gg.loc["control", "count"]
        seg_rows.append([f"{col}={val}", int(gg["count"].sum()), lift, pp])
seg = pd.DataFrame(seg_rows, columns=["segment", "users", "abs_lift", "p_raw"])
seg["p_holm"] = multipletests(seg.p_raw, method="holm")[1]
seg.to_csv(RES / "segments.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.barh(seg.segment, seg.abs_lift * 100, color=["#2b6cb0" if p < 0.05 else "#a0aec0" for p in seg.p_holm])
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Absolute conversion lift (pp)")
ax.set_title("Lift by segment (blue = significant after Holm correction)")
plt.tight_layout(); plt.savefig(RES / "segment_lift.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.errorbar([0], [diff * 100], yerr=[[diff * 100 - ci[0] * 100], [ci[1] * 100 - diff * 100]], fmt="o", capsize=6)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks([]); ax.set_ylabel("Conversion lift (pp), 95% CI")
ax.set_title("Primary metric: conversion")
plt.tight_layout(); plt.savefig(RES / "primary_lift.png", dpi=140); plt.close()

pd.DataFrame({"metric": ["n_per_arm_required", "srm_p", "conv_control", "conv_variant", "abs_lift_pp", "p_value",
                         "rev_p_raw", "rev_p_cuped", "variance_reduction"],
              "value": [n_per_arm, srm_p, c["mean"], v["mean"], diff * 100, pval, p_raw, p_cu, var_reduction]}
             ).to_csv(RES / "summary.csv", index=False)

sig = seg[seg.p_holm < 0.05].segment.tolist()
decision = "ship the variant" if (pval < 0.05 and ci[0] > 0) else "do not ship on this evidence"
readme = f"""# A/B test analysis: paywall-offer experiment

**Synthetic data, real method.** Users, assignment and outcomes are simulated with a fixed seed in
`analysis.py`; no employer data is used. The point is the analysis workflow.

## Experiment
Does a new paywall offer (variant) raise conversion to paid subscription versus the current offer (control)?
Primary metric: conversion rate. Secondary: sessions per user in the test window. Guardrail: sample-ratio check.

## 1. Design (power and sample size)
Baseline conversion {baseline:.1%}; minimum detectable effect +{mde_abs * 100:.1f}pp ({mde_abs / baseline:.0%} relative);
alpha 0.05, power 80%. Required users per arm: **{n_per_arm:,}**. The test ran with {len(df):,} users
({counts['control']:,} control / {counts['variant']:,} variant).

## 2. Sample-ratio mismatch check
Expected a 50/50 split. Chi-square p = {srm_p:.3f}. {'No sign of an assignment problem.' if srm_p > 0.01 else 'WARNING: possible assignment bug; investigate before trusting results.'}

## 3. Primary result
Control {c['mean']:.2%} vs variant {v['mean']:.2%}: absolute lift **{diff * 100:+.2f}pp** ({rel:+.1%} relative),
95% CI [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]pp, p = {pval:.4f}. Decision: **{decision}**.

## 4. Variance reduction with CUPED (sessions per user)
Using pre-experiment engagement as a covariate cut the standard error of the sessions estimate
(variance reduction {var_reduction:.0%}). p-value moved from {p_raw:.3f} (raw) to {p_cu:.3f} (CUPED).
Smaller variance means the same test can detect smaller effects or run shorter.

## 5. Segments (with correction)
Segment cuts inflate false positives, so p-values are Holm-adjusted. Segments significant after
correction: {', '.join(sig) if sig else 'none'}. Segment results are for generating hypotheses, not for
shipping decisions on their own.

![primary](results/primary_lift.png)
![segments](results/segment_lift.png)

## Caveats I would raise with a real experiment
- Pre-register the metric and stopping rule; avoid peeking at p-values.
- Check novelty effects and run for at least one full weekly cycle.
- Look at guardrails (refunds, churn at 30/60 days), not just conversion.

## Run it
`pip install pandas numpy scipy statsmodels matplotlib` then `python analysis.py`.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(f"n/arm={n_per_arm} srm_p={srm_p:.3f} control={c['mean']:.4f} variant={v['mean']:.4f} lift={diff*100:+.2f}pp p={pval:.4f}")
print(f"CI=({ci[0]*100:+.2f},{ci[1]*100:+.2f}) rev p raw={p_raw:.3f} cuped={p_cu:.3f} var_red={var_reduction:.2f}")
print(seg.round(4).to_string(index=False))

