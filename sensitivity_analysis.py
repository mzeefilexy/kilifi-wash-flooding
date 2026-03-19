
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.feature_selection import mutual_info_regression
from statsmodels.othermod.betareg import BetaModel
import patsy

# Load data
df = pd.read_csv("kilifi_CHU_data_18112025.csv")
df["CHU"] = df["CHU"].str.strip().str.replace("\n", "", regex=False)
df = df.rename(columns={"CHU": "chu_name", "round": "Round", "flood_pct": "flood_freq_pct"})
df["Setting"] = df["Setting"].astype("category")
df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
req_base = ["chu_name", "Setting", "Round", "safe_water_pct", "latrine_access_pct", "flood_freq_pct", "elevation_m"]
df = df.dropna(subset=req_base).copy()

# Candidate covariates
land_cover_cols = ["Water", "Rangeland", "Trees", "Crops", "Bare Ground"]
candidate_covs = [col for col in land_cover_cols if col in df.columns and df[col].var() > 1e-6]

def zscore(s):
    mu, sd = s.mean(), s.std(ddof=0)
    if np.isclose(sd, 0):
        return pd.Series(np.zeros_like(s), index=s.index), float(mu), 1.0
    return (s - mu) / sd, float(mu), float(sd)

df["flood_z"], _, _ = zscore(df["flood_freq_pct"].astype(float))
df["elev_z"], _, _ = zscore(df["elevation_m"].astype(float))
df["round_z"], _, _ = zscore(df["Round"].astype(float))

scaled_covs = {}
for col in candidate_covs:
    scaled_name = col.lower().replace(" ", "_") + "_z"
    df[scaled_name], _, _ = zscore(df[col].astype(float))
    scaled_covs[col] = scaled_name

setting_map = {"Peri-Urban": -0.5, "Rural": 0.5}
df["setting_ec"] = df["Setting"].map(setting_map).astype(float)

# Long format
rows = []
for _, r in df.iterrows():
    for is_lat in [0, 1]:
        rows.append({
            "chu_name": r["chu_name"],
            "outcome": r["safe_water_pct"] if is_lat == 0 else r["latrine_access_pct"],
            "is_latrine_ec": -0.5 if is_lat == 0 else 0.5,
            "flood_z": r["flood_z"],
            "elev_z": r["elev_z"],
            "round_z": r["round_z"],
            "setting_ec": r["setting_ec"],
            **{scaled: r[scaled] for scaled in scaled_covs.values()}
        })

df_long = pd.DataFrame(rows).dropna(subset=["outcome"])

# Mutual information screening exactly as in main model
all_covs = ["flood_z", "elev_z", "round_z"] + list(scaled_covs.values())
mi_scores = {}
for cov in all_covs:
    mask = np.isfinite(df_long[cov]) & np.isfinite(df_long["outcome"])
    if mask.sum() >= 10:
        mi = mutual_info_regression(df_long.loc[mask, [cov]], df_long.loc[mask, "outcome"], random_state=42)
        mi_scores[cov] = mi[0]
    else:
        mi_scores[cov] = 0

selected_covs = [cov for cov, mi in mi_scores.items() if mi > 0.01]
if len(selected_covs) == 0:
    selected_covs = sorted(mi_scores, key=mi_scores.get, reverse=True)[:5]

terms = ["flood_z * is_latrine_ec", "is_latrine_ec * setting_ec"]
for cov in selected_covs:
    if cov != "flood_z":
        terms.append(cov)

main_formula = "outcome ~ " + " + ".join(terms)
round_formula = main_formula + " + round_z"

# Main mixed model on raw percentages
m_main = smf.mixedlm(main_formula, df_long, groups=df_long["chu_name"]).fit(reml=True)
m_main_round = smf.mixedlm(round_formula, df_long, groups=df_long["chu_name"]).fit(reml=True)

# Logit transformed mixed model
eps = 1e-4
df_long["outcome_prop"] = (df_long["outcome"] / 100).clip(eps, 1 - eps)
df_long["outcome_logit"] = np.log(df_long["outcome_prop"] / (1 - df_long["outcome_prop"]))

m_logit = smf.mixedlm(main_formula.replace("outcome", "outcome_logit"), df_long, groups=df_long["chu_name"]).fit(reml=True)
m_logit_round = smf.mixedlm(round_formula.replace("outcome", "outcome_logit"), df_long, groups=df_long["chu_name"]).fit(reml=True)

# Beta regression with CHU clustered robust standard errors
df_long["y"] = df_long["outcome_prop"]
y1, X1 = patsy.dmatrices(main_formula.replace("outcome", "y"), df_long, return_type="dataframe")
y2, X2 = patsy.dmatrices(round_formula.replace("outcome", "y"), df_long, return_type="dataframe")

m_beta = BetaModel(y1, X1).fit(disp=False, cov_type="cluster", cov_kwds={"groups": df_long["chu_name"]})
m_beta_round = BetaModel(y2, X2).fit(disp=False, cov_type="cluster", cov_kwds={"groups": df_long["chu_name"]})

# Summaries for the key interaction term
rows = []
for name, res in [
    ("Main model", m_main),
    ("Raw % + forced round", m_main_round),
    ("Logit mixed model", m_logit),
    ("Logit mixed + forced round", m_logit_round),
    ("Cluster-robust beta", m_beta),
    ("Cluster-robust beta + forced round", m_beta_round),
]:
    ci = res.conf_int().loc["flood_z:is_latrine_ec"].tolist()
    rows.append({
        "model": name,
        "estimate": float(res.params["flood_z:is_latrine_ec"]),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": float(res.pvalues["flood_z:is_latrine_ec"])
    })

summary = pd.DataFrame(rows)
summary.to_csv("sensitivity_interaction_summary.csv", index=False)
print(summary.round(4))
