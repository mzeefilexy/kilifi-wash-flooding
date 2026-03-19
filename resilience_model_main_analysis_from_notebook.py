# %% Cell 1
# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_selection import mutual_info_regression

# -----------------------------
# Load & clean data
# -----------------------------
df = pd.read_csv('/content/kilifi_CHU_data_18112025.csv')
df['CHU'] = df['CHU'].str.strip().str.replace('\n', '', regex=False)

df = df.rename(columns={
    'CHU': 'chu_name',
    'round': 'Round',
    'flood_pct': 'flood_freq_pct'
})

df['Setting'] = df['Setting'].astype('category')
df['Round'] = pd.to_numeric(df['Round'], errors='coerce')
req_base = ['chu_name', 'Setting', 'Round', 'safe_water_pct', 'latrine_access_pct',
            'flood_freq_pct', 'elevation_m']
df = df.dropna(subset=req_base).copy()

# -----------------------------
# Candidate covariates: LAND COVER ONLY (no rwi)
# -----------------------------
land_cover_cols = ['Water', 'Rangeland', 'Trees', 'Crops', 'Bare Ground']
candidate_covs = [col for col in land_cover_cols if col in df.columns and df[col].var() > 1e-6]

# -----------------------------
# Center/scale
# -----------------------------
def zscore(s):
    mu, sd = s.mean(), s.std(ddof=0)
    if np.isclose(sd, 0):
        return pd.Series(np.zeros_like(s), index=s.index), float(mu), 1.0
    return (s - mu) / sd, float(mu), float(sd)

df['flood_z'], flood_mu, flood_sd = zscore(df['flood_freq_pct'].astype(float))
df['elev_z'],  elev_mu,  elev_sd  = zscore(df['elevation_m'].astype(float))
df['round_z'], round_mu, round_sd = zscore(df['Round'].astype(float))

scaled_covs = {}
for col in candidate_covs:
    scaled_name = col.lower().replace(' ', '_') + '_z'
    df[scaled_name], _, _ = zscore(df[col].astype(float))
    scaled_covs[col] = scaled_name

setting_map = {'Peri-Urban': -0.5, 'Rural': 0.5}
df['setting_ec'] = df['Setting'].map(setting_map).astype(float)

# -----------------------------
# Long format
# -----------------------------
rows = []
for _, r in df.iterrows():
    for is_lat in [0, 1]:
        is_lat_ec = -0.5 if is_lat == 0 else 0.5
        outcome = r['safe_water_pct'] if is_lat == 0 else r['latrine_access_pct']
        row_dict = {
            'chu_name': r['chu_name'],
            'outcome': outcome,
            'is_latrine_ec': is_lat_ec,
            'flood_z': r['flood_z'],
            'elev_z': r['elev_z'],
            'round_z': r['round_z'],
            'setting_ec': r['setting_ec'],
            'Setting': r['Setting'],
            'flood_freq_pct': r['flood_freq_pct'],
            'Round': r['Round'],
            'elevation_m': r['elevation_m']
        }
        for scaled in scaled_covs.values():
            row_dict[scaled] = r[scaled]
        rows.append(row_dict)

df_long = pd.DataFrame(rows)
df_long = df_long.dropna(subset=['outcome'])

# -----------------------------
# Mutual info screening
# -----------------------------
all_covs = ['flood_z', 'elev_z', 'round_z'] + list(scaled_covs.values())
mi_scores = {}
for cov in all_covs:
    mask = np.isfinite(df_long[cov]) & np.isfinite(df_long['outcome'])
    if mask.sum() >= 10:
        mi = mutual_info_regression(
            df_long.loc[mask, [cov]],
            df_long.loc[mask, 'outcome'],
            random_state=42
        )
        mi_scores[cov] = mi[0]
    else:
        mi_scores[cov] = 0

selected_covs = [cov for cov, mi in mi_scores.items() if mi > 0.01]
if len(selected_covs) == 0:
    selected_covs = sorted(mi_scores, key=mi_scores.get, reverse=True)[:5]

# -----------------------------
# Build formula
# -----------------------------
terms = ["flood_z * is_latrine_ec", "is_latrine_ec * setting_ec"]
for cov in selected_covs:
    if cov not in ['flood_z']:
        terms.append(cov)

fe_formula = "outcome ~ " + " + ".join(terms)

# -----------------------------
# FIT CONVERGED MODEL: RANDOM INTERCEPT ONLY
# -----------------------------
m_final = smf.mixedlm(
    fe_formula,
    df_long,
    groups=df_long['chu_name']
).fit(reml=True)

print("✅ MODEL CONVERGED:", m_final.converged)
print("\nFINAL MODEL SUMMARY")
print(m_final.summary())

# -----------------------------
# Resilience scoring (unchanged logic)
# -----------------------------
def slope_per_plus1_flood(chu_row, is_lat):
    is_lat_ec = -0.5 if is_lat == 0 else 0.5
    def make_row(F_original):
        data = {
            'flood_z': (F_original - flood_mu) / (flood_sd if not np.isclose(flood_sd, 0) else 1.0),
            'is_latrine_ec': is_lat_ec,
            'setting_ec': setting_map[str(chu_row['Setting'])],
            'elev_z': (float(chu_row['elevation_m']) - elev_mu) / (elev_sd if not np.isclose(elev_sd, 0) else 1.0),
            'round_z': (float(chu_row['Round']) - round_mu) / (round_sd if not np.isclose(round_sd, 0) else 1.0),
            'chu_name': chu_row['chu_name']
        }
        for scaled in scaled_covs.values():
            data[scaled] = 0.0
        return pd.DataFrame([data])

    F = float(chu_row['flood_freq_pct'])
    pred_A = m_final.predict(make_row(F))
    pred_B = m_final.predict(make_row(F + 1.0))
    return float(pred_B.iloc[0] - pred_A.iloc[0])

cluster = df.groupby('chu_name', as_index=False).agg({
    'safe_water_pct': 'mean',
    'latrine_access_pct': 'mean',
    'flood_freq_pct': 'mean',
    'Setting': 'first',
    'elevation_m': 'mean',
    'Round': 'mean'
})

cluster['slope_water']   = cluster.apply(lambda r: slope_per_plus1_flood(r, is_lat=0), axis=1)
cluster['slope_latrine'] = cluster.apply(lambda r: slope_per_plus1_flood(r, is_lat=1), axis=1)

cluster['sw_resil']  = cluster['safe_water_pct']     - cluster['slope_water']   * cluster['flood_freq_pct']
cluster['lat_resil'] = cluster['latrine_access_pct'] - cluster['slope_latrine'] * cluster['flood_freq_pct']

def normalize_symm(x):
    x = x.astype(float)
    xmin, xmax = x.min(), x.max()
    if np.isclose(xmax, xmin):
        return pd.Series(np.zeros_like(x), index=x.index)
    return 2 * (x - xmin) / (xmax - xmin) - 1

cluster['Safe_Water_Resilience'] = normalize_symm(cluster['sw_resil'])
cluster['Latrine_Resilience']    = normalize_symm(cluster['lat_resil'])
cluster['Overall_Resilience']    = 0.5 * (cluster['Safe_Water_Resilience'] + cluster['Latrine_Resilience'])

ranking = cluster[['chu_name', 'Setting', 'Safe_Water_Resilience',
                   'Latrine_Resilience', 'Overall_Resilience']].sort_values(
                        'Overall_Resilience', ascending=False).round(3)

print("\n🏆 FINAL RESILIENCE RANKING (Converged Model)")
print(ranking.to_string(index=False))

# %% Cell 2
print("\nMutual Information Scores:")
for cov, score in mi_scores.items():
    print(f"{cov}: {score:.4f}")

print(f"\nSelected Covariates (MI > 0.01 or top 5 if none pass): {selected_covs}")

if 'round_z' not in selected_covs:
    print(f"\n'round_z' was not included in the final model because its mutual information score ({mi_scores.get('round_z', 0):.4f}) was below the threshold of 0.01, and it was not among the top 5 covariates selected if no others passed the threshold.")

# %% Cell 3
import statsmodels.formula.api as smf

# Fit an OLS model using the same formula as the mixed-effects model
m_ols = smf.ols(fe_formula, data=df_long).fit()

print("\nORDINARY LEAST SQUARES MODEL FIT")
print(f"Log-Likelihood: {m_ols.llf:.3f}")
print(f"AIC: {m_ols.aic:.3f}")
print(f"BIC: {m_ols.bic:.3f}")

# %% Cell 4
import matplotlib.pyplot as plt

# Calculate and print AIC, BIC, and ICC
print("\nMODEL FIT AND DIAGNOSTICS")
print(f"AIC: {m_final.aic:.3f}")
print(f"BIC: {m_final.bic:.3f}")
# ICC calculation for MixedLM is not directly available as a single value like in some software,
# but the 'Group Var' from the summary can be interpreted as the variance of the random intercepts.
# A common way to approximate ICC in simple random intercept models is:
# ICC = Group Variance / (Group Variance + Residual Variance)
group_var = m_final.cov_re.iloc[0, 0]
residual_var = m_final.scale
icc_approx = group_var / (group_var + residual_var)
print(f"Approximate ICC: {icc_approx:.3f}")

# Generate a plot of residuals
plt.figure(figsize=(10, 6))
plt.scatter(m_final.fittedvalues, m_final.resid, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted values")
plt.grid(True)
plt.show()

# %% Cell 5
import scipy.stats as stats
import matplotlib.pyplot as plt

# Q-Q plot for normality of residuals
plt.figure(figsize=(8, 6))
stats.probplot(m_final.resid, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot of Residuals")
plt.grid(True)
plt.show()

# %% Cell 6
import matplotlib.pyplot as plt
import numpy as np

# Scale-Location plot (sqrt of absolute residuals vs fitted values) for homoscedasticity
plt.figure(figsize=(10, 6))
plt.scatter(m_final.fittedvalues, np.sqrt(np.abs(m_final.resid)), alpha=0.5)
plt.xlabel("Fitted values")
plt.ylabel(r"$\sqrt{|Residuals|}$") # Use a raw string for the LaTeX expression
plt.title("Scale-Location Plot")
plt.grid(True)
plt.show()

# %% Cell 7
# Save model summary as text
with open('model_summary.txt', 'w') as f:
    f.write(m_final.summary().as_text())

# Save resilience ranking
ranking.to_csv('resilience_ranking.csv', index=False)

# Save descriptive stats
desc_stats = df.groupby('Setting')[['safe_water_pct', 'latrine_access_pct', 'flood_freq_pct', 'elevation_m']].agg(['mean', 'min', 'max'])
desc_stats.to_csv('descriptives_by_setting.csv')

# Save coefficient table
coef_df = pd.DataFrame({
    'Variable': m_final.params.index,
    'Coefficient': m_final.params.values,
    'StdErr': m_final.bse.values,
    'P-value': m_final.pvalues.values,
    'CI_lower': m_final.conf_int()[0].values,
    'CI_upper': m_final.conf_int()[1].values
})
coef_df.to_csv('model_estimates.csv', index=False)

# %% Cell 8
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import io

# Provided resilience ranking data as a string (same as before)
ranking_data = """chu_name    Setting  Safe_Water_Resilience  Latrine_Resilience  Overall_Resilience
Mwele-Kisurutini Peri-Urban                  1.000               0.908               0.954
         Kombeni Peri-Urban                  0.789               0.720               0.755
            Buni Peri-Urban                  0.919               0.568               0.743
       Vishakani Peri-Urban                  0.423               0.906               0.665
           Kwale      Rural                  0.059               1.000               0.530
      Kibwabwani      Rural                 -0.941               0.604              -0.168
        Mutsengo      Rural                 -0.892               0.165              -0.363
        Viragoni      Rural                 -0.942              -0.335              -0.639
     Mnazimwenga      Rural                 -0.977              -0.708              -0.842
     Tsangatsini      Rural                 -1.000              -1.000              -1.000
"""

# Read the string data into a pandas DataFrame
ranking = pd.read_csv(io.StringIO(ranking_data), sep=r'\s+', engine='python', header=0)

# Load geographic data
gdf = gpd.read_file('/content/kilifi_chu_10_cleaned.geojson')

# Reconcile 'Mwele-Kisurutini' by removing trailing whitespace from the 'CHU' column in gdf
gdf['CHU'] = gdf['CHU'].str.strip()

# Merge resilience data with geographic data
merged_gdf = gdf.merge(ranking, left_on='CHU', right_on='chu_name')

# --- Map for Latrine Resilience ---
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged_gdf.plot(column='Latrine_Resilience', ax=ax, legend=True, cmap='coolwarm_r')
ax.set_title(" ")
ax.grid(True)

# Add CHU labels
merged_gdf.apply(lambda x: ax.annotate(text=x['chu_name'], xy=x.geometry.centroid.coords[0], ha='center', fontsize=8), axis=1)

# North arrow and scale bar (Latrine Map)
ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.88),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=2, headwidth=8),
            ha='center', va='center', fontsize=12, fontweight='bold')

xmin, xmax = ax.get_xlim()
scalebar = AnchoredSizeBar(
    ax.transData, (xmax-xmin)*0.2, '10 km',
    loc='lower right', pad=0.5, color='black', frameon=False
)
ax.add_artist(scalebar)

plt.show()

# %% Cell 9
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import io

# Provided resilience ranking data as a string (same as before)
ranking_data = """chu_name    Setting  Safe_Water_Resilience  Latrine_Resilience  Overall_Resilience
Mwele-Kisurutini Peri-Urban                  1.000               0.908               0.954
         Kombeni Peri-Urban                  0.789               0.720               0.755
            Buni Peri-Urban                  0.919               0.568               0.743
       Vishakani Peri-Urban                  0.423               0.906               0.665
           Kwale      Rural                  0.059               1.000               0.530
      Kibwabwani      Rural                 -0.941               0.604              -0.168
        Mutsengo      Rural                 -0.892               0.165              -0.363
        Viragoni      Rural                 -0.942              -0.335              -0.639
     Mnazimwenga      Rural                 -0.977              -0.708              -0.842
     Tsangatsini      Rural                 -1.000              -1.000              -1.000
"""

# Read the string data into a pandas DataFrame
ranking = pd.read_csv(io.StringIO(ranking_data), sep=r'\s+', engine='python', header=0)

# Load geographic data
gdf = gpd.read_file('/content/kilifi_chu_10_cleaned.geojson')

# Reconcile 'Mwele-Kisurutini' by removing trailing whitespace from the 'CHU' column in gdf
gdf['CHU'] = gdf['CHU'].str.strip()

# Merge resilience data with geographic data
merged_gdf = gdf.merge(ranking, left_on='CHU', right_on='chu_name')

# --- Map for Safe Water Resilience ---
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged_gdf.plot(column='Safe_Water_Resilience', ax=ax, legend=True, cmap='coolwarm_r')
ax.set_title(" ")
ax.grid(True)

# Add CHU labels
merged_gdf.apply(lambda x: ax.annotate(text=x['chu_name'], xy=x.geometry.centroid.coords[0], ha='center', fontsize=8), axis=1)

# North arrow and scale bar (Safe Water Map)
ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.88),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=2, headwidth=8),
            ha='center', va='center', fontsize=12, fontweight='bold')

xmin, xmax = ax.get_xlim()
scalebar = AnchoredSizeBar(
    ax.transData, (xmax-xmin)*0.2, '10 km',
    loc='lower right', pad=0.5, color='black', frameon=False
)
ax.add_artist(scalebar)

plt.show()

# %% Cell 10
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import io

# Provided resilience ranking data as a string (same as before)
ranking_data = """chu_name    Setting  Safe_Water_Resilience  Latrine_Resilience  Overall_Resilience
Mwele-Kisurutini Peri-Urban                  1.000               0.908               0.954
         Kombeni Peri-Urban                  0.789               0.720               0.755
            Buni Peri-Urban                  0.919               0.568               0.743
       Vishakani Peri-Urban                  0.423               0.906               0.665
           Kwale      Rural                  0.059               1.000               0.530
      Kibwabwani      Rural                 -0.941               0.604              -0.168
        Mutsengo      Rural                 -0.892               0.165              -0.363
        Viragoni      Rural                 -0.942              -0.335              -0.639
     Mnazimwenga      Rural                 -0.977              -0.708              -0.842
     Tsangatsini      Rural                 -1.000              -1.000              -1.000
"""

# Read the string data into a pandas DataFrame
ranking = pd.read_csv(io.StringIO(ranking_data), sep=r'\s+', engine='python', header=0)

# Load geographic data
gdf = gpd.read_file('/content/kilifi_chu_10_cleaned.geojson')

# Reconcile 'Mwele-Kisurutini' by removing trailing whitespace from the 'CHU' column in gdf
gdf['CHU'] = gdf['CHU'].str.strip()

# Merge resilience data with geographic data
merged_gdf = gdf.merge(ranking, left_on='CHU', right_on='chu_name')

# --- Map for Overall Resilience ---
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged_gdf.plot(column='Overall_Resilience', ax=ax, legend=True, cmap='coolwarm_r')
ax.set_title(" ")
ax.grid(True)

# Add CHU labels
merged_gdf.apply(lambda x: ax.annotate(text=x['chu_name'], xy=x.geometry.centroid.coords[0], ha='center', fontsize=8), axis=1)

# North arrow and scale bar (Overall Resilience Map)
ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.88),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=2, headwidth=8),
            ha='center', va='center', fontsize=12, fontweight='bold')

xmin, xmax = ax.get_xlim()
scalebar = AnchoredSizeBar(
    ax.transData, (xmax-xmin)*0.2, '10 km',
    loc='lower right', pad=0.5, color='black', frameon=False
)
ax.add_artist(scalebar)

plt.show()

