from scipy.stats import spearmanr, kruskal
import numpy as np
import pandas as pd

scored = pd.read_csv("data/processed/svf_scored_results.csv")

# Redundancy check: how correlated are cluster metrics with total_words?
for col in ['cluster_count', 'switch_count', 'mean_cluster_size', 'max_cluster_size']:
    r, p = spearmanr(scored['total_words'], scored[col])
    print(f"  Spearman(total_words, {col}): r={r:.3f}, p={p:.4f}")

# K-W for the new metrics
diag_order = ['HC', 'MCI', 'non-AD', 'AD']
for col in ['mean_cluster_size', 'cluster_count', 'switch_count', 'similarity_slope']:
    groups = [scored[scored['diagnosis']==d][col].dropna().values for d in diag_order]
    H, p = kruskal(*groups)
    print(f"  {col}: H={H:.3f}, p={p:.4f}")

# The key test: does mean_cluster_size predict group AFTER controlling
# for total_words? Quick check via partial correlation or just residuals.
from sklearn.linear_model import LinearRegression
X = scored[['total_words']].values
y = scored['mean_cluster_size'].values
mask = ~np.isnan(y)
residuals = y[mask] - LinearRegression().fit(X[mask], y[mask]).predict(X[mask])
scored_valid = scored[mask].copy()
scored_valid['mcs_residual'] = residuals
groups_resid = [scored_valid[scored_valid['diagnosis']==d]['mcs_residual'].values for d in diag_order]
H_resid, p_resid = kruskal(*groups_resid)
print(f"\n  mean_cluster_size residuals (controlling for total_words): H={H_resid:.3f}, p={p_resid:.4f}")