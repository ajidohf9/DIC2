import os
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'model_data.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

boston = data['boston_housing']['methods_data']

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

palette = {
    'Lasso (L1 Ranking)': '#1f77b4',
    'Forward SFS': '#ff7f0e',
    'RFE': '#2ca02c',
    'ElasticNet Ranking': '#d62728',
    'Exhaustive Search': '#9467bd'
}

markers = {
    'Lasso (L1 Ranking)': 'o',
    'Forward SFS': 's',
    'RFE': '^',
    'ElasticNet Ranking': 'd',
    'Exhaustive Search': 'v'
}

for m_name, res in boston.items():
    ks = [r['k'] for r in res]
    rmses = [r['test_rmse'] for r in res]
    r2s = [r['test_r2'] for r in res]
    ax1.plot(ks, rmses, label=m_name, color=palette[m_name], marker=markers[m_name], linewidth=2, markersize=6)
    ax2.plot(ks, r2s, label=m_name, color=palette[m_name], marker=markers[m_name], linewidth=2, markersize=6)

ax1.set_title('Boston Housing: RMSE Comparison across 5 Feature Selection Methods', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Number of Features (k)', fontsize=11)
ax1.set_ylabel('Test RMSE ($1k) - Lower is Better', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=9, loc='upper right')

ax2.set_title('Boston Housing: R-squared Comparison across 5 Feature Selection Methods', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Number of Features (k)', fontsize=11)
ax2.set_ylabel('Test R-squared - Higher is Better', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=9, loc='lower right')

plt.tight_layout()
out_path = os.path.join(BASE_DIR, 'boston_housing_all_5_algorithms_comparison_plot.png')
plt.savefig(out_path)
plt.close()
print(f"Saved {out_path}")
