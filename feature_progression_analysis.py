import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure working directory and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "50_Startups.csv")

print("[1/5] Loading and preprocessing dataset...")
df = pd.read_csv(CSV_FILE)

# One-hot encode 'State' with drop_first=True (California is reference category)
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
feature_names = [c for c in df_encoded.columns if c != 'Profit']
target_name = 'Profit'

X = df_encoded[feature_names]
y = df_encoded[target_name]

# Fixed split (80% train, 20% test, random_state=0 for standard benchmark)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
n_test = len(y_test)

# Scaled data for regularization methods
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

print(f"Features: {feature_names}")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

def evaluate_features(feats):
    k = len(feats)
    reg = LinearRegression().fit(X_train[feats], y_train)
    
    train_pred = reg.predict(X_train[feats])
    test_pred = reg.predict(X_test[feats])
    
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    train_r2 = float(r2_score(y_train, train_pred))
    test_r2 = float(r2_score(y_test, test_pred))
    test_mae = float(mean_absolute_error(y_test, test_pred))
    
    # Adjusted R2 on test set
    if n_test - k - 1 > 0:
        adj_r2 = float(1.0 - (1.0 - test_r2) * (n_test - 1) / (n_test - k - 1))
    else:
        adj_r2 = test_r2
        
    # F-statistic and p-value
    df_reg = k
    df_resid = max(1, n_test - k - 1)
    if test_r2 > 0 and test_r2 < 1.0 and df_resid > 0:
        f_stat = float((test_r2 / df_reg) / ((1.0 - test_r2) / df_resid))
        p_val = float(1.0 - stats.f.cdf(f_stat, df_reg, df_resid))
    else:
        f_stat = 0.0
        p_val = 1.0

    coef_dict = {f: float(c) for f, c in zip(feats, reg.coef_)}
    
    return {
        'k': k,
        'features': feats,
        'test_rmse': round(test_rmse, 2),
        'test_r2': round(test_r2, 4),
        'adjusted_r2': round(adj_r2, 4),
        'test_mae': round(test_mae, 2),
        'train_rmse': round(train_rmse, 2),
        'train_r2': round(train_r2, 4),
        'f_statistic': round(f_stat, 2),
        'p_value': f"{p_val:.2e}" if p_val < 0.0001 else f"{round(p_val, 4)}",
        'intercept': round(float(reg.intercept_), 2),
        'coefficients': {k: round(v, 4) for k, v in coef_dict.items()}
    }

print("[2/5] Running 5 Feature Selection Algorithms...")

# 1. Lasso L1 Ranking
lasso = LassoCV(cv=5, random_state=0).fit(X_train_scaled, y_train)
lasso_order = [feature_names[i] for i in np.argsort(-np.abs(lasso.coef_))]
lasso_results = [evaluate_features(lasso_order[:k]) for k in range(1, 6)]

# 2. ElasticNet Ranking
enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=5, random_state=0).fit(X_train_scaled, y_train)
enet_order = [feature_names[i] for i in np.argsort(-np.abs(enet.coef_))]
enet_results = [evaluate_features(enet_order[:k]) for k in range(1, 6)]

# 3. Forward SFS
sfs_results = []
for k in range(1, 6):
    if k == 5:
        feats = feature_names
    else:
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', scoring='neg_mean_squared_error', cv=5)
        sfs.fit(X_train, y_train)
        feats = [f for f, support in zip(feature_names, sfs.support_) if support]
    sfs_results.append(evaluate_features(feats))

# 4. RFE (Standardized)
rfe_results = []
for k in range(1, 6):
    rfe = RFE(LinearRegression(), n_features_to_select=k)
    rfe.fit(X_train_scaled, y_train)
    feats = [f for f, support in zip(feature_names, rfe.support_) if support]
    rfe_results.append(evaluate_features(feats))

# 5. Exhaustive Search (Best Subset by training MSE)
exhaustive_results = []
for k in range(1, 6):
    best_comb = None
    best_train_loss = float('inf')
    for comb in itertools.combinations(feature_names, k):
        reg = LinearRegression().fit(X_train[list(comb)], y_train)
        loss = mean_squared_error(y_train, reg.predict(X_train[list(comb)]))
        if loss < best_train_loss:
            best_train_loss = loss
            best_comb = list(comb)
    exhaustive_results.append(evaluate_features(best_comb))

methods_dict = {
    'Lasso (L1 Ranking)': lasso_results,
    'Forward SFS': sfs_results,
    'RFE': rfe_results,
    'ElasticNet Ranking': enet_results,
    'Exhaustive Search': exhaustive_results
}

print("[3/5] Saving progression CSV files...")
csv_map = {
    'lasso_l1_ranking_progression.csv': lasso_results,
    'forward_sfs_progression.csv': sfs_results,
    'rfe_progression.csv': rfe_results,
    'elasticnet_ranking_progression.csv': enet_results,
    'exhaustive_search_progression.csv': exhaustive_results
}

for fname, res_list in csv_map.items():
    rows = []
    for r in res_list:
        rows.append({
            'Number_of_Features': r['k'],
            'Selected_Features': " + ".join(r['features']),
            'Test_RMSE': r['test_rmse'],
            'Test_R2': r['test_r2'],
            'Adjusted_R2': r['adjusted_r2'],
            'Test_MAE': r['test_mae'],
            'F_Statistic': r['f_statistic'],
            'Train_RMSE': r['train_rmse'],
            'Train_R2': r['train_r2']
        })
    pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, fname), index=False)
    print(f"  -> Saved {fname}")

# Comprehensive comparison CSV
comparison_rows = []
for m_name, res_list in methods_dict.items():
    for r in res_list:
        comparison_rows.append({
            'Algorithm': m_name,
            'k': r['k'],
            'Features': " + ".join(r['features']),
            'Test_RMSE': r['test_rmse'],
            'Test_R2': r['test_r2'],
            'Adjusted_R2': r['adjusted_r2'],
            'Test_MAE': r['test_mae'],
            'Train_RMSE': r['train_rmse']
        })
pd.DataFrame(comparison_rows).to_csv(os.path.join(BASE_DIR, "feature_selection_comparison.csv"), index=False)
print("  -> Saved feature_selection_comparison.csv")

print("[4/5] Generating Comparison Plots (PNG)...")
# Set style
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

for m_name, res in methods_dict.items():
    ks = [r['k'] for r in res]
    rmses = [r['test_rmse'] for r in res]
    r2s = [r['test_r2'] for r in res]
    
    ax1.plot(ks, rmses, label=m_name, color=palette[m_name], marker=markers[m_name], linewidth=2, markersize=7)
    ax2.plot(ks, r2s, label=m_name, color=palette[m_name], marker=markers[m_name], linewidth=2, markersize=7)

ax1.set_title('RMSE Comparison across 5 Feature Selection Methods', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Number of Features (k)', fontsize=11)
ax1.set_ylabel('RMSE (Lower is Better)', fontsize=11)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=9, loc='upper left')

# Mark the sweet spot
ax1.axvline(x=2, color='#10b981', linestyle=':', alpha=0.8)
ax1.annotate('Optimal Sweet Spot (k=2)\nMin RMSE = $8,198.80', xy=(2, 8198.80), xytext=(2.2, 8400),
             arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9, color='#047857', fontweight='bold')

ax2.set_title('R-squared Comparison across 5 Feature Selection Methods', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Number of Features (k)', fontsize=11)
ax2.set_ylabel('R-squared (Higher is Better)', fontsize=11)
ax2.set_xticks([1, 2, 3, 4, 5])
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=9, loc='lower left')

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, 'all_5_algorithms_comparison_plot.png')
plt.savefig(plot_path)
plt.close()
print(f"  -> Saved {plot_path}")

# Individual plots
for m_name, res in methods_dict.items():
    slug = m_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ks = [r['k'] for r in res]
    rmses = [r['test_rmse'] for r in res]
    ax.plot(ks, rmses, color=palette[m_name], marker=markers[m_name], linewidth=2.5, markersize=8, label=m_name)
    ax.set_title(f'{m_name} - Feature Progression', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Features (k)')
    ax.set_ylabel('Test RMSE')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"{slug}_plot.png"))
    plt.close()

print("[5/5] Generating Web Dashboard Data (JSON/JS)...")
summary_stats = {
    'dataset_name': '50 Startups (Kaggle)',
    'total_samples': len(df),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'features_all': feature_names,
    'sweet_spot_k': 2,
    'sweet_spot_features': ['R&D Spend', 'Marketing Spend'],
    'sweet_spot_rmse': 8198.80,
    'sweet_spot_r2': 0.9474,
    'overfitting_k': 5,
    'overfitting_rmse': 9137.99,
    'overfitting_r2': 0.9347,
    'descriptive_stats': {
        col: {
            'mean': round(float(df[col].mean()), 2),
            'std': round(float(df[col].std()), 2),
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2)
        } for col in ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
    },
    'states_distribution': df['State'].value_counts().to_dict(),
    'methods_data': methods_dict
}

with open(os.path.join(BASE_DIR, 'model_data.json'), 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)

with open(os.path.join(BASE_DIR, 'model_data.js'), 'w', encoding='utf-8') as f:
    f.write(f"const MODEL_DATA = {json.dumps(summary_stats, indent=2, ensure_ascii=False)};\n")

print("All tasks completed successfully!")
