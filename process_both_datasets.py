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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_dataset(df, target_col, name, max_k=None, categorical_cols=None):
    print(f"\n==========================================")
    print(f"Processing Dataset: {name}")
    print(f"==========================================")
    
    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        df_encoded = df.copy()
        
    feature_names = [c for c in df_encoded.columns if c != target_col]
    X = df_encoded[feature_names]
    y = df_encoded[target_col]
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    n_test = len(y_test)
    total_features = len(feature_names)
    eval_max_k = min(total_features, max_k if max_k else total_features)
    
    # Scaled
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    
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
        
        if n_test - k - 1 > 0:
            adj_r2 = float(1.0 - (1.0 - test_r2) * (n_test - 1) / (n_test - k - 1))
        else:
            adj_r2 = test_r2
            
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
            'coefficients': {k_col: round(v, 4) for k_col, v in coef_dict.items()}
        }

    # 1. Lasso L1 Ranking
    lasso = LassoCV(cv=5, random_state=0).fit(X_train_scaled, y_train)
    lasso_order = [feature_names[i] for i in np.argsort(-np.abs(lasso.coef_))]
    lasso_results = [evaluate_features(lasso_order[:k]) for k in range(1, eval_max_k + 1)]

    # 2. ElasticNet Ranking
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=5, random_state=0).fit(X_train_scaled, y_train)
    enet_order = [feature_names[i] for i in np.argsort(-np.abs(enet.coef_))]
    enet_results = [evaluate_features(enet_order[:k]) for k in range(1, eval_max_k + 1)]

    # 3. Forward SFS
    sfs_results = []
    for k in range(1, eval_max_k + 1):
        if k == total_features:
            feats = feature_names
        else:
            sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', scoring='neg_mean_squared_error', cv=5)
            sfs.fit(X_train, y_train)
            feats = [f for f, support in zip(feature_names, sfs.support_) if support]
        sfs_results.append(evaluate_features(feats))

    # 4. RFE (Standardized)
    rfe_results = []
    for k in range(1, eval_max_k + 1):
        rfe = RFE(LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train_scaled, y_train)
        feats = [f for f, support in zip(feature_names, rfe.support_) if support]
        rfe_results.append(evaluate_features(feats))

    # 5. Exhaustive Search / Greedy Best Subset for higher dimensional
    exhaustive_results = []
    if total_features <= 5:
        for k in range(1, eval_max_k + 1):
            best_comb = None
            best_train_loss = float('inf')
            for comb in itertools.combinations(feature_names, k):
                reg = LinearRegression().fit(X_train[list(comb)], y_train)
                loss = mean_squared_error(y_train, reg.predict(X_train[list(comb)]))
                if loss < best_train_loss:
                    best_train_loss = loss
                    best_comb = list(comb)
            exhaustive_results.append(evaluate_features(best_comb))
    else:
        # For larger datasets like Boston Housing (13 features), use stepwise forward subset
        current_subset = []
        remaining = list(feature_names)
        for k in range(1, eval_max_k + 1):
            best_f = None
            best_loss = float('inf')
            for f in remaining:
                cand = current_subset + [f]
                reg = LinearRegression().fit(X_train[cand], y_train)
                loss = mean_squared_error(y_train, reg.predict(X_train[cand]))
                if loss < best_loss:
                    best_loss = loss
                    best_f = f
            current_subset.append(best_f)
            remaining.remove(best_f)
            exhaustive_results.append(evaluate_features(list(current_subset)))

    methods_dict = {
        'Lasso (L1 Ranking)': lasso_results,
        'Forward SFS': sfs_results,
        'RFE': rfe_results,
        'ElasticNet Ranking': enet_results,
        'Exhaustive Search': exhaustive_results
    }

    # Find global sweet spot
    best_item = None
    lowest_rmse = float('inf')
    for m, prog in methods_dict.items():
        for item in prog:
            if item['test_rmse'] < lowest_rmse:
                lowest_rmse = item['test_rmse']
                best_item = item

    return {
        'name': name,
        'target_col': target_col,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_all': feature_names,
        'max_k': eval_max_k,
        'sweet_spot_k': best_item['k'],
        'sweet_spot_features': best_item['features'],
        'sweet_spot_rmse': best_item['test_rmse'],
        'sweet_spot_r2': best_item['test_r2'],
        'methods_data': methods_dict
    }

# Run 1: 50 Startups
df_startups = pd.read_csv(os.path.join(BASE_DIR, "50_Startups.csv"))
startups_res = analyze_dataset(df_startups, target_col='Profit', name='50 Startups (Kaggle)', max_k=5, categorical_cols=['State'])

# Run 2: Boston Housing
df_boston = pd.read_csv(os.path.join(BASE_DIR, "Boston_Housing.csv"))
boston_res = analyze_dataset(df_boston, target_col='medv', name='Boston Housing Prices', max_k=13)

all_datasets = {
    '50_startups': startups_res,
    'boston_housing': boston_res
}

# Export combined json and js
with open(os.path.join(BASE_DIR, 'model_data.json'), 'w', encoding='utf-8') as f:
    json.dump(all_datasets, f, indent=2, ensure_ascii=False)

with open(os.path.join(BASE_DIR, 'model_data.js'), 'w', encoding='utf-8') as f:
    f.write(f"const MULTI_DATASET_STORE = {json.dumps(all_datasets, indent=2, ensure_ascii=False)};\n")
    # Maintain backwards compatibility
    f.write(f"const MODEL_DATA = MULTI_DATASET_STORE['50_startups'];\n")

print("\nSuccessfully processed both 50 Startups and Boston Housing datasets!")
