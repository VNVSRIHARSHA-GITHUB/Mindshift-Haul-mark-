import pandas as pd
import numpy as np
import glob
import os
import gc
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# Kaggle specific installations (uncomment if running in a fresh environment)
# !pip install lightgbm xgboost catboost optuna -q

import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

INPUT_DIR = "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge"
#should change the path accordingly
FEATURES_PATH = '/kaggle/input/datasets/nvdpda24b046/cached-features-parquet/cached_features.parquet'

ID_MAPPING_PATH = "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge/id_mapping_new.csv" 

def load_data():
    """Loads target training data and the submission ID mappings."""
    print(f"Loading ID mapping directly from: {ID_MAPPING_PATH}")
    if not os.path.exists(ID_MAPPING_PATH):
        raise FileNotFoundError(f"Could not find {ID_MAPPING_PATH}. Please check the path!")
        
    id_mapping = pd.read_csv(ID_MAPPING_PATH)
    
    # Locate summary files
    smry_files = glob.glob(os.path.join(INPUT_DIR, 'smry_*_train_ordered.csv'))
    if not smry_files:
        smry_files = glob.glob(os.path.join(INPUT_DIR, '*', 'smry_*_train_ordered.csv'))
        
    print(f"Found {len(smry_files)} summary files. Merging...")
    df_train = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
    return id_mapping, df_train

def get_features():
    """Loads the cached features parquet file from the specified dataset path."""
    if os.path.exists(FEATURES_PATH):
        print(f"Loading features from {FEATURES_PATH}")
        return pd.read_parquet(FEATURES_PATH)
    elif os.path.exists('/kaggle/working/cached_features.parquet'):
        print("Loading features from /kaggle/working/cached_features.parquet")
        return pd.read_parquet('/kaggle/working/cached_features.parquet')
    else:
        raise FileNotFoundError(f"{FEATURES_PATH} not found. Please ensure the dataset is attached.")

def main():
  
    id_mapping, target_df = load_data()
    features_df = get_features()
    
    target_df['date'] = pd.to_datetime(target_df['date']).dt.strftime('%Y-%m-%d')
    id_mapping['date'] = pd.to_datetime(id_mapping['date']).dt.strftime('%Y-%m-%d')
    features_df['date'] = pd.to_datetime(features_df['date']).dt.strftime('%Y-%m-%d')
    
    train_df = pd.merge(target_df, features_df, on=['vehicle', 'date', 'shift'], how='left')
    test_df = pd.merge(id_mapping, features_df, on=['vehicle', 'date', 'shift'], how='left')
    
    cat_cols = ['vehicle', 'shift', 'operator_id']
    num_cols = ['mean_speed', 'max_speed', 'mean_altitude', 'std_altitude', 
                'max_altitude', 'min_altitude', 'idle_pings', 'total_pings', 
                'min_cumdist', 'max_cumdist', 
                'distance_travelled', 'net_lift']
                
    features = cat_cols + num_cols
    
    for c in cat_cols:
        # Convert to string and fill NaNs with 'UNKNOWN'
        train_df[c] = train_df[c].fillna('UNKNOWN').astype(str)
        test_df[c] = test_df[c].fillna('UNKNOWN').astype(str)
        
    train_df = train_df.dropna(subset=['acons']).reset_index(drop=True)
    
    X_base = train_df[features].copy()
    y = train_df['acons']
    X_test_base = test_df[features].copy()
    

    
    # 1. CatBoost
    X_cb = X_base.copy()
    X_test_cb = X_test_base.copy()
    
    # 2. LightGBM 
    X_lgbm = X_base.copy()
    X_test_lgbm = X_test_base.copy()
    for c in cat_cols:
        X_lgbm[c] = X_lgbm[c].astype('category')
        X_test_lgbm[c] = pd.Categorical(X_test_lgbm[c], categories=X_lgbm[c].cat.categories)
        
    # 3. XGBoost
    X_xgb = X_lgbm.copy()
    X_test_xgb = X_test_lgbm.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].cat.codes
        # Unseen categories in test get mapped to -1
        X_test_xgb[c] = X_test_xgb[c].cat.codes
        
    print(f"Train shapes: X={X_base.shape}, y={len(y)}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective_cb(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1000),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_seed': 42,
            'verbose': 0
        }
        scores = []
        for train_idx, val_idx in kf.split(X_cb):
            X_tr, X_va = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            model = CatBoostRegressor(**params, cat_features=cat_cols)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)
            preds = model.predict(X_va)
            scores.append(root_mean_squared_error(y_va, np.clip(preds, 0, None)))
        return np.mean(scores)


    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_cb = optuna.create_study(direction='minimize')
    study_cb.optimize(objective_cb, n_trials=10)
    print(f"Best CatBoost RMSE: {study_cb.best_value:.4f}")
    
    best_cb_params = study_cb.best_params
    best_cb_params['verbose'] = 0

    # Final Stack Ensembles


    cb_test_preds = np.zeros(len(X_test_base))
    lgbm_test_preds = np.zeros(len(X_test_base))
    xgb_test_preds = np.zeros(len(X_test_base))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_base)):

        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        # CatBoost
        X_tr_cb = X_cb.iloc[train_idx]
        cb = CatBoostRegressor(**best_cb_params, cat_features=cat_cols)
        cb.fit(X_tr_cb, y_tr, verbose=0)
        cb_test_preds += np.clip(cb.predict(X_test_cb), 0, None) / 5.0
        
        # LightGBM
        X_tr_lgbm = X_lgbm.iloc[train_idx]
        lgbm = LGBMRegressor(
            n_estimators=600, 
            learning_rate=0.03, 
            max_depth=7, 
            num_leaves=31, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            verbosity=-1
        )
        lgbm.fit(X_tr_lgbm, y_tr, categorical_feature=cat_cols)
        lgbm_test_preds += np.clip(lgbm.predict(X_test_lgbm), 0, None) / 5.0
        
        # XGBoost
        X_tr_xgb, X_va_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
        xgb = XGBRegressor(
            n_estimators=600, 
            learning_rate=0.03, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        )
        xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=0)
        xgb_test_preds += np.clip(xgb.predict(X_test_xgb), 0, None) / 5.0

    print("\nAveraging Predictions from all 3 models...")
    # Weights scaled towards CatBoost and LightGBM
    final_preds = (cb_test_preds * 0.45) + (lgbm_test_preds * 0.40) + (xgb_test_preds * 0.15)
    
    # Save the output to Kaggle's working directory
    output_path = '/kaggle/working/submission_ensemble.csv'
    sub = pd.DataFrame({'id': id_mapping['id'], 'acons': final_preds})
    sub.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()