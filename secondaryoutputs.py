import pandas as pd
import numpy as np
import glob
import os
import gc
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# !pip install lightgbm xgboost catboost optuna -q

import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

INPUT_DIR = "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input/mindshift-analytics-haul-mark-challenge"

FEATURES_PATH = '/kaggle/input/datasets/nvdpda24b046/cached-features-parquet/cached_features.parquet'


ID_MAPPING_PATH = "/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge/id_mapping_new.csv" 


def load_data():
    """Loads target training data and the submission ID mappings."""
    if not os.path.exists(ID_MAPPING_PATH):
        raise FileNotFoundError(f"Could not find {ID_MAPPING_PATH}. Please check the path!")
        
    id_mapping = pd.read_csv(ID_MAPPING_PATH)
    
    smry_files = glob.glob(os.path.join(INPUT_DIR, 'smry_*_train_ordered.csv'))
    if not smry_files:
        smry_files = glob.glob(os.path.join(INPUT_DIR, '*', 'smry_*_train_ordered.csv'))
        
    df_train = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
    return id_mapping, df_train

def get_features():
    """Loads the cached features parquet file from the specified dataset path."""
    
    if os.path.exists(FEATURES_PATH):
        return pd.read_parquet(FEATURES_PATH)
    elif os.path.exists('/kaggle/working/cached_features.parquet'):
        return pd.read_parquet('/kaggle/working/cached_features.parquet')
    else:
        raise FileNotFoundError(f"{FEATURES_PATH} not found.")

def generate_secondary_reports(train_df):
    """Generates all separate secondary business outputs required for the report."""
    
    route_features = ['distance_travelled', 'net_lift', 'mean_altitude', 
                      'std_altitude', 'max_altitude', 'min_altitude']
    X_route = train_df[route_features].fillna(0)
    
    benchmark_model = LGBMRegressor(n_estimators=150, max_depth=5, random_state=42, verbosity=-1)
    benchmark_model.fit(X_route, train_df['acons'])
    train_df['expected_fuel_independent_of_dumper'] = benchmark_model.predict(X_route)
    
    benchmark_output = train_df[['vehicle', 'date', 'shift', 'distance_travelled', 'net_lift', 
                                 'expected_fuel_independent_of_dumper', 'acons']].copy()
    benchmark_output.rename(columns={'acons': 'actual_fuel'}, inplace=True)
    benchmark_output = benchmark_output.sort_values(by=['vehicle', 'date', 'shift'])
    
    benchmark_output.to_csv('/kaggle/working/1_route_benchmark.csv', index=False)
    
    train_df['fuel_wasted'] = train_df['acons'] - train_df['expected_fuel_independent_of_dumper']
    
    # 2a. Dumper
    dumper_eff = train_df.groupby('vehicle').agg(
        total_shifts=('shift', 'count'),
        avg_actual_fuel=('acons', 'mean'),
        avg_expected_fuel=('expected_fuel_independent_of_dumper', 'mean'),
        avg_fuel_wasted_vs_benchmark=('fuel_wasted', 'mean')
    ).reset_index().sort_values('avg_fuel_wasted_vs_benchmark')
    dumper_eff.to_csv('/kaggle/working/2a_dumper_efficiency.csv', index=False)
    
    # 2b. Operator
    op_eff = train_df.groupby('operator_id').agg(
        total_shifts=('shift', 'count'),
        avg_fuel_wasted_vs_benchmark=('fuel_wasted', 'mean')
    ).reset_index().sort_values('avg_fuel_wasted_vs_benchmark')
    op_eff.to_csv('/kaggle/working/2b_operator_efficiency.csv', index=False)
    
   

        
    daily_totals = train_df.groupby(['vehicle', 'date']).agg(
        daily_actual_fuel=('acons', 'sum'),
        daily_predicted_fuel=('pred_acons', 'sum')
    ).reset_index()
    
    daily_totals['calibration_factor'] = daily_totals['daily_actual_fuel'] / (daily_totals['daily_predicted_fuel'] + 1e-9)
    train_df = pd.merge(train_df, daily_totals[['vehicle', 'date', 'calibration_factor']], on=['vehicle', 'date'], how='left')
    train_df['calibrated_pred_acons'] = train_df['pred_acons'] * train_df['calibration_factor']
    
    daily_consistency = train_df[['vehicle', 'date', 'shift', 'acons', 'pred_acons', 'calibrated_pred_acons', 'calibration_factor']].copy()
    daily_consistency.rename(columns={'acons': 'actual_shift_fuel', 'pred_acons': 'raw_model_prediction', 'calibrated_pred_acons': 'daily_calibrated_prediction'}, inplace=True)
    daily_consistency = daily_consistency.sort_values(by=['vehicle', 'date', 'shift'])
    
    daily_consistency.to_csv('/kaggle/working/4_daily_fuel_consistency.csv', index=False)




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
        train_df[c] = train_df[c].fillna('UNKNOWN').astype(str)
        test_df[c] = test_df[c].fillna('UNKNOWN').astype(str)
        
    train_df = train_df.dropna(subset=['acons']).reset_index(drop=True)
    
    X_base = train_df[features].copy()
    y = train_df['acons']
    X_test_base = test_df[features].copy()
    
    # Model Setup
    X_cb = X_base.copy()
    X_test_cb = X_test_base.copy()
    
    X_lgbm = X_base.copy()
    X_test_lgbm = X_test_base.copy()
    for c in cat_cols:
        X_lgbm[c] = X_lgbm[c].astype('category')
        X_test_lgbm[c] = pd.Categorical(X_test_lgbm[c], categories=X_lgbm[c].cat.categories)
        
    X_xgb = X_lgbm.copy()
    X_test_xgb = X_test_lgbm.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].cat.codes
        X_test_xgb[c] = X_test_xgb[c].cat.codes
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Optuna
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

    print("Running Optuna for CatBoost...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_cb = optuna.create_study(direction='minimize')
    study_cb.optimize(objective_cb, n_trials=10)
    best_cb_params = study_cb.best_params
    best_cb_params['verbose'] = 0

    print("\nTraining Final Ensembles...")
    cb_test_preds = np.zeros(len(X_test_base))
    lgbm_test_preds = np.zeros(len(X_test_base))
    xgb_test_preds = np.zeros(len(X_test_base))
    
    cb_oof_preds = np.zeros(len(X_base))
    lgbm_oof_preds = np.zeros(len(X_base))
    xgb_oof_preds = np.zeros(len(X_base))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_base)):
        print(f"--- Training Fold {fold+1}/5 ---")
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        # CatBoost
        X_tr_cb, X_va_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
        cb = CatBoostRegressor(**best_cb_params, cat_features=cat_cols)
        cb.fit(X_tr_cb, y_tr, verbose=0)
        cb_oof_preds[val_idx] = np.clip(cb.predict(X_va_cb), 0, None)
        cb_test_preds += np.clip(cb.predict(X_test_cb), 0, None) / 5.0
        
        # LightGBM
        X_tr_lgbm, X_va_lgbm = X_lgbm.iloc[train_idx], X_lgbm.iloc[val_idx]
        lgbm = LGBMRegressor(n_estimators=600, learning_rate=0.03, max_depth=7, 
                             num_leaves=31, subsample=0.8, colsample_bytree=0.8, 
                             random_state=42, verbosity=-1)
        lgbm.fit(X_tr_lgbm, y_tr, categorical_feature=cat_cols)
        lgbm_oof_preds[val_idx] = np.clip(lgbm.predict(X_va_lgbm), 0, None)
        lgbm_test_preds += np.clip(lgbm.predict(X_test_lgbm), 0, None) / 5.0
        
        # XGBoost
        X_tr_xgb, X_va_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
        xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, 
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=0)
        xgb_oof_preds[val_idx] = np.clip(xgb.predict(X_va_xgb), 0, None)
        xgb_test_preds += np.clip(xgb.predict(X_test_xgb), 0, None) / 5.0

    weights = {'cb': 0.45, 'lgbm': 0.40, 'xgb': 0.15}
    
    final_test_preds = (cb_test_preds * weights['cb']) + \
                       (lgbm_test_preds * weights['lgbm']) + \
                       (xgb_test_preds * weights['xgb'])
                       
    final_train_preds = (cb_oof_preds * weights['cb']) + \
                        (lgbm_oof_preds * weights['lgbm']) + \
                        (xgb_oof_preds * weights['xgb'])

    train_df['pred_acons'] = final_train_preds
    generate_secondary_reports(train_df)

if __name__ == "__main__":
    main()
