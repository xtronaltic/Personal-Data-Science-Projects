import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retail_forecast.io import load_panel
from retail_forecast.backtest import run_leave_one_brand_out_backtest, BacktestConfig
from retail_forecast.foundation_models import (
    FoundationModelConfig,
    TimesFMForecaster,
    ChronosBoltForecaster,
)
from retail_forecast.meta_ensemble import MetaEnsembleForecaster
from retail_forecast.calibration_v6 import apply_v6_calibration
from retail_forecast.calibration_v7 import apply_v7_calibration
from retail_forecast.constants import HIERARCHY_COLS
from retail_forecast.utils import set_global_seeds

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    set_global_seeds(42)
    
    print("="*80)
    print("FULL PIPELINE EXECUTION: FROM SCRATCH COMPARISON")
    print("1. Meta-Learner + V6 (Uncertainty-Aware Conditional)")
    print("2. Meta-Learner + V7 (Context-Aware Conditional)")
    print("="*80)
    
    start_time = time.time()

    # 1. Generate Analog Forecasts (LOBO)
    print("\n[STEP 1/6] Generating Analog Forecasts (Leave-One-Brand-Out)...")
    panel_path = Path("Dataset/Historical_Data.csv")
    
    # Configure backtest
    config = BacktestConfig(
        early_weeks=4,
        horizon=26,
        top_k=50,
        n_sims=1000, 
        include_ensemble=False, 
    )
    
    backtest_results, metrics, _ = run_leave_one_brand_out_backtest(
        panel_path=panel_path,
        config=config
    )
    
    print(f"  > Generated {len(backtest_results)} analog forecast rows.")
    
    # 2. Prepare Data for Foundation Models
    print("\n[STEP 2/6] Preparing Input Contexts for Foundation Models...")
    df_panel = load_panel(panel_path)
    
    group_cols = HIERARCHY_COLS + ['metric']
    unique_series = backtest_results[group_cols].drop_duplicates()
    print(f"  > Found {len(unique_series)} unique series to forecast with Foundation Models.")
    
    contexts = []
    metadata = []
    
    for _, row in unique_series.iterrows():
        mask = pd.Series(True, index=df_panel.index)
        for col in HIERARCHY_COLS:
            val = row[col]
            mask &= (df_panel[col] == val)
        
        series_data = df_panel[mask].sort_values('date')
        metric_col = row['metric']
        if metric_col not in series_data.columns:
            continue
            
        values = pd.to_numeric(series_data[metric_col], errors='coerce').dropna().values
        early_weeks = config.early_weeks
        if len(values) >= early_weeks:
            context = values[:early_weeks]
            contexts.append(context)
            
            # Compute Context Features (V7 Requirement)
            ctx_mean = np.mean(context)
            ctx_std = np.std(context)
            x = np.arange(len(context))
            slope = np.polyfit(x, context, 1)[0] if len(context) > 1 else 0
            
            meta_dict = row.to_dict()
            meta_dict['ctx_mean'] = ctx_mean
            meta_dict['ctx_std'] = ctx_std
            meta_dict['ctx_slope'] = slope
            metadata.append(meta_dict)
            
    print(f"  > Prepared {len(contexts)} contexts.")

    # 3. Generate Foundation Model Forecasts (Batch)
    print("\n[STEP 3/6] Running Foundation Models (TimesFM & Chronos)...")
    
    fm_config = FoundationModelConfig(prediction_length=26)
    
    # TimesFM
    print("  > Running TimesFM...")
    tfm = TimesFMForecaster(fm_config)
    try:
        tfm_results = []
        for ctx in tqdm(contexts, desc="TimesFM"):
            res = tfm.forecast(ctx)
            tfm_results.append(res)
    except Exception as e:
        print(f"    ! Error in TimesFM: {e}")
        tfm_results = [None] * len(contexts)

    # Chronos
    print("  > Running Chronos-Bolt...")
    chronos = ChronosBoltForecaster(fm_config)
    try:
        chronos_results = chronos.forecast_batch(contexts)
    except Exception as e:
        print(f"    ! Batch Chronos failed, trying sequential: {e}")
        chronos_results = []
        for ctx in tqdm(contexts, desc="Chronos"):
            try:
                res = chronos.forecast(ctx)
                chronos_results.append(res)
            except:
                chronos_results.append(None)

    # 4. Merge Forecasts & Features
    print("\n[STEP 4/6] Merging and Constructing Ensembles...")
    
    fm_lookup = {}
    ctx_lookup = {}
    for meta, tfm_res, chr_res in zip(metadata, tfm_results, chronos_results):
        key = tuple(meta[k] for k in group_cols)
        fm_lookup[key] = (tfm_res, chr_res)
        ctx_lookup[key] = (meta['ctx_mean'], meta['ctx_std'], meta['ctx_slope'])
    
    # Load Meta-Learner
    print("  > Loading Meta-Learner...")
    meta_learner = None
    try:
        meta_learner = MetaEnsembleForecaster()
    except Exception as e:
        print(f"    ! Could not load Meta-Learner ({e}). Will use Analog/Average fallback.")

    dfs_to_concat = []
    
    grouped = backtest_results.groupby(group_cols)
    
    for key, group in grouped:
        key_dict = dict(zip(group_cols, key))
        lookup_key = tuple(key_dict[k] for k in group_cols)
        
        # Initialize columns
        group = group.copy()
        
        # Context Features
        if lookup_key in ctx_lookup:
            c_mean, c_std, c_slope = ctx_lookup[lookup_key]
            group['ctx_mean'] = c_mean
            group['ctx_std'] = c_std
            group['ctx_slope'] = c_slope
        else:
            group['ctx_mean'] = np.nan
            group['ctx_std'] = np.nan
            group['ctx_slope'] = np.nan

        # Foundation Models
        analog_p50 = group['p50'].values
        
        if lookup_key in fm_lookup:
            tfm_res, chr_res = fm_lookup[lookup_key]
            
            indices = group['horizon_step'].values.astype(int) - 1
            t_vals = np.full(len(group), np.nan)
            c_vals = np.full(len(group), np.nan)
            
            if tfm_res is not None:
                valid = indices < len(tfm_res['p50'])
                t_vals[valid] = tfm_res['p50'][indices[valid]]
                
            if chr_res is not None:
                valid = indices < len(chr_res['p50'])
                c_vals[valid] = chr_res['p50'][indices[valid]]
            
            group['timesfm_p50'] = t_vals
            group['chronos_p50'] = c_vals
            group['analog_p50'] = analog_p50
            
            # Impute NaNs with Analog
            ana_arr = analog_p50
            tfm_arr = np.where(np.isnan(t_vals), analog_p50, t_vals)
            chr_arr = np.where(np.isnan(c_vals), analog_p50, c_vals)
            
            # Calculate Ensemble Features
            comps = np.column_stack([ana_arr, tfm_arr, chr_arr])
            ens_std = np.std(comps, axis=1)
            ens_range = np.max(comps, axis=1) - np.min(comps, axis=1)
            ens_mean = np.mean(comps, axis=1)
            
            group['ens_std'] = ens_std
            group['ens_range'] = ens_range
            group['ens_mean'] = ens_mean
            
            # Predict
            if meta_learner is not None:
                X_meta = pd.DataFrame({
                    'analog_p50': ana_arr,
                    'timesfm_p50': tfm_arr,
                    'chronos_p50': chr_arr,
                    'horizon_step': group['horizon_step'].values,
                    'ctx_mean': group['ctx_mean'].values,
                    'ctx_std': group['ctx_std'].values,
                    'ctx_slope': group['ctx_slope'].values,
                    'ens_std': ens_std,
                    'ens_range': ens_range,
                    'ens_mean': ens_mean
                })
                try:
                    group['meta_p50'] = meta_learner._model.predict(X_meta)
                except:
                    group['meta_p50'] = ens_mean
            else:
                group['meta_p50'] = ens_mean
                
            group['hybrid_p50'] = group['meta_p50']
            
        else:
            # Fallback
            group['timesfm_p50'] = np.nan
            group['chronos_p50'] = np.nan
            group['analog_p50'] = analog_p50
            group['meta_p50'] = analog_p50
            group['hybrid_p50'] = analog_p50
            
        dfs_to_concat.append(group)

    full_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # 5. Apply Calibration
    print("\n[STEP 5/6] Comparing Calibration Methods...")
    
    # --- Config 1: V6 (Baseline) ---
    print("  > Config 1: V6 (Uncertainty-Aware)...")
    try:
        df_v6 = apply_v6_calibration(full_df, p50_col='meta_p50')
        df_v6['final_p50'] = df_v6['v6_p50']
        df_v6['final_p10'] = df_v6['v6_p10']
        df_v6['final_p90'] = df_v6['v6_p90']
    except Exception as e:
        print(f"    ! V6 failed: {e}")
        df_v6 = full_df.copy()
        df_v6['final_p50'] = df_v6['meta_p50']
        df_v6['final_p10'] = df_v6['meta_p50'] * 0.5
        df_v6['final_p90'] = df_v6['meta_p50'] * 1.5

    # --- Config 2: V7 (New) ---
    print("  > Config 2: V7 (Context-Aware)...")
    try:
        df_v7 = apply_v7_calibration(full_df, p50_col='meta_p50')
        df_v7['final_p50'] = df_v7['v7_p50']
        df_v7['final_p10'] = df_v7['v7_p10']
        df_v7['final_p90'] = df_v7['v7_p90']
    except Exception as e:
        print(f"    ! V7 failed: {e}")
        df_v7 = full_df.copy() # Fallback
        df_v7['final_p50'] = df_v7['meta_p50']
        df_v7['final_p10'] = df_v7['meta_p50'] * 0.5
        df_v7['final_p90'] = df_v7['meta_p50'] * 1.5

    # --- SAVE PRODUCTION RESULTS (V7) ---
    print("\n[SAVING] Updating outputs/hybrid_production_results.csv with V7 Results...")
    
    def build_key(r):
        return f"{r['trademark']}||{r['brand']}||{r['markets']}||{r.get('manufacturer','')}||{r.get('category','')}"

    df_save = df_v7.copy()
    df_save['key'] = df_save.apply(build_key, axis=1)
    df_save['hybrid_p50'] = df_save['final_p50']
    df_save['prod_p10'] = df_save['final_p10']
    df_save['prod_p90'] = df_save['final_p90']
    
    cols_to_save = [
        'key', 'y_true', 'horizon_step', 'metric',
        'hybrid_p50', 'prod_p10', 'prod_p90', 'prod_p50',
        'analog_p50', 'timesfm_p50', 'chronos_p50',
        'ctx_mean', 'ctx_std', 'ctx_slope',
        'markets', 'trademark', 'brand'
    ]
    cols_to_save = [c for c in cols_to_save if c in df_save.columns]
    
    df_save[cols_to_save].to_csv('outputs/hybrid_production_results.csv', index=False)
    print("  > Saved successfully.")

    # 6. Compute Metrics
    print("\n[STEP 6/6] Computing Final Metrics...")
    
    def get_metrics(df, name):
        y = df['y_true'].values
        p50 = df['final_p50'].values
        p10 = df['final_p10'].values
        p90 = df['final_p90'].values
        
        wmape = np.sum(np.abs(y - p50)) / np.sum(np.abs(y)) * 100
        coverage = np.mean((y >= p10) & (y <= p90)) * 100
        width = np.median((p90 - p10) / np.maximum(p50, 1)) * 100
        
        alpha = 0.20
        score = ((p90 - p10) + 
                 (2/alpha) * np.maximum(p10 - y, 0) + 
                 (2/alpha) * np.maximum(y - p90, 0)).mean()
        
        return {
            'Name': name,
            'WMAPE (%)': wmape,
            'Coverage (%)': coverage,
            'Rel Width (%)': width,
            'Interval Score': score
        }
    
    m1 = get_metrics(df_v6, "V6 (Baseline)")
    m2 = get_metrics(df_v7, "V7 (New Context-Aware)")
    
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    
    res_df = pd.DataFrame([m1, m2])
    diff = {
        'Name': 'Improvement (V7 vs V6)',
        'WMAPE (%)': m1['WMAPE (%)'] - m2['WMAPE (%)'],
        'Coverage (%)': m2['Coverage (%)'] - m1['Coverage (%)'], # Higher better
        'Rel Width (%)': m1['Rel Width (%)'] - m2['Rel Width (%)'], # Lower better (positive diff = improvement)
        'Interval Score': m1['Interval Score'] - m2['Interval Score'] # Lower better
    }
    res_df = pd.concat([res_df, pd.DataFrame([diff])], ignore_index=True)
    
    print(res_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*80)
    
    elapsed = time.time() - start_time
    print(f"\nTotal Execution Time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
