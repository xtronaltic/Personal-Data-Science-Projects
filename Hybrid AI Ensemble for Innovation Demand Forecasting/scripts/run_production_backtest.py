
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
    OPTIMIZED_HYBRID_WEIGHTS,
    OPTIMIZED_3WAY_WEIGHTS
)
from retail_forecast.meta_ensemble import MetaEnsembleForecaster
from retail_forecast.calibration_v7 import apply_v7_calibration
from retail_forecast.constants import HIERARCHY_COLS

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("PRODUCTION BACKTEST: META-LEARNER + V7 CALIBRATION")
    print("="*80)
    
    start_time = time.time()

    # 1. Generate Analog Forecasts (LOBO)
    print("\n[STEP 1/5] Generating Analog Forecasts (Leave-One-Brand-Out)...")
    panel_path = Path("Dataset/Historical_Data.csv")
    
    # Configure backtest
    config = BacktestConfig(
        early_weeks=4,
        horizon=26,
        top_k=50,
        n_sims=1000, 
        include_ensemble=False, # We only need Analog raw output here
    )
    
    # This runs the rigorous LOBO process for Analog
    backtest_results, metrics, _ = run_leave_one_brand_out_backtest(
        panel_path=panel_path,
        config=config
    )
    
    print(f"  > Generated {len(backtest_results)} analog forecast rows.")
    
    # 2. Prepare Data for Foundation Models
    print("\n[STEP 2/5] Preparing Input Contexts for Foundation Models...")
    df_panel = load_panel(panel_path)
    
    # columns in backtest_results: markets, manufacturer, ..., metric, horizon_step, y_true, p50...
    group_cols = HIERARCHY_COLS + ['metric']
    
    # Get unique series from backtest results
    unique_series = backtest_results[group_cols].drop_duplicates()
    print(f"  > Found {len(unique_series)} unique series to forecast with Foundation Models.")
    
    contexts = []
    metadata = []
    
    for _, row in unique_series.iterrows():
        # Filter panel data for this series
        mask = pd.Series(True, index=df_panel.index)
        for col in HIERARCHY_COLS:
            val = row[col]
            mask &= (df_panel[col] == val)
            
        series_data = df_panel[mask].sort_values('date')
        
        # Metric column
        metric_col = row['metric']
        if metric_col not in series_data.columns:
            continue
            
        # Extract values
        values = pd.to_numeric(series_data[metric_col], errors='coerce').dropna().values
        
        # The backtest used first 'early_weeks' as context.
        early_weeks = config.early_weeks
        if len(values) >= early_weeks:
            context = values[:early_weeks]
            contexts.append(context)
            
            # Compute Context Features
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

    # Run foundation models (TimesFM + Chronos)
    print("\n2. Running Foundation Models (TimesFM + Chronos)...")
    fm_config = FoundationModelConfig(prediction_length=26)
    fm_ensemble = FoundationModelEnsemble(config=fm_config, include_analog=False)
    
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

    # 4. Merge Forecasts & Run Meta-Learner
    print("\n[STEP 4/5] Running Meta-Learner (Stacking)...")
    
    # Mapping logic
    fm_lookup = {}
    ctx_lookup = {}
    for meta, tfm_res, chr_res in zip(metadata, tfm_results, chronos_results):
        key = tuple(meta[k] for k in group_cols)
        fm_lookup[key] = (tfm_res, chr_res)
        ctx_lookup[key] = (meta['ctx_mean'], meta['ctx_std'], meta['ctx_slope'])
    
    # Load Meta-Learner
    print("  > Loading XGBoost Meta-Learner...")
    meta_learner = MetaEnsembleForecaster()
    
    dfs_to_concat = []
    grouped = backtest_results.groupby(group_cols)
    
    for key, group in grouped:
        key_dict = dict(zip(group_cols, key))
        lookup_key = tuple(key_dict[k] for k in group_cols)
        
        # Initialize default columns
        group = group.copy()
        
        # Extract Analog
        analog_p50 = group['p50'].values
        
        # Extract Context Features
        if lookup_key in ctx_lookup:
            c_mean, c_std, c_slope = ctx_lookup[lookup_key]
            group['ctx_mean'] = c_mean
            group['ctx_std'] = c_std
            group['ctx_slope'] = c_slope
        else:
            group['ctx_mean'] = np.nan
            group['ctx_std'] = np.nan
            group['ctx_slope'] = np.nan

        if lookup_key in fm_lookup:
            tfm_res, chr_res = fm_lookup[lookup_key]
            
            indices = group['horizon_step'].values.astype(int) - 1
            t_vals = np.full(len(group), np.nan)
            c_vals = np.full(len(group), np.nan)
            
            # Extract TimesFM
            if tfm_res is not None:
                valid = indices < len(tfm_res['p50'])
                t_vals[valid] = tfm_res['p50'][indices[valid]]
                
            # Extract Chronos
            if chr_res is not None:
                valid = indices < len(chr_res['p50'])
                c_vals[valid] = chr_res['p50'][indices[valid]]
            
            # Store components for V6 feature engineering
            group['timesfm_p50'] = t_vals
            group['chronos_p50'] = c_vals
            group['analog_p50'] = analog_p50
            
            # Impute NaNs with Analog (fallback) for Meta-Learner input
            ana_arr = analog_p50
            tfm_arr = np.where(np.isnan(t_vals), analog_p50, t_vals)
            chr_arr = np.where(np.isnan(c_vals), analog_p50, c_vals)
            
            # Calculate Ensemble Features
            comps = np.column_stack([ana_arr, tfm_arr, chr_arr])
            ens_std = np.std(comps, axis=1)
            ens_range = np.max(comps, axis=1) - np.min(comps, axis=1)
            ens_mean = np.mean(comps, axis=1)
            
            # Run Meta-Learner Prediction
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
            group['meta_p50'] = meta_learner._model.predict(X_meta)
            
            # Set hybrid_p50 for V6 calibration
            group['hybrid_p50'] = group['meta_p50']
            
        else:
            # Fallback if foundation models failed
            group['timesfm_p50'] = np.nan
            group['chronos_p50'] = np.nan
            group['analog_p50'] = analog_p50
            group['meta_p50'] = analog_p50
            group['hybrid_p50'] = analog_p50
            
        dfs_to_concat.append(group)

    full_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # 5. Apply V7 Calibration
    print("\n[STEP 5/5] Applying V7 Calibration (Context-Aware)...")
    
    try:
        # V7 uses pre-trained XGBoost models to predict interval width
        # It requires: analog_p50, timesfm_p50, chronos_p50, horizon_step, ctx_mean...
        df_calibrated = apply_v7_calibration(full_df, p50_col='meta_p50')
        
        # Set final production columns
        df_calibrated['prod_p50'] = df_calibrated['v7_p50']
        df_calibrated['prod_p10'] = df_calibrated['v7_p10']
        df_calibrated['prod_p90'] = df_calibrated['v7_p90']
        
        # Calculate Metrics
        y = df_calibrated['y_true'].values
        p50 = df_calibrated['prod_p50'].values
        p10 = df_calibrated['prod_p10'].values
        p90 = df_calibrated['prod_p90'].values
        
        wmape = np.sum(np.abs(y - p50)) / np.sum(np.abs(y)) * 100
        coverage = np.mean((y >= p10) & (y <= p90)) * 100
        width = np.median((p90 - p10) / np.maximum(p50, 1)) * 100
        
        print("\n" + "="*60)
        print("FINAL PRODUCTION METRICS")
        print("="*60)
        print(f"WMAPE:     {wmape:.2f}%")
        print(f"Coverage:  {coverage:.2f}%")
        print(f"Rel Width: {width:.0f}%")
        print("="*60)
        
        # Save Results
        print("\n[SAVING] Updating outputs/hybrid_production_results.csv...")
        
        def build_key(r):
            return f"{r['trademark']}||{r['brand']}||{r['markets']}||{r.get('manufacturer','')}||{r.get('category','')}"

        df_calibrated['key'] = df_calibrated.apply(build_key, axis=1)
        
        # Ensure we save all necessary columns
        cols_to_save = [
            'key', 'y_true', 'horizon_step', 'metric',
            'hybrid_p50', 'prod_p10', 'prod_p90', 'prod_p50',
            'analog_p50', 'timesfm_p50', 'chronos_p50',
            'ctx_mean', 'ctx_std', 'ctx_slope',
            'markets', 'trademark', 'brand'
        ]
        # Only save columns that exist
        cols_to_save = [c for c in cols_to_save if c in df_calibrated.columns]
        
        df_calibrated[cols_to_save].to_csv('outputs/hybrid_production_results.csv', index=False)
        print("  > Saved successfully.")
        
    except Exception as e:
        print(f"    ! V6 Calibration failed: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nTotal Execution Time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
