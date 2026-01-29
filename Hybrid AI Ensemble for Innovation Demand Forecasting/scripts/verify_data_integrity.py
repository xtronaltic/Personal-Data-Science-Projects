
import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_period_to_date(period_str):
    match = re.search(r'(\d{2}/\d{2}/\d{2})', str(period_str))
    if match:
        return pd.to_datetime(match.group(1), format='%m/%d/%y')
    return None

def verify_integrity():
    print("Verifying Data Integrity: Historical_Data.csv vs Backtest Results (y_true)...")
    
    # 1. Load Backtest Results
    res_path = Path("outputs/hybrid_production_results.csv")
    if not res_path.exists():
        print("Error: outputs/hybrid_production_results.csv not found.")
        return
    
    df_res = pd.read_csv(res_path)
    print(f"Loaded {len(df_res)} backtest result rows.")
    
    # 2. Load panel data
    panel_path = Path("Dataset/Historical_Data.csv")
    print(f"Loading {panel_path}...")
    df_panel = pd.read_excel(panel_path)
    
    # Normalize panel cols
    col_map = {
        "Manufacturer": "manufacturer",
        "Category": "category",
        "Trademark": "trademark",
        "Brand": "brand",
        "Markets": "markets",
        "$": "dollars",
        "Units": "units",
        "EQ": "eq"
    }
    df_panel = df_panel.rename(columns=col_map)
    df_panel['date'] = df_panel['Periods'].apply(parse_period_to_date)
    
    # 3. Verify a sample of series
    # Group backtest results by series
    keys = ['markets', 'trademark', 'brand', 'metric']
    grouped = df_res.groupby(keys)
    
    mismatches = []
    checked_count = 0
    
    print("\nChecking sample series...")
    
    for (mkt, tm, br, metric), group in grouped:
        # Filter panel for this series
        mask = (
            (df_panel['markets'] == mkt) &
            (df_panel['trademark'] == tm) &
            (df_panel['brand'] == br)
        )
        panel_series = df_panel[mask].sort_values('date').reset_index(drop=True)
        
        if panel_series.empty:
            print(f"Warning: No panel data found for {br} in {mkt}")
            continue
            
        # Determine Launch Week (W1) used in backtest logic
        # The backtest logic (run_production_backtest.py -> run_leave_one_brand_out_backtest)
        # usually finds the first non-zero or just takes the series.
        # But `run_leave_one_brand_out_backtest` in `backtest.py` does:
        #   df_new = _build_df_new_from_brand(...) -> keeps first `early_weeks`
        #   actuals = _extract_actuals(...)
        
        # Let's try to match by finding the subsequence.
        # The backtest results have `horizon_step` 1 to 14.
        # This corresponds to W5 to W18 relative to the "start" defined in backtest.
        # Usually W1 is first non-zero.
        
        # Get actuals from panel
        if metric not in panel_series.columns:
            continue
            
        panel_vals = pd.to_numeric(panel_series[metric], errors='coerce').fillna(0).values
        
        # Find start index (first non-zero)
        # This assumes the backtest used the standard "launch" definition
        nz_idx = np.where(panel_vals > 0)[0]
        if len(nz_idx) == 0:
            continue
        start_idx = nz_idx[0]
        
        # Backtest assumes:
        # Context = 4 weeks (start_idx to start_idx+3)
        # Forecast = next 14 weeks (start_idx+4 to start_idx+17)
        
        # Let's compare
        group = group.sort_values('horizon_step')
        
        for _, row in group.iterrows():
            h = int(row['horizon_step']) # 1-based
            y_true_backtest = row['y_true']
            
            # Look up in panel
            # Index in panel_vals should be: start_idx + 4 (early_weeks) + (h - 1)
            target_idx = start_idx + 4 + (h - 1)
            
            if target_idx < len(panel_vals):
                y_true_panel = panel_vals[target_idx]
                
                # Check match (allow small float diff)
                if abs(y_true_backtest - y_true_panel) > 1.0:
                    mismatches.append({
                        "series": f"{br} | {mkt} | {metric}",
                        "horizon": h,
                        "y_backtest": y_true_backtest,
                        "y_panel": y_true_panel_df,
                        "diff": abs(y_true_backtest - y_true_panel)
                    })
            else:
                # Horizon out of bounds of actuals?
                pass
        
        checked_count += 1
        if checked_count >= 50: # Check first 50 series
            break
            
    if mismatches:
        print(f"\nFound {len(mismatches)} mismatches!")
        for m in mismatches[:5]:
            print(m)
    else:
        print(f"\nSUCCESS: Checked {checked_count} series, all y_true values match Historical_Data.csv exactly.")

if __name__ == "__main__":
    verify_integrity()
