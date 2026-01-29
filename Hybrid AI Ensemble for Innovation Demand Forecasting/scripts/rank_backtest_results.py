
import pandas as pd
import numpy as np

def main():
    print("Ranking Backtest Results by WMAPE...")
    
    # Load results
    try:
        df = pd.read_csv("outputs/hybrid_production_results.csv")
    except FileNotFoundError:
        print("Error: outputs/hybrid_production_results.csv not found.")
        return

    # Filter for EQ metric
    df_eq = df[df['metric'] == 'eq'].copy()
    
    if df_eq.empty:
        print("No EQ metrics found.")
        return

    # Group by series
    group_cols = ['brand', 'markets', 'trademark']
    
    results = []
    
    for key_vals, group in df_eq.groupby(group_cols):
        # Require full horizon (14 weeks)
        if len(group) < 14:
            continue
            
        y_true = group['y_true'].values
        y_pred = group['hybrid_p50'].values 
        p10 = group['prod_p10'].values
        p90 = group['prod_p90'].values
        
        # Avoid division by zero
        sum_actual = np.sum(np.abs(y_true))
        if sum_actual < 1.0: 
            continue
            
        wmape = np.sum(np.abs(y_true - y_pred)) / sum_actual * 100
        coverage = np.mean((y_true >= p10) & (y_true <= p90)) * 100
        
        # Store info
        record = dict(zip(group_cols, key_vals))
        record['wmape'] = wmape
        record['coverage'] = coverage
        record['total_eq'] = sum_actual
        results.append(record)
        
    # Convert to DF
    df_rank = pd.DataFrame(results)
    
    # Sort by WMAPE (Ascending)
    df_rank = df_rank.sort_values('wmape')
    
    print("\n" + "="*80)
    print("TOP 3 BRANDS (by Lowest WMAPE - EQ Volume)")
    print("="*80)
    
    top3 = df_rank.head(3)
    
    for i, row in top3.iterrows():
        print(f"\nRANK #{i+1}:")
        print(f"  Brand:    {row['brand']}")
        print(f"  Channel:  {row['markets']}")
        print(f"  WMAPE:    {row['wmape']:.2f}%")
        print(f"  Coverage: {row['coverage']:.1f}%")
        print(f"  Total EQ: {row['total_eq']:,.0f}")
        
    print("\n" + "="*80)
    print("WORST 3 BRANDS (by Highest WMAPE - EQ Volume)")
    print("="*80)
    
    bottom3 = df_rank.tail(3)
    for i, row in bottom3.iterrows():
        print(f"\nBOTTOM #{len(df_rank)-i}:")
        print(f"  Brand:    {row['brand']}")
        print(f"  Channel:  {row['markets']}")
        print(f"  WMAPE:    {row['wmape']:.2f}%")
        print(f"  Coverage: {row['coverage']:.1f}%")
        print(f"  Total EQ: {row['total_eq']:,.0f}")

if __name__ == "__main__":
    main()
