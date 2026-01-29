import pandas as pd
import subprocess
import sys
import re
from pathlib import Path
from tqdm import tqdm

def parse_period_to_date(period_str):
    match = re.search(r'(\d{2}/\d{2}/\d{2})', str(period_str))
    if match:
        return pd.to_datetime(match.group(1), format='%m/%d/%y')
    return None

def find_eligible_brands(panel_path):
    print(f"Loading {panel_path} to find eligible brands...")
    df = pd.read_excel(panel_path)
    
    # Normalize cols
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
    df = df.rename(columns=col_map)
    df['date'] = df['Periods'].apply(parse_period_to_date)
    
    eligible = []
    
    # Group by full hierarchy
    keys = ['markets', 'manufacturer', 'category', 'trademark', 'brand']
    
    for key_vals, group in tqdm(df.groupby(keys), desc="Scanning brands"):
        group = group.sort_values('date')
        
        # Check if length >= 18 weeks from first non-zero dollar sale
        # (Assuming checking dollars is enough proxy for 'launched')
        if 'dollars' in group.columns:
            nz = group[group['dollars'] > 0]
            if len(nz) >= 30:
                # Store the combo
                record = dict(zip(keys, key_vals))
                eligible.append(record)
                
    return pd.DataFrame(eligible)

def main():
    panel_path = "Dataset/Historical_Data.csv"
    eligible_path = "outputs/eligible_brands.csv"
    
    # 1. Find or Load Eligible Brands
    if Path(eligible_path).exists():
        print(f"Loading eligible brands from {eligible_path}...")
        df_eligible = pd.read_csv(eligible_path)
    else:
        df_eligible = find_eligible_brands(panel_path)
        df_eligible.to_csv(eligible_path, index=False)
        print(f"Saved {len(df_eligible)} eligible brands to {eligible_path}")

    print(f"Found {len(df_eligible)} brands to process.")
    
    # 2. Run Case Studies
    metrics = ["dollars", "units", "eq"]
    
    # Ensure output dir
    Path("outputs/Production Readiness Report").mkdir(parents=True, exist_ok=True)
    
    # We will run sequentially.
    # Total runs = Brands * 3
    
    for i, row in df_eligible.iterrows():
        print(f"\n[{i+1}/{len(df_eligible)}] Brand: {row['brand']} ({row['markets']})")
        
        for metric in metrics:
            print(f"   > Running for {metric.upper()}...", end="", flush=True)
            
            cmd = [
                sys.executable, "Case_Study.py",
                "--channel", str(row['markets']),
                "--manufacturer", str(row['manufacturer']),
                "--category", str(row['category']),
                "--trademark", str(row['trademark']),
                "--brand", str(row['brand']),
                "--metric", metric,
                "--calibration", "v7"
            ]
            
            try:
                # Run silently unless error
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(" DONE")
                else:
                    print(" FAILED")
                    print(f"     Error: {result.stderr.strip().splitlines()[-1] if result.stderr else 'Unknown'}")
                    # Write full log if needed
                    with open(f"outputs/error_{row['brand']}_{metric}.log", "w") as f:
                        f.write(result.stdout + "\n" + result.stderr)

            except Exception as e:
                print(f" EXCEPTION: {e}")

    print("\nAll case studies completed.")

if __name__ == "__main__":
    main()
