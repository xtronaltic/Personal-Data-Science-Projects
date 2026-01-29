
import pandas as pd
import subprocess
from pathlib import Path
import sys

def main():
    print("Running Case Studies for all eligible brands...")
    
    # Load eligible list
    try:
        df = pd.read_csv('outputs/eligible_brands.csv')
    except FileNotFoundError:
        print("Error: outputs/eligible_brands.csv not found.")
        sys.exit(1)
        
    print(f"Found {len(df)} series to process.")
    
    success_count = 0
    fail_count = 0
    
    # Ensure output dir exists
    output_dir = Path("outputs/Production Readiness Report")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, row in df.iterrows():
        print(f"\n[{i+1}/{len(df)}] Processing: {row['brand']} ({row['markets']})")
        
        cmd = [
            sys.executable, "Case_Study.py",
            "--channel", str(row['markets']),
            "--manufacturer", str(row['manufacturer']),
            "--category", str(row['category']),
            "--trademark", str(row['trademark']),
            "--brand", str(row['brand'])
        ]
        
        try:
            # Run and capture output to avoid clutter, print only if error
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  -> Success")
                # Extract WMAPE from output if possible
                for line in result.stdout.split('\n'):
                    if "WMAPE:" in line:
                        print(f"     {line.strip()}")
                success_count += 1
            else:
                print(f"  -> Failed (Code {result.returncode})")
                print(f"     Error: {result.stderr[:200]}...") # Print first 200 chars of error
                fail_count += 1
                
        except Exception as e:
            print(f"  -> Exception: {e}")
            fail_count += 1
            
    print("\n" + "="*60)
    print(f"Batch Processing Complete.")
    print(f"Successful: {success_count}")
    print(f"Failed:     {fail_count}")
    print("="*60)

if __name__ == "__main__":
    main()
