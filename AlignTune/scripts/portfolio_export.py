import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

DEFAULT_INCLUDE = [
    "README.md",
    "MODEL_CARD.md",
    "data/DATA_CARD.md",
    "reports/ablation_long_summary.json",
    "reports/ablation_summary.json",
    "reports/judge_cases.1000.jsonl",
    "reports/judge_winrate.json",
    "reports/RESULTS.md",
    "reports/safety_summary.json",
    "./reports/analysis_outputs/DPO_win_examples.csv",
    "./reports/analysis_outputs/judge_cases.1000.csv",
    "./reports/analysis_outputs/SFT_win_examples.csv",
    "./reports/analysis_outputs/Tie_examples.csv",
    "./reports/judge_analysis.ipynb",
]

def add_path(z: ZipFile, base: Path, p: Path):
    if p.is_dir():
        for sub in p.rglob("*"):
            if sub.is_file():
                z.write(sub, arcname=str(base / sub.relative_to(p)))
    elif p.is_file():
        z.write(p, arcname=str(base / p.name))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./portfolio.zip")
    ap.add_argument(
        "--include",
        nargs="*",
        default=DEFAULT_INCLUDE,
        help="files to include if present (placed at archive root)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for p in args.include:
            path = Path(p)
            if path.exists() and path.is_file():
                add_path(z, Path(""), path)

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
