import os
import json
import argparse
from pathlib import Path

def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def ascii_bar(x, width=24):
    x = max(0.0, min(1.0, float(x)))
    n = int(round(x * width))
    return "█" * n + " " * (width - n)

def maybe_plot(rows, out_png):
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ablation",
        default="reports/ablation_summary.json",
        help="input JSON from run_ablation.py",
    )
    ap.add_argument(
        "--ablation_long",
        default=None,
        help="optional input JSON for long-prompt subset ablation",
    )
    ap.add_argument(
        "--eval", default="reports/eval_report.json", help="optional eval JSON for one model"
    )
    ap.add_argument("--out_md", default="reports/RESULTS.md")
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--judge", default="reports/judge_winrate.json")
    ap.add_argument("--safety", default="reports/safety_summary.json")
    args = ap.parse_args()

    abl = load_json(args.ablation) or {"runs": []}
    abl_long = load_json(args.ablation_long) if args.ablation_long else None
    runs = abl.get("runs", [])
    eval_single = load_json(args.eval)
    judge = load_json(args.judge)
    safety = load_json(args.safety)

    lines = []
    lines.append("# Results Summary")
    lines.append("")

    if runs:
        lines.append("## Ablation (Balanced‑Prompt Subset)")
        lines.append("")
        lines.append("| Model | N | Exact Match | ROUGE-L | Toxic Flags |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in runs:
            lines.append(
                f"| {r['label']} | {r.get('n', 0)} | {r.get('exact_match', 0):.3f} | "
                f"{r.get('rougeL', 0):.3f} | {r.get('toxic_flags', 0)} |"
            )

    if eval_single:
        lines.append("## Eval (detailed)")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(eval_single, indent=2))
        lines.append("```")

    if abl_long and abl_long.get("runs"):
        lines.append("")
        lines.append("## Ablation (Long‑Prompt Subset)")
        lines.append("")
        lines.append("| Model | N | Exact Match | ROUGE-L | Toxic Flags |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in abl_long.get("runs", []):
            lines.append(
                f"| {r['label']} | {r.get('n', 0)} | {r.get('exact_match', 0):.3f} | "
                f"{r.get('rougeL', 0):.3f} | {r.get('toxic_flags', 0)} |"
            )

    if judge:
        lines.append("")
        lines.append("## Judge Win-Rate")
        lines.append("")
        a, b = judge.get("model_a"), judge.get("model_b")
        wr_a = judge.get("win_rate_a", 0.0)
        ci_la, ci_ha = judge.get("ci_low_a", judge.get("ci_low", 0.0)), judge.get("ci_high_a", judge.get("ci_high", 0.0))
        wr_b = judge.get("win_rate_b", 1.0 - wr_a)
        ci_lb, ci_hb = judge.get("ci_low_b", 0.0), judge.get("ci_high_b", 0.0)
        ties = judge.get("ties", 0)
        N = judge.get("n", 0)
        symmetric = bool(judge.get("symmetric", False))
        lines.append(
            f"{a} vs {b}: {wr_a*100:.1f}% win for {a} (95% CI: {ci_la*100:.1f}–{ci_ha*100:.1f}%)."
        )
        lines.append("")
        lines.append(
            f"{b} vs {a}: {wr_b*100:.1f}% win for {b} (95% CI: {ci_lb*100:.1f}–{ci_hb*100:.1f}%)."
        )
        lines.append("")
        if symmetric:
            lines.append(f"Ties (pooled across both directions): {ties}/{2*N}.")
        else:
            lines.append(f"Ties: {ties}/{N}.")

    if safety:
        lines.append("")
        lines.append("## Safety Quick-Check")
        lines.append("")
        a, b = safety.get("model_a"), safety.get("model_b")
        ra, rb = safety.get("a", {}), safety.get("b", {})
        lines.append("| Model | Attacks | Harmful Flags | Refusals | Attack Success Rate |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| {a} | {ra.get('n_attacks', 0)} | {ra.get('harmful_flags', 0)} | "
            f"{ra.get('refusals', 0)} | {ra.get('attack_success_rate', 0):.2f} |"
        )
        lines.append(
            f"| {b} | {rb.get('n_attacks', 0)} | {rb.get('harmful_flags', 0)} | "
            f"{rb.get('refusals', 0)} | {rb.get('attack_success_rate', 0):.2f} |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out_md}")

if __name__ == "__main__":
    main()
