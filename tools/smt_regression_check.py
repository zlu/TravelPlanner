import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from agents.smt_runner import run_smt


def analyze_range(root: Path, set_type: str, start_idx: int, end_idx: int):
    results = []
    for idx in range(start_idx, end_idx + 1):
        plan_path = root / set_type / "gpt_nl" / str(idx) / "plans" / "plan.txt"
        error_path = root / set_type / "gpt_nl" / str(idx) / "plans" / "error.txt"
        status = "missing"
        error = None
        if plan_path.exists():
            plan_text = plan_path.read_text().strip()
            if plan_text:
                status = "ok"
            else:
                status = "empty"
        if error_path.exists():
            err_text = error_path.read_text().strip()
            if err_text:
                status = "error"
                error = err_text
        results.append(
            {
                "index": idx,
                "status": status,
                "plan_path": str(plan_path),
                "error": error,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Check SMT solver regression around a range of indices.")
    parser.add_argument("--set_type", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--start_idx", type=int, default=90)
    parser.add_argument("--end_idx", type=int, default=110)
    parser.add_argument("--input_root", default="smt_output", help="Existing SMT output root to analyze.")
    parser.add_argument(
        "--output_root",
        default="smt_token_output",
        help="Output root to use when running missing items.",
    )
    parser.add_argument(
        "--run_missing",
        action="store_true",
        help="Run SMT for the range and write outputs under output_root.",
    )
    parser.add_argument("--report_dir", default="smt_token_output")
    args = parser.parse_args()

    if args.run_missing:
        max_items = args.end_idx - args.start_idx + 1
        run_smt(
            set_type=args.set_type,
            model_name="deepseek:deepseek-chat",
            max_items=max_items,
            smt_repo=None,
            dataset_path=None,
            output_root=Path(args.output_root),
            skip_existing=False,
            start_idx=args.start_idx,
        )

    analysis_root = Path(args.output_root if args.run_missing else args.input_root)
    results = analyze_range(analysis_root, args.set_type, args.start_idx, args.end_idx)
    summary = {
        "set_type": args.set_type,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "root": str(analysis_root),
        "results": results,
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "smt_regression_report.json"
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote SMT regression report to {report_path}")


if __name__ == "__main__":
    main()
