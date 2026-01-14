import argparse
import csv
import json
from pathlib import Path

from eval import eval_score


def _parse_run(arg: str) -> tuple[str, str]:
    if "=" not in arg:
        raise ValueError("Run must be formatted as label=path")
    label, path = arg.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("Run must include non-empty label and path")
    return label, path


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = ["label"] + [k for k in rows[0].keys() if k != "label"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation metrics across runs.")
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--run", action="append", required=True, help="label=path to jsonl")
    parser.add_argument("--output_dir", type=str, default="evaluation/reports")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    details_map = {}
    for run_arg in args.run:
        label, path = _parse_run(run_arg)
        scores, detail = eval_score(args.set_type, file_path=path)
        row = {"label": label}
        row.update(scores)
        summaries.append(row)
        details_map[label] = detail

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _write_csv(summary_csv, summaries)

    if args.details:
        for label, detail in details_map.items():
            detail_path = output_dir / f"details_{_safe_label(label)}.json"
            detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib is not available; skipping plots.")
            return
        keys = [
            "Delivery Rate",
            "Commonsense Constraint Micro Pass Rate",
            "Commonsense Constraint Macro Pass Rate",
            "Hard Constraint Micro Pass Rate",
            "Hard Constraint Macro Pass Rate",
            "Final Pass Rate",
        ]
        labels = [row["label"] for row in summaries]
        for key in keys:
            values = [row.get(key, 0) for row in summaries]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, values)
            ax.set_title(key)
            ax.set_ylabel("Rate")
            ax.set_ylim(0, 1.0)
            fig.tight_layout()
            fig.savefig(output_dir / f"plot_{_safe_label(key)}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
