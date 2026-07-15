"""Create privacy-safe error-analysis tables from an existing prediction CSV."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumonia_ai.evaluation.core import (  # noqa: E402
    add_deterministic_uncertainty,
    failure_detection,
    failure_detection_all,
    failure_detection_table,
    validate_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ensemble-predictions")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = add_deterministic_uncertainty(
        validate_predictions(pd.read_csv(args.ensemble_predictions or args.predictions))
    )
    predicted = (frame["probability"] >= args.threshold).astype(int)
    target = frame["binary_target"].astype(int)
    frame["outcome"] = "tn"
    frame.loc[(target == 1) & (predicted == 1), "outcome"] = "tp"
    frame.loc[(target == 0) & (predicted == 1), "outcome"] = "fp"
    frame.loc[(target == 1) & (predicted == 0), "outcome"] = "fn"
    frame.loc[frame.outcome == "fp"].to_csv(output / "false_positives.csv", index=False)
    frame.loc[frame.outcome == "fn"].to_csv(output / "false_negatives.csv", index=False)
    errors = frame.loc[frame.outcome.isin(["fp", "fn"])]
    errors.nlargest(20, "confidence").to_csv(
        output / "highest_confidence_errors.csv", index=False
    )
    frame.nlargest(20, "uncertainty").to_csv(
        output / "highest_uncertainty_cases.csv", index=False
    )
    errors.nsmallest(20, "uncertainty").to_csv(
        output / "lowest_uncertainty_errors.csv", index=False
    )
    frame.groupby("outcome").agg(count=("outcome", "size"), mean_probability=("probability", "mean"), median_probability=("probability", "median"), mean_uncertainty=("uncertainty", "mean"), median_uncertainty=("uncertainty", "median")).reset_index().to_csv(output / "outcome_summary.csv", index=False)
    frame.groupby("outcome")["uncertainty"].agg(["mean", "median", "count"]).reset_index().to_csv(output / "uncertainty_by_outcome.csv", index=False)
    pd.DataFrame([failure_detection(frame, args.threshold)]).to_json(output / "error_metrics.json", orient="records", indent=2)
    analyses = failure_detection_all(frame, args.threshold)
    (output / "uncertainty_failure_detection.json").write_text(
        json.dumps(analyses, indent=2, allow_nan=False)
    )
    failure_detection_table(analyses).to_csv(
        output / "uncertainty_failure_detection.csv", index=False
    )


if __name__ == "__main__":
    main()
