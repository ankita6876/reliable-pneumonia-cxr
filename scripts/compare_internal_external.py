"""Compare frozen CheXpert internal-test and RSNA external metrics."""
# ruff: noqa: E501, E701, E702
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-metrics", required=True); parser.add_argument("--external-metrics", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    internal, external = json.loads(Path(args.internal_metrics).read_text()), json.loads(Path(args.external_metrics).read_text())
    if "test" in internal: internal = internal["test"]
    comparison = pd.DataFrame([{"cohort": "CheXpert internal test", **internal}, {"cohort": "RSNA external test", **external}])
    comparison.to_csv(output / "internal_external_comparison.csv", index=False)
    for columns, filename, ylabel in ((["auroc", "auprc", "accuracy", "balanced_accuracy"], "internal_external_performance.png", "Performance"), (["brier_score", "expected_calibration_error", "negative_log_likelihood"], "internal_external_calibration.png", "Calibration error")):
        usable = [column for column in columns if column in comparison]
        figure, axis = plt.subplots(figsize=(8, 4)); comparison.set_index("cohort")[usable].T.plot.bar(ax=axis); axis.set_ylabel(ylabel); axis.set_title("CheXpert internal test vs RSNA external test"); figure.tight_layout(); figure.savefig(output / filename, dpi=300); plt.close(figure)


if __name__ == "__main__":
    main()
