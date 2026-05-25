from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def save_experiment_results(
    dataset_name: str,
    model_name: str,
    rating_results: Dict,
    ranking_results: Dict,
    output_path: str = "results/metrics.csv",
) -> None:
    """
    Save experiment results into a CSV file.

    If the file already exists, the new result will be appended.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "model": model_name,
    }

    for key, value in rating_results.items():
        row[key] = value

    for key, value in ranking_results.items():
        row[key] = value

    new_row = pd.DataFrame([row])

    if output_file.exists():
        existing = pd.read_csv(output_file)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(output_file, index=False)

    print(f"\nExperiment results saved to: {output_file}")