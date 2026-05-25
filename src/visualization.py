from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def save_metric_bar_chart(
    metrics: Dict,
    output_path: str,
    title: str,
    ylabel: str,
) -> None:
    """
    Save a simple bar chart for selected metrics.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Chart saved to: {output_file}")


def save_experiment_charts(
    rating_results: Dict,
    ranking_results: Dict,
    output_dir: str = "results/figures",
) -> None:
    """
    Save rating and ranking metric charts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rating_metrics = {
        "RMSE": rating_results["rmse"],
        "MAE": rating_results["mae"],
    }

    save_metric_bar_chart(
        metrics=rating_metrics,
        output_path=str(output_path / "rating_metrics.png"),
        title="Rating Prediction Metrics",
        ylabel="Error",
    )

    ranking_metrics = {
        "Precision@10": ranking_results.get("precision@10", 0.0),
        "Recall@10": ranking_results.get("recall@10", 0.0),
        "NDCG@10": ranking_results.get("ndcg@10", 0.0),
        "MRR@10": ranking_results.get("mrr@10", 0.0),
        "F1@10": ranking_results.get("f1@10", 0.0),
    }

    save_metric_bar_chart(
        metrics=ranking_metrics,
        output_path=str(output_path / "ranking_metrics_at_10.png"),
        title="Ranking Metrics at K=10",
        ylabel="Score",
    )