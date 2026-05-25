import argparse

from src.config import DATASET_CONFIGS
from src.data_loader import load_thesis_data
from src.evaluation import OptimizedLightningEvaluator
from src.recommender import CompleteThesisRecommender
from src.utils import save_experiment_results
from src.visualization import save_experiment_charts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the hybrid CASM-CF + Matrix Factorization recommender."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="100K",
        choices=["100K", "1M", "10M"],
        help="Dataset version to use: 100K, 1M, or 10M.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = DATASET_CONFIGS[args.dataset]
    model_name = "Hybrid CASM-CF + Matrix Factorization"

    print("=" * 60)
    print("Movie Recommender System - Hybrid Thesis Portfolio Version")
    print("=" * 60)
    print(f"Selected dataset: {config['name']}")

    train_df, test_df, dataset_info = load_thesis_data(config)

    print("\nDataset loaded successfully.")
    print("\nDataset info:")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    print("\nTraining hybrid recommender model...")
    recommender = CompleteThesisRecommender(config)
    recommender.fit(train_df)
    print("Hybrid recommender training finished.")

    evaluator = OptimizedLightningEvaluator(
        k_values=config["k_values"],
        relevance_threshold=config["relevance_threshold"],
    )

    print("\nEvaluating rating prediction...")
    rating_results = evaluator.rating_evaluation(
        recommender,
        test_df,
        sample_size=config["rating_sample_size"],
    )

    print("\nRating prediction results:")
    for key, value in rating_results.items():
        print(f"{key}: {value}")

    print("\nEvaluating ranking quality...")
    ranking_results = evaluator.ranking_evaluation(
        recommender,
        test_df,
        sample_users=config["ranking_sample_users"],
        candidate_sample_size=config["candidate_sample_size"],
    )

    print("\nRanking evaluation results:")
    for key, value in ranking_results.items():
        print(f"{key}: {value}")

    print("\nTesting incremental update...")
    recommender.add_rating_incremental(
        user_id=1,
        item_id=50,
        rating=5.0,
    )
    print("Incremental update test completed successfully.")

    save_experiment_results(
        dataset_name=config["name"],
        model_name=model_name,
        rating_results=rating_results,
        ranking_results=ranking_results,
        output_path="results/metrics.csv",
    )

    save_experiment_charts(
        rating_results=rating_results,
        ranking_results=ranking_results,
        output_dir="results/figures",
    )


if __name__ == "__main__":
    main()