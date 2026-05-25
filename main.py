from src.config import DATASET_CONFIGS
from src.data_loader import load_thesis_data
from src.evaluation import OptimizedLightningEvaluator
from src.recommender import CompleteThesisRecommender


def main():
    config = DATASET_CONFIGS["100K"]

    print("=" * 60)
    print("Movie Recommender System - Hybrid Thesis Portfolio Version")
    print("=" * 60)

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
        sample_size=1000,
    )

    print("\nRating prediction results:")
    for key, value in rating_results.items():
        print(f"{key}: {value}")

    print("\nEvaluating ranking quality...")
    ranking_results = evaluator.ranking_evaluation(
        recommender,
        test_df,
        sample_users=50,
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


if __name__ == "__main__":
    main()