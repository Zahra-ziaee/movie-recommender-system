from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIGS = {
    "100K": {
        "name": "MovieLens-100K",
        "data_path": str(PROJECT_ROOT / "data" / "processed" / "ratings_100k.csv"),
        "k_neighbors": 25,
        "n_factors": 20,
        "max_epochs": 15,
        "learning_rate": 0.005,
        "reg_lambda": 0.02,
        "test_size": 0.2,
        "random_state": 42,
        "relevance_threshold": 4.0,
        "k_values": [5, 10, 20],
        "cache_size": 200000,
        "baseline_weight": 0.30,
        "cf_weight": 0.35,
        "mf_weight": 0.35,
        "rating_sample_size": 1000,
        "ranking_sample_users": 50,
        "candidate_sample_size": 300,
        "max_train_rows": None,
        "max_test_rows": None,
    },

    "1M": {
        "name": "MovieLens-1M",
        "data_path": str(PROJECT_ROOT / "data" / "processed" / "ratings_1m.csv"),
        "k_neighbors": 20,
        "n_factors": 25,
        "max_epochs": 10,
        "learning_rate": 0.005,
        "reg_lambda": 0.02,
        "test_size": 0.2,
        "random_state": 42,
        "relevance_threshold": 4.0,
        "k_values": [5, 10, 20],
        "cache_size": 300000,
        "baseline_weight": 0.30,
        "cf_weight": 0.30,
        "mf_weight": 0.40,
        "rating_sample_size": 2000,
        "ranking_sample_users": 50,
        "candidate_sample_size": 250,
        "max_train_rows": None,
        "max_test_rows": None,
    },

    "10M": {
        "name": "MovieLens-10M",
        "data_path": str(PROJECT_ROOT / "data" / "processed" / "ratings_10m.csv"),
        "k_neighbors": 10,
        "n_factors": 25,
        "max_epochs": 3,
        "learning_rate": 0.005,
        "reg_lambda": 0.02,
        "test_size": 0.2,
        "random_state": 42,
        "relevance_threshold": 4.0,
        "k_values": [5, 10, 20],
        "cache_size": 300000,
        "baseline_weight": 0.30,
        "cf_weight": 0.20,
        "mf_weight": 0.50,
        "rating_sample_size": 1000,
        "ranking_sample_users": 20,
        "candidate_sample_size": 100,

        # Scalable portfolio mode:
        # The full 10M dataset is loaded and its full statistics are reported,
        # but training/evaluation use controlled samples to keep the run feasible
        # on a local laptop.
        "max_train_rows": 1000000,
        "max_test_rows": 50000,
    },
}