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
        "reg_lambda": 0.01,
        "test_size": 0.2,
        "random_state": 42,
        "relevance_threshold": 4.0,
        "k_values": [5, 10, 20],
    },
    "1M": {
        "name": "MovieLens-1M",
        "data_path": str(PROJECT_ROOT / "data" / "processed" / "ratings_1m.csv"),
        "k_neighbors": 25,
        "n_factors": 25,
        "max_epochs": 15,
        "learning_rate": 0.005,
        "reg_lambda": 0.02,
        "test_size": 0.2,
        "random_state": 42,
        "relevance_threshold": 4.0,
        "k_values": [5, 10, 20],
    },
    "10M": {
        "name": "MovieLens-10M",
        "data_path": str(PROJECT_ROOT / "data" / "processed" / "ratings_10m.csv"),
        "k_neighbors": 30,
        "n_factors": 25,
        "max_epochs": 15,
        "learning_rate": 0.005,
        "reg_lambda": 0.02,
        "test_size": 0.2,
        "random_state": 42,
        "relevance_threshold": 4.0,
        "k_values": [5, 10, 20],
        "cache_size": 200000,
    },
}