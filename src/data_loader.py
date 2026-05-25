from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "user_id": "userId",
        "item_id": "movieId",
        "movie_id": "movieId",
    }

    df = df.rename(columns=rename_map)

    required_columns = {"userId", "movieId", "rating"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def compute_dataset_statistics(df: pd.DataFrame) -> Dict:
    n_ratings = len(df)
    n_users = df["userId"].nunique()
    n_items = df["movieId"].nunique()

    density = n_ratings / (n_users * n_items)
    sparsity = 1 - density

    return {
        "total_ratings": n_ratings,
        "total_users": n_users,
        "total_items": n_items,
        "density": density,
        "sparsity": sparsity,
    }


def sample_if_needed(
    df: pd.DataFrame,
    max_rows,
    random_state: int,
) -> pd.DataFrame:
    if max_rows is None:
        return df

    if len(df) <= max_rows:
        return df

    return df.sample(
        n=max_rows,
        random_state=random_state,
    ).reset_index(drop=True)


def load_thesis_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    data_path = Path(config["data_path"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    ratings = pd.read_csv(data_path)
    ratings = standardize_columns(ratings)

    dataset_info = compute_dataset_statistics(ratings)

    train_df, test_df = train_test_split(
        ratings,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        shuffle=True,
    )

    train_df = sample_if_needed(
        train_df,
        max_rows=config.get("max_train_rows"),
        random_state=config.get("random_state", 42),
    )

    test_df = sample_if_needed(
        test_df,
        max_rows=config.get("max_test_rows"),
        random_state=config.get("random_state", 42),
    )

    dataset_info["train_rows_used"] = len(train_df)
    dataset_info["test_rows_used"] = len(test_df)

    return train_df, test_df, dataset_info