from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_movielens_100k() -> None:
    input_path = RAW_DIR / "ml-100k" / "u.data"
    output_path = PROCESSED_DIR / "ratings_100k.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Expected file not found: {input_path}")

    ratings = pd.read_csv(
        input_path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output_path, index=False)

    print("MovieLens 100K prepared successfully.")
    print(f"Saved to: {output_path}")
    print(ratings.head())


def prepare_movielens_1m() -> None:
    input_path = RAW_DIR / "ml-1m" / "ratings.dat"
    output_path = PROCESSED_DIR / "ratings_1m.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Expected file not found: {input_path}")

    ratings = pd.read_csv(
        input_path,
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output_path, index=False)

    print("MovieLens 1M prepared successfully.")
    print(f"Saved to: {output_path}")
    print(ratings.head())


def prepare_movielens_10m() -> None:
    possible_paths = [
        RAW_DIR / "ml-10m" / "ratings.dat",
        RAW_DIR / "ml-10M100K" / "ratings.dat",
    ]

    input_path = None
    for path in possible_paths:
        if path.exists():
            input_path = path
            break

    if input_path is None:
        raise FileNotFoundError(
            "MovieLens 10M ratings.dat not found. Expected one of:\n"
            + "\n".join(str(path) for path in possible_paths)
        )

    output_path = PROCESSED_DIR / "ratings_10m.csv"

    ratings = pd.read_csv(
        input_path,
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output_path, index=False)

    print("MovieLens 10M prepared successfully.")
    print(f"Saved to: {output_path}")
    print(ratings.head())


if __name__ == "__main__":
    prepare_movielens_100k()
    prepare_movielens_1m()
    prepare_movielens_10m()