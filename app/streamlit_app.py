import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATASET_CONFIGS
from src.data_loader import load_thesis_data
from src.recommender import CompleteThesisRecommender


@st.cache_resource
def train_model():
    config = DATASET_CONFIGS["100K"]

    train_df, test_df, dataset_info = load_thesis_data(config)

    recommender = CompleteThesisRecommender(config)
    recommender.fit(train_df)

    return recommender, dataset_info


def main():
    st.set_page_config(
        page_title="Movie Recommender System",
        layout="wide",
    )

    st.title("🎬 Movie Recommender System")
    st.write(
        "A thesis-based hybrid recommender system using "
        "CASM Collaborative Filtering and Matrix Factorization."
    )

    recommender, dataset_info = train_model()

    st.subheader("Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Ratings", f"{dataset_info['total_ratings']:,}")
    col2.metric("Users", f"{dataset_info['total_users']:,}")
    col3.metric("Items", f"{dataset_info['total_items']:,}")
    col4.metric("Sparsity", f"{dataset_info['sparsity'] * 100:.2f}%")

    st.subheader("Get Recommendations")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        max_value=int(dataset_info["total_users"]),
        value=1,
        step=1,
    )

    n_items = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
    )

    if st.button("Generate Recommendations"):
        recommendations = recommender.recommend_items(
            user_id=int(user_id),
            n_items=int(n_items),
            candidate_sample_size=300,
        )

        if recommendations:
            st.success(f"Top {n_items} recommended movie IDs for user {user_id}:")

            for rank, item_id in enumerate(recommendations, start=1):
                st.write(f"{rank}. Movie ID: {item_id}")
        else:
            st.warning("No recommendations found for this user.")

    st.subheader("Project Components")

    st.markdown(
        """
        - Confidence-Aware Similarity Measure (CASM)
        - Collaborative Filtering
        - Matrix Factorization with SGD
        - Hybrid CF + MF Prediction
        - Incremental Update Logic
        - RMSE / MAE / Ranking Metrics Evaluation
        """
    )


if __name__ == "__main__":
    main()

    