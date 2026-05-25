from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.similarity import CASMSimilarityEngine


class CompleteThesisRecommender:
    """
    Thesis-based recommender system.

    Main components:
    - CASM-based collaborative filtering
    - Baseline bias prediction
    - Incremental cache invalidation
    - Top-N recommendation generation
    """

    def __init__(self, config: Dict):
        self.config = config
        self.k_neighbors = config.get("k_neighbors", 25)

        self.global_mean = 0.0
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)

        self.user_biases = {}
        self.item_biases = {}

        self.casm_engine = CASMSimilarityEngine(
            cache_size=config.get("cache_size", 200000)
        )

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())

        for _, row in train_df.iterrows():
            user_id = int(row["userId"])
            item_id = int(row["movieId"])
            rating = float(row["rating"])

            self.user_ratings[user_id][item_id] = rating
            self.item_ratings[item_id][user_id] = rating

        self._compute_biases(train_df)

        return self

    def _compute_biases(self, train_df: pd.DataFrame) -> None:
        user_means = train_df.groupby("userId")["rating"].mean()
        item_means = train_df.groupby("movieId")["rating"].mean()

        self.user_biases = {
            int(user_id): float(user_mean - self.global_mean)
            for user_id, user_mean in user_means.items()
        }

        self.item_biases = {
            int(item_id): float(item_mean - self.global_mean)
            for item_id, item_mean in item_means.items()
        }

    def find_similar_users(self, user_id: int) -> List[Tuple[int, float]]:
        if user_id not in self.user_ratings:
            return []

        target_items = set(self.user_ratings[user_id].keys())
        candidate_users = set()

        for item_id in target_items:
            candidate_users.update(self.item_ratings[item_id].keys())

        candidate_users.discard(user_id)

        similarities = []

        for other_user in candidate_users:
            similarity = self.casm_engine.compute_similarity(
                user_id,
                other_user,
                self.user_ratings,
            )

            if similarity > 0:
                similarities.append((other_user, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:self.k_neighbors]

    def predict_rating(self, user_id: int, item_id: int) -> float:
        baseline_prediction = (
            self.global_mean
            + self.user_biases.get(user_id, 0.0)
            + self.item_biases.get(item_id, 0.0)
        )

        similar_users = self.find_similar_users(user_id)

        weighted_sum = 0.0
        weight_sum = 0.0

        for other_user, similarity in similar_users:
            if item_id in self.user_ratings[other_user]:
                rating = self.user_ratings[other_user][item_id]
                weighted_sum += similarity * rating
                weight_sum += abs(similarity)

        if weight_sum > 0:
            cf_prediction = weighted_sum / weight_sum
            final_prediction = 0.6 * baseline_prediction + 0.4 * cf_prediction
        else:
            final_prediction = baseline_prediction

        return float(np.clip(final_prediction, 0.5, 5.0))

    def recommend_items(
        self,
        user_id: int,
        n_items: int = 10,
        candidate_sample_size: int = 300,
    ) -> List[int]:
        """
        Generate top-N item recommendations for a user.

        To keep evaluation efficient, this method scores a sample of unseen items
        instead of scoring the entire item catalog.
        """
        if user_id not in self.user_ratings:
            return []

        seen_items = set(self.user_ratings[user_id].keys())
        all_items = list(self.item_ratings.keys())

        candidate_items = [
            item_id for item_id in all_items
            if item_id not in seen_items
        ]

        if not candidate_items:
            return []

        if len(candidate_items) > candidate_sample_size:
            rng = np.random.default_rng(seed=42)
            candidate_items = rng.choice(
                candidate_items,
                size=candidate_sample_size,
                replace=False,
            ).tolist()

        scored_items = []

        for item_id in candidate_items:
            score = self.predict_rating(user_id, int(item_id))
            scored_items.append((int(item_id), score))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        return [item_id for item_id, _ in scored_items[:n_items]]

    def add_rating_incremental(self, user_id: int, item_id: int, rating: float) -> None:
        """
        Incremental update mechanism.

        Only the affected user-item structures and related similarity cache
        entries are updated.
        """
        self.user_ratings[user_id][item_id] = rating
        self.item_ratings[item_id][user_id] = rating

        self.casm_engine.invalidate_user_cache(user_id)