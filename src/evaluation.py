from typing import Dict, List, Set

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def precision_at_k(
    recommended_items: List[int],
    relevant_items: Set[int],
    k: int,
) -> float:
    recommended_k = recommended_items[:k]

    if k == 0:
        return 0.0

    hits = sum(1 for item in recommended_k if item in relevant_items)

    return hits / k


def recall_at_k(
    recommended_items: List[int],
    relevant_items: Set[int],
    k: int,
) -> float:
    if not relevant_items:
        return 0.0

    recommended_k = recommended_items[:k]
    hits = sum(1 for item in recommended_k if item in relevant_items)

    return hits / len(relevant_items)


def ndcg_at_k(
    recommended_items: List[int],
    relevant_items: Set[int],
    k: int,
) -> float:
    recommended_k = recommended_items[:k]

    dcg = 0.0

    for index, item_id in enumerate(recommended_k):
        if item_id in relevant_items:
            dcg += 1 / np.log2(index + 2)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1 / np.log2(index + 2) for index in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def mrr_at_k(
    recommended_items: List[int],
    relevant_items: Set[int],
    k: int,
) -> float:
    recommended_k = recommended_items[:k]

    for index, item_id in enumerate(recommended_k):
        if item_id in relevant_items:
            return 1 / (index + 1)

    return 0.0


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


class OptimizedLightningEvaluator:
    """
    Fast evaluator for rating prediction and ranking quality.

    Rating metrics:
    - RMSE
    - MAE

    Ranking metrics:
    - Precision@K
    - Recall@K
    - NDCG@K
    - MRR@K
    - F1@K
    """

    def __init__(self, k_values=None, relevance_threshold: float = 4.0):
        self.k_values = k_values or [5, 10, 20]
        self.relevance_threshold = relevance_threshold

    def rating_evaluation(
        self,
        recommender,
        test_df,
        sample_size: int = 1000,
    ) -> Dict:
        sample_df = test_df.sample(
            n=min(sample_size, len(test_df)),
            random_state=42,
        )

        predictions = []
        actuals = []

        for row in sample_df[["userId", "movieId", "rating"]].itertuples(index=False):
            user_id = int(row.userId)
            item_id = int(row.movieId)
            actual_rating = float(row.rating)

            predicted_rating = recommender.predict_rating(user_id, item_id)

            actuals.append(actual_rating)
            predictions.append(predicted_rating)

        return {
            "rmse": rmse(actuals, predictions),
            "mae": mae(actuals, predictions),
            "sample_size": len(sample_df),
        }

    def _build_candidate_items(
        self,
        recommender,
        user_id: int,
        relevant_items: Set[int],
        candidate_sample_size: int,
    ) -> List[int]:
        """
        Build candidate set using relevant test items plus sampled negative items.

        This prevents ranking evaluation from becoming all-zero just because
        relevant items were not sampled as candidates.
        """
        seen_items = set(recommender.user_ratings.get(user_id, {}).keys())
        all_items = list(recommender.item_ratings.keys())

        valid_relevant_items = [
            int(item_id)
            for item_id in relevant_items
            if int(item_id) not in seen_items
        ]

        negative_candidates = [
            int(item_id)
            for item_id in all_items
            if int(item_id) not in seen_items
            and int(item_id) not in relevant_items
        ]

        rng = np.random.default_rng(seed=42 + int(user_id))

        negative_sample_size = min(
            candidate_sample_size,
            len(negative_candidates),
        )

        if negative_sample_size > 0:
            sampled_negatives = rng.choice(
                negative_candidates,
                size=negative_sample_size,
                replace=False,
            ).astype(int).tolist()
        else:
            sampled_negatives = []

        candidate_items = valid_relevant_items + sampled_negatives

        return list(dict.fromkeys(candidate_items))

    def ranking_evaluation(
        self,
        recommender,
        test_df,
        sample_users: int = 50,
        candidate_sample_size: int = 300,
    ) -> Dict:
        """
        Evaluate recommendation ranking quality.

        Relevant items are defined as:
        rating >= relevance_threshold

        Candidate set:
        relevant test items + sampled negative items

        Important:
        In large sampled datasets, a user may exist in test but not in the
        sampled training set. Those users are skipped because the recommender
        cannot generate personalized recommendations for unseen users.
        """
        relevant_test = test_df[test_df["rating"] >= self.relevance_threshold]

        users_in_model = set(recommender.user_ratings.keys())

        eligible_users = []

        for user_id, group in relevant_test.groupby("userId"):
            user_id = int(user_id)

            if user_id not in users_in_model:
                continue

            seen_items = set(recommender.user_ratings.get(user_id, {}).keys())

            relevant_items = set(group["movieId"].astype(int).tolist())
            relevant_items = {
                item_id for item_id in relevant_items
                if item_id not in seen_items
            }

            if relevant_items:
                eligible_users.append(user_id)

        if len(eligible_users) == 0:
            return {
                "precision@5": 0.0,
                "recall@5": 0.0,
                "ndcg@5": 0.0,
                "mrr@5": 0.0,
                "f1@5": 0.0,
                "precision@10": 0.0,
                "recall@10": 0.0,
                "ndcg@10": 0.0,
                "mrr@10": 0.0,
                "f1@10": 0.0,
                "precision@20": 0.0,
                "recall@20": 0.0,
                "ndcg@20": 0.0,
                "mrr@20": 0.0,
                "f1@20": 0.0,
                "evaluated_users": 0,
                "eligible_users": 0,
            }

        rng = np.random.default_rng(seed=42)
        selected_users = rng.choice(
            eligible_users,
            size=min(sample_users, len(eligible_users)),
            replace=False,
        )

        results = {}

        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            mrr_scores = []
            f1_scores = []

            for user_id in selected_users:
                user_test = test_df[test_df["userId"] == user_id]

                seen_items = set(recommender.user_ratings.get(int(user_id), {}).keys())

                relevant_items = set(
                    user_test[
                        user_test["rating"] >= self.relevance_threshold
                    ]["movieId"].astype(int).tolist()
                )

                relevant_items = {
                    item_id for item_id in relevant_items
                    if item_id not in seen_items
                }

                if not relevant_items:
                    continue

                candidate_items = self._build_candidate_items(
                    recommender=recommender,
                    user_id=int(user_id),
                    relevant_items=relevant_items,
                    candidate_sample_size=candidate_sample_size,
                )

                if not candidate_items:
                    continue

                recommended_items = recommender.recommend_items(
                    int(user_id),
                    n_items=k,
                    candidate_items=candidate_items,
                )

                precision = precision_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                )
                recall = recall_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                )
                ndcg = ndcg_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                )
                mrr = mrr_at_k(
                    recommended_items,
                    relevant_items,
                    k,
                )
                f1 = f1_at_k(precision, recall)

                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)
                f1_scores.append(f1)

            results[f"precision@{k}"] = (
                float(np.mean(precision_scores)) if precision_scores else 0.0
            )
            results[f"recall@{k}"] = (
                float(np.mean(recall_scores)) if recall_scores else 0.0
            )
            results[f"ndcg@{k}"] = (
                float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
            )
            results[f"mrr@{k}"] = (
                float(np.mean(mrr_scores)) if mrr_scores else 0.0
            )
            results[f"f1@{k}"] = (
                float(np.mean(f1_scores)) if f1_scores else 0.0
            )

        results["evaluated_users"] = len(selected_users)
        results["eligible_users"] = len(eligible_users)

        return results