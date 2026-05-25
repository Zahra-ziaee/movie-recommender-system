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
    Fast evaluator for rating prediction.
    Ranking evaluation will be added in the next step.
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

        for _, row in sample_df.iterrows():
            user_id = int(row["userId"])
            item_id = int(row["movieId"])
            actual_rating = float(row["rating"])

            predicted_rating = recommender.predict_rating(user_id, item_id)

            actuals.append(actual_rating)
            predictions.append(predicted_rating)

        return {
            "rmse": rmse(actuals, predictions),
            "mae": mae(actuals, predictions),
            "sample_size": len(sample_df),
        }