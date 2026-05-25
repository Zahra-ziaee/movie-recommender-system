from collections import OrderedDict
from typing import Dict, Set

import numpy as np


class CASMSimilarityEngine:
    """
    Confidence-Aware Similarity Measure (CASM).

    This similarity measure combines:
    - Pearson correlation
    - Support confidence
    - Jaccard overlap
    - User expertise weight
    """

    def __init__(
        self,
        shrinkage_lambda: float = 8.0,
        confidence_alpha: float = 0.65,
        expertise_beta: float = 0.35,
        min_common_items: int = 3,
        cache_size: int = 200000,
    ):
        self.shrinkage_lambda = shrinkage_lambda
        self.confidence_alpha = confidence_alpha
        self.expertise_beta = expertise_beta
        self.min_common_items = min_common_items
        self.cache_size = cache_size
        self.similarity_cache = OrderedDict()

    def _cache_get(self, key):
        if key in self.similarity_cache:
            value = self.similarity_cache.pop(key)
            self.similarity_cache[key] = value
            return value
        return None

    def _cache_set(self, key, value):
        self.similarity_cache[key] = value

        if len(self.similarity_cache) > self.cache_size:
            self.similarity_cache.popitem(last=False)

    @staticmethod
    def pearson_similarity(
        ratings_u: Dict[int, float],
        ratings_v: Dict[int, float],
        common_items: Set[int],
    ) -> float:
        if len(common_items) < 2:
            return 0.0

        u_values = np.array([ratings_u[item] for item in common_items], dtype=np.float32)
        v_values = np.array([ratings_v[item] for item in common_items], dtype=np.float32)

        u_centered = u_values - u_values.mean()
        v_centered = v_values - v_values.mean()

        denominator = np.linalg.norm(u_centered) * np.linalg.norm(v_centered)

        if denominator == 0:
            return 0.0

        return float(np.dot(u_centered, v_centered) / denominator)

    def compute_similarity(
        self,
        user_u: int,
        user_v: int,
        user_ratings: Dict[int, Dict[int, float]],
    ) -> float:
        key = tuple(sorted((user_u, user_v)))

        cached_value = self._cache_get(key)
        if cached_value is not None:
            return cached_value

        ratings_u = user_ratings.get(user_u, {})
        ratings_v = user_ratings.get(user_v, {})

        items_u = set(ratings_u.keys())
        items_v = set(ratings_v.keys())

        common_items = items_u & items_v

        if len(common_items) < self.min_common_items:
            self._cache_set(key, 0.0)
            return 0.0

        pearson = self.pearson_similarity(ratings_u, ratings_v, common_items)

        support = len(common_items)
        support_confidence = support / (support + self.shrinkage_lambda)

        union_size = len(items_u | items_v)
        jaccard = support / union_size if union_size > 0 else 0.0
        jaccard_component = jaccard ** self.confidence_alpha

        expertise_u = min(1.0, len(items_u) / 100.0)
        expertise_v = min(1.0, len(items_v) / 100.0)
        expertise_weight = (expertise_u * expertise_v) ** self.expertise_beta

        casm_similarity = (
            pearson
            * support_confidence
            * jaccard_component
            * expertise_weight
        )

        self._cache_set(key, casm_similarity)

        return float(casm_similarity)

    def invalidate_user_cache(self, user_id: int) -> None:
        keys_to_delete = [
            key for key in self.similarity_cache.keys()
            if user_id in key
        ]

        for key in keys_to_delete:
            del self.similarity_cache[key]