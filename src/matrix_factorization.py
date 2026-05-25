from typing import Dict

import numpy as np
import pandas as pd


class MatrixFactorizationSGD:
    """
    Matrix Factorization model trained with Stochastic Gradient Descent.

    This module learns latent vectors for users and items and predicts ratings as:

        rating_hat = global_mean + user_bias + item_bias + dot(user_factors, item_factors)
    """

    def __init__(
        self,
        n_factors: int = 25,
        learning_rate: float = 0.005,
        reg_lambda: float = 0.02,
        max_epochs: int = 15,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_epochs = max_epochs
        self.random_state = random_state

        self.global_mean = 0.0

        self.user_to_index: Dict[int, int] = {}
        self.item_to_index: Dict[int, int] = {}

        self.user_factors = None
        self.item_factors = None

        self.user_biases = None
        self.item_biases = None

        self.training_history = []

    def fit(self, train_df: pd.DataFrame):
        train_df = train_df.copy()

        unique_users = train_df["userId"].astype(int).unique()
        unique_items = train_df["movieId"].astype(int).unique()

        self.user_to_index = {
            int(user_id): index
            for index, user_id in enumerate(unique_users)
        }

        self.item_to_index = {
            int(item_id): index
            for index, item_id in enumerate(unique_items)
        }

        n_users = len(self.user_to_index)
        n_items = len(self.item_to_index)

        self.global_mean = float(train_df["rating"].mean())

        rng = np.random.default_rng(self.random_state)

        self.user_factors = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(n_users, self.n_factors),
        )

        self.item_factors = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(n_items, self.n_factors),
        )

        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        training_data = train_df[["userId", "movieId", "rating"]].to_numpy(copy=True)

        for epoch in range(1, self.max_epochs + 1):
            rng.shuffle(training_data)

            squared_errors = []

            for user_id, item_id, rating in training_data:
                user_id = int(user_id)
                item_id = int(item_id)
                rating = float(rating)

                user_index = self.user_to_index[user_id]
                item_index = self.item_to_index[item_id]

                prediction = self._predict_by_index(user_index, item_index)
                error = rating - prediction

                squared_errors.append(error ** 2)

                user_vector = self.user_factors[user_index].copy()
                item_vector = self.item_factors[item_index].copy()

                self.user_biases[user_index] += self.learning_rate * (
                    error - self.reg_lambda * self.user_biases[user_index]
                )

                self.item_biases[item_index] += self.learning_rate * (
                    error - self.reg_lambda * self.item_biases[item_index]
                )

                self.user_factors[user_index] += self.learning_rate * (
                    error * item_vector - self.reg_lambda * user_vector
                )

                self.item_factors[item_index] += self.learning_rate * (
                    error * user_vector - self.reg_lambda * item_vector
                )

            epoch_rmse = float(np.sqrt(np.mean(squared_errors)))
            self.training_history.append(
                {
                    "epoch": epoch,
                    "rmse": epoch_rmse,
                }
            )

            print(f"MF epoch {epoch}/{self.max_epochs} - RMSE: {epoch_rmse:.4f}")

        return self

    def _predict_by_index(self, user_index: int, item_index: int) -> float:
        prediction = (
            self.global_mean
            + self.user_biases[user_index]
            + self.item_biases[item_index]
            + np.dot(
                self.user_factors[user_index],
                self.item_factors[item_index],
            )
        )

        return float(np.clip(prediction, 0.5, 5.0))

    def predict(self, user_id: int, item_id: int) -> float:
        user_id = int(user_id)
        item_id = int(item_id)

        if user_id not in self.user_to_index:
            return float(self.global_mean)

        if item_id not in self.item_to_index:
            return float(self.global_mean)

        user_index = self.user_to_index[user_id]
        item_index = self.item_to_index[item_id]

        return self._predict_by_index(user_index, item_index)

    def update_single_interaction(
        self,
        user_id: int,
        item_id: int,
        rating: float,
        n_steps: int = 3,
    ) -> None:
        """
        Incrementally update latent factors for an existing user-item interaction.

        This is a lightweight online update. New users/items are ignored in this
        simple portfolio version and can be handled in future extensions.
        """
        user_id = int(user_id)
        item_id = int(item_id)
        rating = float(rating)

        if user_id not in self.user_to_index or item_id not in self.item_to_index:
            return

        user_index = self.user_to_index[user_id]
        item_index = self.item_to_index[item_id]

        for _ in range(n_steps):
            prediction = self._predict_by_index(user_index, item_index)
            error = rating - prediction

            user_vector = self.user_factors[user_index].copy()
            item_vector = self.item_factors[item_index].copy()

            self.user_biases[user_index] += self.learning_rate * (
                error - self.reg_lambda * self.user_biases[user_index]
            )

            self.item_biases[item_index] += self.learning_rate * (
                error - self.reg_lambda * self.item_biases[item_index]
            )

            self.user_factors[user_index] += self.learning_rate * (
                error * item_vector - self.reg_lambda * user_vector
            )

            self.item_factors[item_index] += self.learning_rate * (
                error * user_vector - self.reg_lambda * item_vector
            )