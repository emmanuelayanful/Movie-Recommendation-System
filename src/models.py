#!/usr/bin/env python3

"""
Alternating least-squares update and evaluation for MovieLens models.

This module implements two explicit-feedback collaborative filtering models:

1. **Bias-only**: ``prediction(u, i) = user_bias[u] + movie_bias[i]``.
2. **Bias plus factors**: ``prediction(u, i) = user_bias[u] + movie_bias[i]
   + user_factors[u] @ movie_factors[i]``.

Both models minimize a regularized *sum of squared errors* on the observed
training ratings. ``lambda_reg`` weights data fit, ``gamma`` penalizes user/item
biases, and ``tau`` penalizes latent factors; see the loss functions below.

Data format
-----------
The training and evaluation functions accept the grouped arrays produced by
``utils.split_into_train_test``:

* ``users``: row ``j`` contains ``(movie_indices, ratings)`` for the user
  identified by ``users_indices[j]``.
* ``movies``: row ``j`` contains ``(user_indices, ratings)`` for the movie
  identified by ``movies_indices[j]``.

Each index array is one-dimensional and integer-valued; each corresponding
rating array is one-dimensional and floating-point. The outer grouped arrays
may have object dtype, but each inner array passed to a Numba-compiled update
must have a numeric dtype.
"""

from __future__ import annotations

import numpy as np
from numba import njit

############################ BIAS ONLY MODEL #############################

@njit
def update_user_bias(
    movie_biases: np.ndarray,
    movie_indices: np.ndarray,
    ratings: np.ndarray,
    lambda_reg: float,
    gamma: float,
) -> float:
    """
    Compute the regularized closed-form bias update for one user.

    Parameters
    ----------
    movie_biases : numpy.ndarray, shape (n_movies,)
        Current item bias estimates.
    movie_indices : numpy.ndarray, shape (n_ratings,)
        Internal movie indices rated by this user.
    ratings : numpy.ndarray, shape (n_ratings,)
        Corresponding observed ratings.
    lambda_reg : float
        Positive weight of the squared-error term.
    gamma : float
        Nonnegative L2 regularization coefficient for biases.

    Returns
    -------
    float
        Updated bias for the selected user.
    """
    residuals = ratings - movie_biases[movie_indices]
    return lambda_reg * np.sum(residuals) / (lambda_reg * len(movie_indices) + gamma)

@njit
def update_movie_bias(
    user_biases: np.ndarray,
    user_indices: np.ndarray,
    ratings: np.ndarray,
    lambda_reg: float,
    gamma: float,
) -> float:
    """
    Compute the regularized closed-form bias update for one movie.

    Parameters
    ----------
    user_biases : numpy.ndarray, shape (n_users,)
        Current user bias estimates.
    user_indices : numpy.ndarray, shape (n_ratings,)
        Internal indices of users who rated the movie.
    ratings : numpy.ndarray, shape (n_ratings,)
        Corresponding observed ratings.
    lambda_reg : float
        Positive weight of the squared-error term.
    gamma : float
        Nonnegative L2 regularization coefficient for biases.

    Returns
    -------
    float
        Updated bias for the selected movie.
    """
    residuals = ratings - user_biases[user_indices]
    return lambda_reg * np.sum(residuals) / (lambda_reg * len(user_indices) + gamma)

@njit
def get_sq_error(
    user: int,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    movie_indices: np.ndarray,
    ratings: np.ndarray,
) -> np.ndarray:
    """
    Return per-rating squared errors for one user (bias-only model).
    """
    predictions = user_biases[user] + movie_biases[movie_indices]
    return np.square(ratings - predictions)

@njit
def get_sum_sq_error(sq_error: np.ndarray) -> float:
    """
    Compute the sum of squared errors.
    """
    return np.sum(sq_error)

def update_biases(
    users_train: np.ndarray,
    movies_train: np.ndarray,
    users_indices_train: np.ndarray,
    movies_indices_train: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    lambda_reg: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one alternating update sweep for the bias-only model.

    Parameters
    ----------
    users_train : numpy.ndarray
        User-grouped training observations; see module-level data format.
    movies_train : numpy.ndarray
        Movie-grouped training observations.
    users_indices_train : numpy.ndarray, shape (n_active_users,)
        User IDs corresponding to the rows of ``users_train``.
    movies_indices_train : numpy.ndarray, shape (n_active_movies,)
        Movie IDs corresponding to the rows of ``movies_train``.
    user_biases : numpy.ndarray, shape (n_users,)
        User biases, updated in place.
    movie_biases : numpy.ndarray, shape (n_movies,)
        Movie biases, updated in place.
    lambda_reg : float
        Positive weight of the squared-error term.
    gamma : float
        Nonnegative L2 bias regularization coefficient.

    Returns
    -------
    tuple of numpy.ndarray
        The same ``(user_biases, movie_biases)`` arrays, after one sweep.
    """
    for row, user_idx in enumerate(users_indices_train):
        movie_indices, ratings = users_train[row]
        user_biases[user_idx] = update_user_bias(
            movie_biases, movie_indices, ratings, lambda_reg, gamma
        )

    for row, movie_idx in enumerate(movies_indices_train):
        user_indices, ratings = movies_train[row]
        movie_biases[movie_idx] = update_movie_bias(
            user_biases, user_indices, ratings, lambda_reg, gamma
        )

    return user_biases, movie_biases

def compute_loss(
    users_train: np.ndarray,
    users_indices_train: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    lambda_reg: float,
    gamma: float,
) -> float:
    """
    Calculate the bias-only training objective (including L2 penalties).

    Parameters
    ----------
    users_train : numpy.ndarray
        User-grouped training observations.
    users_indices_train : numpy.ndarray
        User IDs corresponding to ``users_train``.
    user_biases : numpy.ndarray
        Current user biases.
    movie_biases : numpy.ndarray
        Current movie biases.
    lambda_reg : float
        Positive squared-error weight.
    gamma : float
        Nonnegative bias regularization coefficient.

    Returns
    -------
    float
        Value of the regularized objective on the training observations.
    """
    squared_error_sum = 0.0
    for row, user_idx in enumerate(users_indices_train):
        movie_indices, ratings = users_train[row]
        squared_errors = get_sq_error(
            user_idx, user_biases, movie_biases, movie_indices, ratings
        )
        squared_error_sum += get_sum_sq_error(squared_errors)

    bias_penalty = (
        np.sum(np.square(user_biases)) + np.sum(np.square(movie_biases))
    )
    return float(
        0.5 * lambda_reg * squared_error_sum + 0.5 * gamma * bias_penalty
    )

def compute_rmse(
    users_indices: np.ndarray,
    users: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
) -> float:
    """
    Compute bias-only RMSE over the supplied observed ratings.

    Parameters
    ----------
    users_indices : numpy.ndarray
        User IDs corresponding to the grouped observations.
    users : numpy.ndarray
        User-grouped observations, either training or test.
    user_biases : numpy.ndarray
        Current user biases.
    movie_biases : numpy.ndarray
        Current movie biases.

    Returns
    -------
    float
        Root mean squared prediction error over individual ratings.

    Raises
    ------
    ValueError
        If there are no observations to evaluate.
    """
    total_squared_error = 0.0
    n_ratings = 0

    for row, user_idx in enumerate(users_indices):
        movie_indices, ratings = users[row]
        squared_errors = get_sq_error(
            user_idx, user_biases, movie_biases, movie_indices, ratings
        )
        total_squared_error += get_sum_sq_error(squared_errors)
        n_ratings += len(ratings)

    if n_ratings == 0:
        raise ValueError("RMSE cannot be computed for an empty ratings set.")

    return float(np.sqrt(total_squared_error / n_ratings))

######################## BIAS & LATENT-FACTOR MODEL ########################

@njit
def update_user_bias_with_embeddings(
    user_idx: int,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    movie_biases: np.ndarray,
    movie_indices: np.ndarray,
    ratings: np.ndarray,
    lambda_reg: float,
    gamma: float,
) -> float:
    """
    Update one user bias with user and movie factors held fixed.

    Parameters
    ----------
    user_idx : int
        Internal ID of the user being updated.
    user_embeddings : numpy.ndarray, shape (n_users, n_factors)
        Current user latent-factor matrix.
    movie_embeddings : numpy.ndarray, shape (n_movies, n_factors)
        Current movie latent-factor matrix.
    movie_biases : numpy.ndarray, shape (n_movies,)
        Current movie biases.
    movie_indices : numpy.ndarray, shape (n_ratings,)
        Movie IDs rated by this user.
    ratings : numpy.ndarray, shape (n_ratings,)
        Corresponding observed ratings.
    lambda_reg : float
        Positive squared-error weight.
    gamma : float
        Nonnegative bias regularization coefficient.

    Returns
    -------
    float
        Updated user bias.
    """
    factor_predictions = (
        user_embeddings[user_idx] @ movie_embeddings[movie_indices].T
    )
    residuals = ratings - factor_predictions - movie_biases[movie_indices]
    return lambda_reg * np.sum(residuals) / (lambda_reg * len(movie_indices) + gamma)

@njit
def update_user_embedding(
    user_idx: int,
    movie_embeddings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    movie_indices: np.ndarray,
    ratings: np.ndarray,
    lambda_reg: float,
    tau: float,
) -> np.ndarray:
    """
    Solve the regularized least-squares problem for one user factor.

    Parameters
    ----------
    user_idx : int
        User whose factor is being updated.
    movie_embeddings : numpy.ndarray, shape (n_movies, n_factors)
        Fixed movie latent-factor matrix.
    user_biases : numpy.ndarray, shape (n_users,)
        Current user biases.
    movie_biases : numpy.ndarray, shape (n_movies,)
        Current movie biases.
    movie_indices : numpy.ndarray, shape (n_ratings,)
        IDs of movies rated by the user.
    ratings : numpy.ndarray, shape (n_ratings,)
        Corresponding observed ratings.
    lambda_reg : float
        Positive squared-error weight.
    tau : float
        Positive L2 coefficient for user/movie latent factors.

    Returns
    -------
    numpy.ndarray, shape (n_factors,)
        Updated user factor vector.
    """
    rated_movie_factors = movie_embeddings[movie_indices]
    residuals = ratings - user_biases[user_idx] - movie_biases[movie_indices]

    gram_matrix = rated_movie_factors.T @ rated_movie_factors
    normal_matrix = lambda_reg * gram_matrix + tau * np.eye(
        movie_embeddings.shape[1]
    )
    right_hand_side = lambda_reg * (rated_movie_factors.T @ residuals)
    return np.linalg.solve(normal_matrix, right_hand_side)

@njit
def update_movie_bias_with_embeddings(
    movie_idx: int,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    user_biases: np.ndarray,
    user_indices: np.ndarray,
    ratings: np.ndarray,
    lambda_reg: float,
    gamma: float,
) -> float:
    """
    Update one movie bias with the latent factors held fixed.

    Parameters
    ----------
    movie_idx : int
        Internal ID of the movie being updated.
    user_embeddings : numpy.ndarray, shape (n_users, n_factors)
        Current user factor matrix.
    movie_embeddings : numpy.ndarray, shape (n_movies, n_factors)
        Current movie factor matrix.
    user_biases : numpy.ndarray, shape (n_users,)
        Current user biases.
    user_indices : numpy.ndarray, shape (n_ratings,)
        IDs of users who rated this movie.
    ratings : numpy.ndarray, shape (n_ratings,)
        Corresponding observed ratings.
    lambda_reg : float
        Positive squared-error weight.
    gamma : float
        Nonnegative bias regularization coefficient.

    Returns
    -------
    float
        Updated movie bias.
    """
    factor_predictions = (
        user_embeddings[user_indices] @ movie_embeddings[movie_idx]
    )
    residuals = ratings - factor_predictions - user_biases[user_indices]
    return lambda_reg * np.sum(residuals) / (lambda_reg * len(user_indices) + gamma)

@njit
def update_movie_embedding(
    movie_idx: int,
    user_embeddings: np.ndarray,
    user_biases: np.ndarray,
    movie_biases: np.ndarray,
    user_indices: np.ndarray,
    ratings: np.ndarray,
    lambda_reg: float,
    tau: float,
) -> np.ndarray:
    """
    Solve the regularized least-squares problem for one movie factor.

    Parameters
    ----------
    movie_idx : int
        Movie whose factor is being updated.
    user_embeddings : numpy.ndarray, shape (n_users, n_factors)
        Fixed user latent-factor matrix.
    user_biases : numpy.ndarray, shape (n_users,)
        Current user biases.
    movie_biases : numpy.ndarray, shape (n_movies,)
        Current movie biases.
    user_indices : numpy.ndarray, shape (n_ratings,)
        IDs of users who rated this movie.
    ratings : numpy.ndarray, shape (n_ratings,)
        Corresponding observed ratings.
    lambda_reg : float
        Positive squared-error weight.
    tau : float
        Positive L2 coefficient for user/movie latent factors.

    Returns
    -------
    numpy.ndarray, shape (n_factors,)
        Updated movie factor vector.
    """
    rating_user_factors = user_embeddings[user_indices]
    residuals = ratings - movie_biases[movie_idx] - user_biases[user_indices]

    gram_matrix = rating_user_factors.T @ rating_user_factors
    normal_matrix = lambda_reg * gram_matrix + tau * np.eye(
        user_embeddings.shape[1]
    )
    right_hand_side = lambda_reg * (rating_user_factors.T @ residuals)
    return np.linalg.solve(normal_matrix, right_hand_side)

@njit
def get_sq_error_embeddings(
    user_idx: int,
    user_biases: np.ndarray,
    user_embeddings: np.ndarray,
    movie_biases: np.ndarray,
    movie_embeddings: np.ndarray,
    movie_indices: np.ndarray,
    ratings: np.ndarray,
) -> np.ndarray:
    """
    Return per-rating squared errors for one user (factor model).
    """
    predictions = (
        user_embeddings[user_idx] @ movie_embeddings[movie_indices].T
        + user_biases[user_idx]
        + movie_biases[movie_indices]
    )
    return np.square(ratings - predictions)

def update_user_biases_and_embeddings(
    users_train: np.ndarray,
    users_indices_train: np.ndarray,
    user_biases: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    movie_biases: np.ndarray,
    lambda_reg: float,
    gamma: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one user-side alternating sweep for the factor model.

    Parameters
    ----------
    users_train : numpy.ndarray
        Grouped training observations for active users.
    users_indices_train : numpy.ndarray
        User IDs corresponding to rows of ``users_train``.
    user_biases : numpy.ndarray
        User biases, updated in place.
    user_embeddings : numpy.ndarray
        User factor matrix, updated in place.
    movie_embeddings : numpy.ndarray
        Current (fixed during this sweep) movie factor matrix.
    movie_biases : numpy.ndarray
        Current movie biases.
    lambda_reg : float
        Positive squared-error weight.
    gamma : float
        Nonnegative bias regularization coefficient.
    tau : float
        Positive latent-factor regularization coefficient.

    Returns
    -------
    tuple of numpy.ndarray
        The updated ``(user_biases, user_embeddings)`` arrays.
    """
    for row, user_idx in enumerate(users_indices_train):
        movie_indices, ratings = users_train[row]
        user_biases[user_idx] = update_user_bias_with_embeddings(
            user_idx,
            user_embeddings,
            movie_embeddings,
            movie_biases,
            movie_indices,
            ratings,
            lambda_reg,
            gamma,
        )
        user_embeddings[user_idx] = update_user_embedding(
            user_idx,
            movie_embeddings,
            user_biases,
            movie_biases,
            movie_indices,
            ratings,
            lambda_reg,
            tau,
        )

    return user_biases, user_embeddings

def update_movie_biases_and_embeddings(
    movies_train: np.ndarray,
    movies_indices_train: np.ndarray,
    user_biases: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    movie_biases: np.ndarray,
    lambda_reg: float,
    gamma: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one movie-side alternating sweep for the factor model.

    Parameters
    ----------
    movies_train : numpy.ndarray
        Grouped training observations for active movies.
    movies_indices_train : numpy.ndarray
        Movie IDs corresponding to rows of ``movies_train``.
    user_biases : numpy.ndarray
        Current user biases.
    user_embeddings : numpy.ndarray
        Current user factor matrix.
    movie_embeddings : numpy.ndarray
        Movie factor matrix, updated in place.
    movie_biases : numpy.ndarray
        Movie biases, updated in place.
    lambda_reg : float
        Positive squared-error weight.
    gamma : float
        Nonnegative bias regularization coefficient.
    tau : float
        Positive latent-factor regularization coefficient.

    Returns
    -------
    tuple of numpy.ndarray
        The updated ``(movie_biases, movie_embeddings)`` arrays.
    """
    for row, movie_idx in enumerate(movies_indices_train):
        user_indices, ratings = movies_train[row]
        movie_biases[movie_idx] = update_movie_bias_with_embeddings(
            movie_idx,
            user_embeddings,
            movie_embeddings,
            user_biases,
            user_indices,
            ratings,
            lambda_reg,
            gamma,
        )
        movie_embeddings[movie_idx] = update_movie_embedding(
            movie_idx,
            user_embeddings,
            user_biases,
            movie_biases,
            user_indices,
            ratings,
            lambda_reg,
            tau,
        )

    return movie_biases, movie_embeddings

def compute_loss_with_embeddings(
    users_train: np.ndarray,
    users_indices_train: np.ndarray,
    user_biases: np.ndarray,
    user_embeddings: np.ndarray,
    movie_biases: np.ndarray,
    movie_embeddings: np.ndarray,
    lambda_reg: float,
    gamma: float,
    tau: float,
) -> float:
    """
    Calculate the complete latent-factor training objective.

    Parameters
    ----------
    users_train : numpy.ndarray
        User-grouped training observations.
    users_indices_train : numpy.ndarray
        User IDs corresponding to ``users_train``.
    user_biases : numpy.ndarray
        User biases.
    user_embeddings : numpy.ndarray
        User latent factors.
    movie_biases : numpy.ndarray
        Movie biases.
    movie_embeddings : numpy.ndarray
        Movie latent factors.
    lambda_reg : float
        Positive squared-error weight.
    gamma : float
        Nonnegative L2 bias regularization coefficient.
    tau : float
        Positive L2 latent-factor regularization coefficient.

    Returns
    -------
    float
        Regularized loss on the supplied training ratings.
    """
    squared_error_sum = 0.0
    for row, user_idx in enumerate(users_indices_train):
        movie_indices, ratings = users_train[row]
        squared_errors = get_sq_error_embeddings(
            user_idx,
            user_biases,
            user_embeddings,
            movie_biases,
            movie_embeddings,
            movie_indices,
            ratings,
        )
        squared_error_sum += get_sum_sq_error(squared_errors)

    bias_penalty = (
        np.sum(np.square(user_biases)) + np.sum(np.square(movie_biases))
    )
    factor_penalty = (
        np.sum(np.square(user_embeddings))
        + np.sum(np.square(movie_embeddings))
    )
    return float(
        0.5 * lambda_reg * squared_error_sum
        + 0.5 * gamma * bias_penalty
        + 0.5 * tau * factor_penalty
    )

def compute_rmse_with_embeddings(
    users: np.ndarray,
    users_indices: np.ndarray,
    user_biases: np.ndarray,
    user_embeddings: np.ndarray,
    movie_biases: np.ndarray,
    movie_embeddings: np.ndarray,
) -> float:
    """
    Compute the factor-model RMSE over supplied observed ratings.

    Parameters
    ----------
    users : numpy.ndarray
        User-grouped observations, training or test.
    users_indices : numpy.ndarray
        User IDs corresponding to ``users``.
    user_biases : numpy.ndarray
        Current user biases.
    user_embeddings : numpy.ndarray
        Current user factors.
    movie_biases : numpy.ndarray
        Current movie biases.
    movie_embeddings : numpy.ndarray
        Current movie factors.

    Returns
    -------
    float
        Root mean squared prediction error over individual ratings.

    Raises
    ------
    ValueError
        If there are no observed ratings to evaluate.
    """
    total_squared_error = 0.0
    n_ratings = 0

    for row, user_idx in enumerate(users_indices):
        movie_indices, ratings = users[row]
        squared_errors = get_sq_error_embeddings(
            user_idx,
            user_biases,
            user_embeddings,
            movie_biases,
            movie_embeddings,
            movie_indices,
            ratings,
        )
        total_squared_error += get_sum_sq_error(squared_errors)
        n_ratings += len(ratings)

    if n_ratings == 0:
        raise ValueError(
            "RMSE cannot be computed for an empty ratings set."
        )

    return float(np.sqrt(total_squared_error / n_ratings))
