#!/usr/bin/env python 
import os
import numpy as np
from numba import jit, njit


##############################################################################
############################## BIAS ONLY MODEL ###############################
##############################################################################

@jit(nopython=True)
def update_user_bias(movie_biases, movie_indices, ratings, LAMBDA, GAMMA):
    """
    Update user bias using the biased collaborative filtering model.

    Args:
        movie_biases (arr): Array containing the current movie biases.
        movie_indices (arr): Array containing the indices of the movies rated by the user.
        ratings (arr): Array containing the ratings given by the user to the movies.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.

    Returns:
        float: New value of the user bias.
    """
    predictions = movie_biases[movie_indices]
    residual = ratings - predictions
    return LAMBDA * np.sum(residual) / (LAMBDA * len(movie_indices) + GAMMA)

@jit(nopython=True)
def update_movie_bias(user_biases, user_indices, ratings, LAMBDA, GAMMA):
    """
    Update movie bias using the biased collaborative filtering model.

    Args:
        user_biases (arr): Array containing the current user biases.
        user_indices (arr): Array containing the indices of the users who rated the movie.
        ratings (arr): Array containing the ratings given by the users to the movie.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.
    
    Returns:
        float: New value of the movie bias.
    """
    predictions = user_biases[user_indices]
    residual = ratings - predictions
    return LAMBDA * np.sum(residual) / (LAMBDA * len(user_indices) + GAMMA)


@jit(nopython = True)
def getSqError(user, user_biases, movie_biases, movie_indices, ratings):
    prediction = user_biases[user] + movie_biases[movie_indices]
    return np.square(ratings - prediction)

@jit(nopython = True)
def getSumSqError(sq_error):
    return np.sum(sq_error)

def update_biases(users_train, movies_train, users_indices_train, movies_indices_train,
                  user_biases, movie_biases, LAMBDA, GAMMA):
    """
    Update the biases for the biased collaborative filtering model.

    Args:
        users_train (arr): Array containing the training data for users.
        movies_train (arr): Array containing the training data for movies.
        users_indices_train (arr): Array containing the indices of the users in the training data.
        movies_indices_train (arr): Array containing the indices of the movies in the training data.
        user_biases (arr): Array containing the current user biases.
        movie_biases (arr): Array containing the current movie biases.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.

    Returns:
        tuple: New values of the user biases and movie biases.
    """
    for idx, user_idx in enumerate(users_indices_train):
        ratings = users_train[idx][1]
        movie_indices = users_train[idx][0]
        user_biases[user_idx] = update_user_bias(
            movie_biases, movie_indices, ratings, LAMBDA, GAMMA
            )
    for idx, movie_idx in enumerate(movies_indices_train):
        ratings = movies_train[idx][1]
        user_indices = movies_train[idx][0]
        movie_biases[movie_idx] = update_movie_bias(
            user_biases, user_indices, ratings, LAMBDA, GAMMA
            )
    return user_biases, movie_biases
    
 
def compute_loss(users_train, users_indices_train, user_biases, movie_biases, LAMBDA, GAMMA):
    """
    Compute the loss of the biased collaborative filtering model.

    Args:
        users_train (arr): Array containing the training data for users.
        users_indices_train (arr): Array containing the indices of the users in the training data.
        user_biases (arr): Array containing the current user biases.
        movie_biases (arr): Array containing the current movie biases.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.

    Returns:
        float: Loss value of the biased collaborative filtering model.
    """
    loss = 0.0
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        sq_error = getSqError(user_idx, user_biases, movie_biases, movie_indices, ratings)
        sum_sq_errors = getSumSqError(sq_error)
        bias_terms =  np.sum(np.square(user_biases)) + np.sum(np.square(movie_biases))
        loss += 0.5 * LAMBDA * sum_sq_errors  + 0.5 * GAMMA * bias_terms
    return loss


def compute_rmse(users_indices, users, user_biases, movie_biases):
    """
    Compute the root mean squared error of the biased collaborative filtering model.

    Args:
        users_indices (arr): Array containing the indices of the users.
        users (arr): Array containing the ratings data for users.
        user_biases (arr): Array containing the current user biases.
        movie_biases (arr): Array containing the current movie biases.

    Returns:
        float: Root mean squared error value of the biased collaborative filtering model.
    """
    rmse = []
    for idx, user_idx in enumerate(users_indices):
        movie_indices = users[idx][0]
        ratings = users[idx][1]
        sq_error = getSqError(user_idx, user_biases, movie_biases, movie_indices, ratings)
        rmse.extend(sq_error)
    return np.sqrt(np.mean(rmse))


##############################################################################
######################## BIAS & EMBEDDINGS ONLY MODEL ########################
##############################################################################

@jit(nopython=True)
def update_user_bias_with_embeddings(user_idx, user_embeddings, movie_embeddings, 
                                     movie_biases, movie_indices, ratings, LAMBDA, GAMMA):
    """
    Update user bias using the collaborative filtering model with embeddings

    Args:
        user_idx (int): The index of the user
        user_embeddings (arr): Array containing the embeddings for the users.
        movie_embeddings (arr): Array containing the embeddings for the movies.
        movie_biases (arr): Array containing the current movie biases.
        movie_indices (arr): Array containing the indices of the movies rated by the user.
        ratings (arr): Array containing the ratings given by the user to the movies.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.

    Returns:
        float: New value of the user bias.
    """
    predictions = user_embeddings[user_idx] @ movie_embeddings[movie_indices].T + movie_biases[movie_indices]
    residual = ratings - predictions
    return LAMBDA * np.sum(residual) / (LAMBDA * len(movie_indices) + GAMMA)


@jit(nopython=True)
def update_user_embedding(user_idx, movie_embeddings, user_biases,
                          movie_biases, movie_indices, ratings, LAMBDA, TAU):
    """
    Update user embedding using the collaborative filtering model with embeddings.

    Args:
        user_idx (int): The index of the user
        movie_embeddings (arr): Array containing the embeddings for the movies.
        user_biases (arr): Array containing the current user biases.
        movie_biases (arr): Array containing the current movie biases.
        movie_indices (arr): Array containing the indices of the movies rated by the user.
        ratings (arr): Array containing the ratings given by the user to the movies.
        LAMBDA (float): Regularization parameter for the bias update.
        TAU (_type_): _description_

    Returns:
        (arr): Array containing the current updated user embeddings
    """
    predictions = user_biases[user_idx] + movie_biases[movie_indices]
    residual = ratings - predictions
    other_term = LAMBDA * movie_embeddings[movie_indices].T @ residual
    inverse_term = LAMBDA * movie_embeddings[movie_indices].T @ movie_embeddings[movie_indices] \
        + TAU * np.eye(movie_embeddings.shape[1])
    return np.linalg.solve(inverse_term, other_term)
    

@jit(nopython=True)
def update_movie_bias_with_embeddings(movie_idx, user_embeddings, movie_embeddings, user_biases, user_indices, ratings, LAMBDA, GAMMA):
    """
    Update movie bias using the biased collaborative filtering model with embeddings.

    Args:
        movie_idx (int): The index of the movie
        movie_embeddings (arr): Array containing the embeddings for the movies.
        user_biases (arr): Array containing the current user biases.
        user_indices (arr): Array containing the indices of the users who rated the movie.
        ratings (arr): Array containing the ratings given by the users to the movie.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.
    
    Returns:
        float: New value of the movie bias.
    """
    predictions = user_embeddings[user_indices] @ movie_embeddings[movie_idx].T + user_biases[user_indices]
    residual = ratings - predictions
    return LAMBDA * np.sum(residual) / (LAMBDA * len(user_indices) + GAMMA)

@jit(nopython=True)
def update_movie_embedding(movie_idx, user_embeddings, user_biases,
                          movie_biases, user_indices, ratings, LAMBDA, TAU):
    """
    Update movie embedding using the biased collaborative filtering model with embedding.

    Args:
        movie_idx (int): The index of the movie
        user_embeddings (arr): Array containing the embeddings for the users.
        user_biases (arr): Array containing the current user biases.
        movie_biases (arr): Array containing the current movie biases.
        user_indices (arr): Array containing the indices of the users who rated the movie.
        ratings (arr): Array containing the ratings given by the users to the movie.
        LAMBDA (float): Regularization parameter for the bias update.
        TAU (_type_): _description_

    Returns:
        (arr): Array containing the current updated movie embedding
    """
    predictions = movie_biases[movie_idx] + user_biases[user_indices]
    residual = ratings - predictions
    other_term = LAMBDA * user_embeddings[user_indices].T @ residual
    inverse_term = LAMBDA * user_embeddings[user_indices].T @ user_embeddings[user_indices] \
        + TAU * np.eye(user_embeddings.shape[1])
    return np.linalg.solve(inverse_term, other_term)


@jit(nopython = True)
def getSqErrorEmbeddings(user_idx, user_biases, user_embeddings, movie_biases, movie_embeddings, movie_indices, ratings):
    prediction = user_embeddings[user_idx] @ movie_embeddings[movie_indices].T + user_biases[user_idx] + movie_biases[movie_indices]
    return np.square(ratings - prediction)

@jit(nopython = True)
def getSumSqError(sq_error):
    return np.sum(sq_error)


def update_user_biases_and_embeddings(users_train, users_indices_train,
                  user_biases, user_embeddings, movie_embeddings, movie_biases, LAMBDA, GAMMA, TAU):
    """
    Update the biases  and embeddingsfor the biased collaborative filtering model with embeddings.

    Args:
        users_train (arr): Array containing the training data for users.
        users_indices_train (arr): Array containing the indices of the users in the training data.
        user_biases (arr): Array containing the current user biases.
        user_embeddings (arr): Array containing the embeddings for the users.
        movie_embeddings (arr): Array containing the embeddings for the movies.
        movie_biases (arr): Array containing the current movie biases.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.

    Returns:
        tuple: New values of the user biases and user embeddings.
    """
    for idx, user_idx in enumerate(users_indices_train):
        ratings = users_train[idx][1]
        movie_indices = users_train[idx][0]
        user_biases[user_idx] = update_user_bias_with_embeddings(
            user_idx, user_embeddings, movie_embeddings, movie_biases, movie_indices, ratings, LAMBDA, GAMMA
            )
        user_embeddings[user_idx] = update_user_embedding(
            user_idx, movie_embeddings, user_biases, movie_biases, movie_indices, ratings, LAMBDA, TAU
            )
    return user_biases, user_embeddings
    

def update_movie_biases_and_embeddings(movies_train, movies_indices_train,
                  user_biases, user_embeddings, movie_embeddings, movie_biases, LAMBDA, GAMMA, TAU):
    """
    Update the biases for the biased collaborative filtering model with embeddings.

    Args:
        movies_train (arr): Array containing the training data for movies.
        movie_indices_train (arr): Array containing the indices of the movies in the training data.
        user_biases (arr): Array containing the current user biases.
        user_embeddings (arr): Array containing the embeddings for the users.
        movie_embeddings (arr): Array containing the embeddings for the movies.
        movie_biases (arr): Array containing the current movie biases.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.
    Returns:
        tiple: New values of the movie biases and movie embeddings
    """
    for idx, movie_idx in enumerate(movies_indices_train):
        ratings = movies_train[idx][1]
        user_indices = movies_train[idx][0]
        movie_biases[movie_idx] = update_movie_bias_with_embeddings(
            movie_idx, user_embeddings, movie_embeddings, user_biases, user_indices, ratings, LAMBDA, GAMMA
        )
        movie_embeddings[movie_idx] = update_movie_embedding(
            movie_idx, user_embeddings, user_biases, movie_biases, user_indices, ratings, LAMBDA, TAU
            )
    return movie_biases, movie_embeddings
    
def compute_loss_with_embeddings(users_train, users_indices_train, user_biases, user_embeddings, 
                                 movie_biases, movie_embeddings, LAMBDA, GAMMA, TAU):
    """
    Compute the loss of the biased collaborative filtering model with embeddings.

    Args:
        users_train (arr): Array containing the training data for users.
        users_indices_train (arr): Array containing the indices of the users in the training data.
        user_biases (arr): Array containing the current user biases.
        user_embeddings (arr): Array containing user embeddings.
        movie_biases (arr): Array containing the current movie biases.
        movie_embeddings (arr): Array containing movie embeddings.
        LAMBDA (float): Regularization parameter for the bias update.
        GAMMA (float): Regularization parameter for the bias update.

    Returns:
        float: Loss value of the biased collaborative filtering model with embeddings.
    """
    loss = 0.0
    for idx, user_idx in enumerate(users_indices_train):
        movie_indices = users_train[idx][0]
        ratings = users_train[idx][1]
        sq_error = getSqErrorEmbeddings(user_idx, user_biases, user_embeddings, movie_biases, movie_embeddings, movie_indices, ratings)
        sum_sq_errors = getSumSqError(sq_error)
        loss += sum_sq_errors
    bias_terms = 0.5 * GAMMA * (np.sum(np.square(user_biases)) + np.sum(np.square(movie_biases)))
    embedding_terms = 0.5 * TAU * (np.sum(np.square(user_embeddings)) + np.sum(np.square(movie_embeddings)))
    return 0.5 * LAMBDA * loss + bias_terms + embedding_terms

def compute_rmse_with_embeddings(users, users_indices, user_biases, user_embeddings, 
                                 movie_biases, movie_embeddings):
    """
    Compute the root mean squared error of the biased collaborative filtering model with embeddings.

    Args:
        users (arr): Array containing the ratings data for users.
        users_indices (arr): Array containing the indices of the users.
        user_biases (arr): Array containing the current user biases.
        user_embeddings (arr): Array containing user embeddings.
        movie_biases (arr): Array containing the current movie biases.
        movie_embeddings (arr): Array containing movie embeddings.

    Returns:
        float: Root mean squared error value of the biased collaborative filtering model with embeddings.
    """
    rmse = []
    for idx, user_idx in enumerate(users_indices):
        movie_indices = users[idx][0]
        ratings = users[idx][1]
        sq_error = getSqErrorEmbeddings(user_idx, user_biases, user_embeddings, movie_biases,\
            movie_embeddings, movie_indices, ratings)
        rmse.extend(sq_error)
    return np.sqrt(np.mean(rmse))
