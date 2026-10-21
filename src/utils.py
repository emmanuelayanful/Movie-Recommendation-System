#!/usr/bin/env python 

import os
import sys
import pickle
import shutil
import zipfile
import logging
import numpy as np
import polars as pl
import seaborn as sns
import urllib.request
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


FONTSIZE = 18
params = {
    'axes.labelsize': FONTSIZE,
    'axes.titlesize': FONTSIZE,
    'xtick.labelsize': FONTSIZE,
    'legend.fontsize': FONTSIZE,
    'ytick.labelsize': FONTSIZE
}

plt.rcParams.update(params)


def setup_logging(experiment_folder, filename):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # prevent duplicate handlers if called multiple times
    logger.handlers.clear()
    
    # Create a file handler to log to a file in the experiment folder
    log_filename = os.path.join(experiment_folder, filename)
    file_handler = logging.FileHandler(log_filename, mode="w")
    # file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create a stream handler to print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



def save_data_to_pickle(save_path, data, filename, log=True):
    """
    Save a data object to a pickle file.
    
    Note: This function uses pickle version 4, which is more secure than previous versions.
    
    Parameters:
        save_path (str): Directory where the pickle file will be saved.
        data (object): The data object to save.
        filename (str): Name of the pickle file without the extension.
        log (bool): If True, print a message indicating a file has been successfully saved.
    """
    outfile = open(os.path.join(save_path, f'{filename}.pkl'), 'wb')
    pickle.dump(data, outfile, protocol=4)
    outfile.close()
    if log:
        print(f'Saved {filename} to .pkl file at {save_path}')
    


def load_data_from_pickle(save_path, filename, log=True):
    """
    Load a data object from a pickle file.
    
    Args:
        save_path (str): Directory where the pickle file is located.
        filename (str): Name of the pickle file without the extension.
        log (bool): If True, print a message indicating the file loaded.

    Returns:
        object: The loaded data object.
    """
    infile = open(os.path.join(save_path, f'{filename}.pkl'), 'rb')
    data = pickle.load(infile)
    infile.close()
    if log:
        print(f'Loaded {filename} from .pkl file at {save_path}')
    return data


def save_numpy_array(save_path, data, filename, log=True):
    """
    Save a NumPy array to file. Optionally save as a .pkl file.
    
    Args:
        save_path (str): Directory where the file will be saved.
        data (numpy.ndarray): The NumPy array to save.
        filename (str): Name of the file without the extension.
        as_pickle (bool): If True, save as a pickle file; otherwise save as .npy.
        log (bool): If True, print a message indicating the file saved.
    """
    np.save(os.path.join(save_path, f'{filename}.npy'), data, allow_pickle=True)
    if log:
        print(f'Saved {filename} to .npy file at {save_path}')
    


def load_numpy_array(save_path, filename, log=True):
    """
    Load a NumPy array from file.
    
    Args:
        save_path (str): Directory where the file is located.
        filename (str): Name of the file without the extension.
        log (bool): If True, print a message indicating the file loaded.

    Returns:
        numpy.ndarray: The loaded NumPy array.
    """
    if log:
        print(f'Loaded {filename} from .npy file at {save_path}')
        return np.load(os.path.join(save_path, f'{filename}.npy'), allow_pickle=True)
    else:
        return np.load(os.path.join(save_path, f'{filename}.npy'), allow_pickle=True)


###################### Data Access and Preprocess ########################
def load_ratings_data(base_path, dataset_name):
    """
    Load the ratings data and extract mappings from users to ids and vice versa.

    Args:
        base_path (str): Base directory where the  ratings filewould be.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    raw_folder = os.path.join(dataset_folder, "raw")             # ./data/ml-32m/raw
    preprocess_folder = os.path.join(dataset_folder, "preprocess")

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)
    ratings_file = os.path.join(raw_folder,'ratings.csv')

    # Load ratings data
    try:
        print('\nLoading ratings file')
        ratings = pl.read_csv(ratings_file)
    except FileNotFoundError:
        print(f'\n{ratings_file} not found. Downloading {dataset_name} from MovieLens website...')
        
        url = f'https://files.grouplens.org/datasets/movielens/{dataset_name}.zip'
        zip_path = os.path.join(raw_folder, f"{dataset_name}.zip")
        
        urllib.request.urlretrieve(url, zip_path)
        print('\nRatings file downloaded. Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_folder)

            for file in os.listdir(f"{raw_folder}/{dataset_name}"):
                shutil.move(
                    os.path.join(raw_folder, dataset_name, file),
                    os.path.join(raw_folder, file)
                )
        shutil.rmtree(f"{raw_folder}/{dataset_name}")
        ratings = pl.read_csv(ratings_file)
        print("\nLoaded.\n")

    # Group ratings by users and by movies
    data_by_users = ratings.group_by(
        'userId', maintain_order=True
    ).agg(pl.col('movieId'), pl.col('rating'))
    data_by_movies = ratings.group_by(
        'movieId', maintain_order=True
    ).agg(pl.col('userId'), pl.col('rating'))

    # Create mappings for users and movies
    map_user_to_id = {
        user_id: id for id, user_id in enumerate(data_by_users['userId'].to_list())
    }
    map_movie_to_id = {
        movie_id: id for id, movie_id in enumerate(data_by_movies['movieId'].to_list())
    }

    map_id_to_user = {
        id: user_id for id, user_id in enumerate(data_by_users['userId'].to_list())
    }
    map_id_to_movie = {
        id: movie_id for id, movie_id in enumerate(data_by_movies['movieId'].to_list())
    }

    # Save mappings to disk for future use
    save_data_to_pickle(preprocess_folder, map_user_to_id, 'map_user_to_id')
    save_data_to_pickle(preprocess_folder, map_movie_to_id, 'map_movie_to_id')
    save_data_to_pickle(preprocess_folder, map_id_to_user, 'map_id_to_user')
    save_data_to_pickle(preprocess_folder, map_id_to_movie, 'map_id_to_movie')
    save_data_to_pickle(preprocess_folder, ratings, 'ratings')

    # Delete files to avoid overloading memory
    del ratings, data_by_movies, data_by_users, map_id_to_movie, map_id_to_user, map_user_to_id, map_movie_to_id



def load_movie_data(base_path, dataset_name):
    """
    Load the movies data and create mappings from movies to respective titles.

    Args:
        base_path (str): Base directory where the  ratings file would be.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    raw_folder = os.path.join(dataset_folder, "raw")             # ./data/ml-32m/raw
    preprocess_folder = os.path.join(dataset_folder, "preprocess")

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)
    
    movies_file = os.path.join(raw_folder, 'movies.csv')
    if not os.path.exists(movies_file):
        print(f"{movies_file} not found. Calling load_ratings_data function to get data...")
        load_ratings_data(base_path, dataset_name)

    print(f"\nLoading {movies_file}...")
    movies = pl.read_csv(movies_file)
    links = pl.read_csv(os.path.join(raw_folder, 'links.csv'))
    print("\nLoaded.\n")

    # Map movies to titles
    map_movie_to_title = {
        movieid: title for movieid, title in zip(movies['movieId'].to_list(), movies['title'].to_list())
    }

    # Save mappings to disk for future use
    save_data_to_pickle(preprocess_folder, map_movie_to_title, 'map_movie_to_title')
    save_data_to_pickle(preprocess_folder, movies, 'movies')
    save_data_to_pickle(preprocess_folder, links, 'links')
    
    # Delete variables to save memory
    del movies, map_movie_to_title, links


def split_into_train_test(base_path: str, experiment_path: str, dataset_name: str, seed: int=42, test_size: float=.2):
    """
    Split ratings data into train and test sets.

    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        seed (int): Random seed for reproducibility. Defaults to 42.
        test_size (float): Fraction of the dat to be used as test data. Defaults to 0.2.
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)   # ./experiments/ml-32m
    data_split_folder = os.path.join(experiment_folder, "data_split")   # ./experiments/ml-32m/data_split

    os.makedirs(data_split_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    # Load ratimngs data and useful mappings
    ratings = load_data_from_pickle(preprocess_folder, 'ratings', log=False)
    map_movie_to_id = load_data_from_pickle(preprocess_folder, 'map_movie_to_id')
    map_user_to_id = load_data_from_pickle(preprocess_folder, 'map_user_to_id')

    # Perform a train-test split using polars gt function
    ratings = ratings.with_columns(
        [
            pl.col("userId").replace_strict(map_user_to_id).alias("userIdx"),
            pl.col("movieId").replace_strict(map_movie_to_id).alias("movieIdx")
        ]
    )

    # Add a random column for train-test split within each group
    rng = np.random.default_rng(seed)

    # Compute a random number for each row within the user group
    ratings_with_split = ratings.with_columns(
        pl.Series(
            "split", rng.uniform(size=ratings.height) # generate numbers for all rows
        )
    )

    # Assign rows to train or test based on threshold within each group.
    train_df = ratings_with_split.filter(pl.col("split") > test_size).drop("split")
    test_df = ratings_with_split.filter(pl.col("split") <= test_size).drop("split")

    # Split users and movies into train and test groups
    # Train
    users_train = train_df.group_by(
        'userIdx', maintain_order=True
    ).agg(pl.col('movieIdx'), pl.col('rating'))

    movies_train = train_df.group_by(
        'movieIdx', maintain_order=True
    ).agg(pl.col('userIdx'), pl.col('rating'))

    # Test
    users_test = test_df.group_by(
        'userIdx', maintain_order=True
    ).agg(pl.col('movieIdx'), pl.col('rating'))

    movies_test = test_df.group_by(
        'movieIdx', maintain_order=True
    ).agg(pl.col('userIdx'), pl.col('rating'))

    # Convert to numpy arrays
    # Train
    users_indices_train = users_train[:, 0].to_numpy()
    users_train = users_train[:, 1 : ].to_numpy()
    movies_indices_train = movies_train[:, 0].to_numpy()
    movies_train = movies_train[:, 1 : ].to_numpy()
    # Test
    users_indices_test = users_test[:, 0].to_numpy()
    users_test = users_test[:, 1 : ].to_numpy()
    movies_indices_test = movies_test[:, 0].to_numpy()
    movies_test = movies_test[:, 1 : ].to_numpy()

    # Save train and test dataframes to disk for future use
    save_numpy_array(data_split_folder, users_train, 'users_train')
    save_numpy_array(data_split_folder, users_test, 'users_test')
    save_numpy_array(data_split_folder, movies_train, 'movies_train')
    save_numpy_array(data_split_folder, movies_test, 'movies_test')
    save_numpy_array(data_split_folder, users_indices_train, 'users_indices_train')
    save_numpy_array(data_split_folder, users_indices_test, 'users_indices_test')
    save_numpy_array(data_split_folder, movies_indices_train, 'movies_indices_train')
    save_numpy_array(data_split_folder, movies_indices_test, 'movies_indices_test')

    # Delete variables to save memory
    del ratings, test_df, train_df, users_test, ratings_with_split, rng
    del movies_test, map_user_to_id, map_movie_to_id, users_train, movies_train


########################### Plotting #################################
def plot_distribution(base_path: str, experiment_path: str, dataset_name: str, title: str, xlabel: str, ylabel: str, color: str='g', kind: str='small'):
    """
    Calculate average movie ratings and plot their distributions.

    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabek (str): The label for the y-axis.
        color (str): Color for the histograms bars.
        kind (str): Use to set the value of "bw_adjust" parameter in the histogram plot depending on the size of the dataset we are using(e.g., 'small', 'large')
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    ratings = load_data_from_pickle(preprocess_folder, 'ratings', log=False)

    # Calculate the mean rating for each movie
    mean_ratings = (
        ratings.group_by("movieId")
        .agg(pl.col("rating").mean().alias("avg_rating"))
        .select("avg_rating")
        .to_numpy().flatten()
    )

    plt.figure(figsize=(8, 6))
    bins = 10
    bw_adjust = 2 if kind == 'small' else 3
    plt.hist(mean_ratings, bins=bins, density=True, color=color, edgecolor='black', alpha=0.7)
    kde = sns.kdeplot(mean_ratings, bw_adjust=bw_adjust, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(image_folder, f"ratings_distribution_{dataset_name}.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
    plt.show()


def plot_power_law_distribution(base_path: str, experiment_path: str, dataset_name: str, title: str, xlabel: str, ylabel: str):
    """
    Calculate average movie ratings and plot their distributions.

    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabek (str): The label for the y-axis.
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    ratings = load_data_from_pickle(preprocess_folder, 'ratings', log=False)

    # Calculate the number of rating for each movie
    movie_degree = (
        ratings.group_by("movieId")
        .count()
        .select(pl.col("count").alias("movie_ratings"))
        .to_numpy().flatten()
    )

    # Calculate the number of rating for each user
    user_degree = (
        ratings.group_by("userId")
        .count()
        .select(pl.col("count").alias("user_ratings"))
        .to_numpy().flatten()
    )

    # Unique degrees and their frequencies
    movie_degrees, movie_freq = np.unique(movie_degree, return_counts=True)
    user_degrees, user_freq = np.unique(user_degree, return_counts=True)

    # Plot power law distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(movie_degrees, movie_freq, marker='.', ls='none', c='g', label='movie')
    ax.loglog(user_degrees, user_freq, marker='.', ls='none', c='b', label='user')
    plt.axvline(x=20, ls='--', c='r')
    plt.text(21, max(movie_freq), 'Number of ratings = 20', c='r', fontsize=18)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.legend()
    plt.savefig(os.path.join(image_folder, f"power_law{dataset_name}.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
    plt.show()


def plot_loss_and_rmse(base_path: str, experiment_path: str, dataset_name: str, loss_history: dict, rmse_history: dict, model_type: str, add_title: bool=False):
    """
    Plot loss and RMSE over iterations for a given model type.
    Saves the plot in the specified folder.

    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        loss_history (list): Loss values for each iteration.
        rmse_history (dict): RMSE values for each iteration for train and test.
        model_type (str): Type of model (e.g., 'bias', 'bias_factors').
        add_title (bool, optional): Add title to the plot. Defaults to False.
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)
    
    if add_title:
        # Plot loss over iterations
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(loss_history, label="Train loss", color='blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Negative Log Likelihood (Loss) over Iterations')
        ax.legend()
        if model_type == 'bias':
            plt.savefig(os.path.join(image_folder, f"loss_plot_{dataset_name}_bias.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(image_folder, f"loss_plot_{dataset_name}.pdf"), format='pdf', dpi=3000, bbox_inches='tight')

        # Plot RMSE over iterations
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(rmse_history['train'], label="Train RMSE", color='red')
        ax.plot(rmse_history['test'], label="Test RMSE", color='blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Squared Error (RMSE) over Iterations')
        ax.legend()
        if model_type == 'bias':
            plt.savefig(os.path.join(image_folder, f"rmse_plot_{dataset_name}_bias.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(image_folder, f"rmse_plot_{dataset_name}.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
        plt.show()
    
    else:
        # Plot loss over iterations
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(loss_history, label="Train loss", color='blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        if model_type == 'bias':
            plt.savefig(os.path.join(image_folder, f"loss_plot_{dataset_name}_bias.pdf"), format='pdf', dpi=3000, bbox_inches='tight' )
        else:
            plt.savefig(os.path.join(image_folder, f"loss_plot_{dataset_name}.pdf"), format='pdf', dpi=3000, bbox_inches='tight' )

        # Plot RMSE over iterations
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(rmse_history['train'], label="Train RMSE", color='red')
        ax.plot(rmse_history['test'], label="Test RMSE", color='blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE')
        ax.legend()
        if model_type == 'bias':
            plt.savefig(os.path.join(image_folder, f"rmse_plot_{dataset_name}_bias.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(image_folder, f"rmse_plot_{dataset_name}.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
        plt.show()


def plot_embeddings(base_path: str, experiment_path: str, dataset_name: str, use_pca: bool=False, 
                    movie_ids=[1, 2116, 3114, 4993, 5952, 45722, 63992, 72407, 78772,
                               78499, 91104, 61160, 4886]):
    """
    Plot embeddings of movies for a given model type.
    Saves the plot in the specified folder.

    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        use_pca (bool): Choose between using PCA or T-SNE. Defaults to T-SNE
        num_movies (int, optional): Number of movies to plot embeddings. Defaults to 100.
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)
    
    model_path = os.path.join(experiment_folder, 'model_files')
    map_movie_to_id = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_movie_to_id', log=False)
    map_movie_to_title = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'),'map_movie_to_title', log=False)
    movie_embeddings = load_numpy_array(model_path, 'movie_embeddings', log=False)
    
    movie_indices = [map_movie_to_id[movie_id] for movie_id in movie_ids]
    # Reduce dimension if not equal to 2
    if movie_embeddings.shape[1] != 2 and use_pca:
        pca = PCA(n_components=2)
        movie_embeddings = pca.fit_transform(movie_embeddings[movie_indices])
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=10,          # typical: 5–50
            learning_rate='auto',
            init='random',
            random_state=42
        )
        movie_embeddings = tsne.fit_transform(movie_embeddings[movie_indices])
        
    plt.figure(figsize=(10, 6))
    plt.scatter(movie_embeddings[:, 0], movie_embeddings[:, 1], color='blue', s=10)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")

    for i, movie_id in enumerate(movie_ids):
        x_coord = movie_embeddings[i, 0]
        y_coord = movie_embeddings[i, 1]
        plt.text(x_coord, y_coord, map_movie_to_title[movie_id], fontsize=13)
    
    if use_pca:
        plt.savefig(os.path.join(image_folder, f"2D_embedding_plot_{dataset_name}_pca.pdf"), format='pdf', dpi=3000, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(image_folder, f"2D_embedding_plot_{dataset_name}_tsne.pdf"), format='pdf', dpi=3000, bbox_inches='tight')

    plt.show()


######################### Recommend Movies ##########################
def recommend_movies_streamlit(base_path: str, experiment_path: str, dataset_name: str, movie_ids: list, 
                     movie_ratings: list, top_n: int, LAMBDA: float, GAMMA: float, TAU: float):
    """
    Recommend movies for a *new* user based on rated movies.

    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        movie_ids (list) : List of movieId values that the user has rated.
        movie_ratings (list) : Corresponding ratings the user gave to those movie_ids.
        top_n (int) : Number of recommendations to return.
        LAMBDA, GAMMA, TAU  (float): Hyperparameters matching your training setup.
    """    

    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)
    
    model_path = os.path.join(experiment_folder, 'model_files')
    map_movie_to_id = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_movie_to_id', log=False)
    map_movie_to_title = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'),'map_movie_to_title', log=False)
    movie_embeddings = load_numpy_array(model_path, 'movie_embeddings', log=False)
    map_id_to_movie = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_id_to_movie', log=False)
    movie_biases = load_numpy_array(model_path, 'movie_biases', log=False)

    k = movie_embeddings.shape[1]   # embedding dimension

    # Build matrices for the rated movies
    # Map movie_ids to internal indices
    rated_indices = [map_movie_to_id[m_id] for m_id in movie_ids]

    # Stack embeddings and biases
    E = movie_embeddings[rated_indices]    
    b = movie_biases[rated_indices]       
    r = np.array(movie_ratings, dtype=float)

    n_rated = E.shape[0]

    # Initialize user embedding and bias
    user_embedding = np.zeros((k,))
    user_bias = 0.0

    # Iterative closed-form updates
    I_k = np.eye(k)

    for _ in range(5):
        # Predicted ratings with current user_embedding & bias (for rated items)
        pred = E @ user_embedding + b + user_bias
        residual = r - pred                        

        # Update user_embedding:
        inverse_term = LAMBDA * (E.T @ E) + TAU * I_k   
        other_term = E.T @ (LAMBDA * (r - (b + user_bias)))
        user_embedding = np.linalg.solve(inverse_term, other_term)

        # Update user_bias
        pred_no_ubias = E @ user_embedding + b
        residual_no_ubias = r - pred_no_ubias
        user_bias = LAMBDA * residual_no_ubias.sum() / (LAMBDA * n_rated + GAMMA)

    # Compute scores for all movies 
    # shape: (num_movies,)
    scores = movie_embeddings @ user_embedding + user_bias + 0.05 * movie_biases

    # Don't recommend the movies the user already rated
    mask = np.ones_like(scores, dtype=bool)
    mask[rated_indices] = False

    scores_filtered = scores.copy()
    scores_filtered[~mask] = -np.inf  # so they never appear in top-N

    # Top-N indices
    top_indices = np.argsort(scores_filtered)[-top_n:][::-1]

    # Map to movie IDs and titles
    top_movie_ids = [map_id_to_movie[i] for i in top_indices]
    item_titles = [map_movie_to_title[m_id] for m_id in top_movie_ids]

    return item_titles


def recommend_movies(base_path: str, experiment_path: str, dataset_name: str, movie_ids: list, 
                     movie_ratings: list, top_n: int, LAMBDA: float, GAMMA: float, TAU: float):
    """
    Recommend movies for a *new* user based on rated movies.

    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        movie_ids (list) : List of movieId values that the user has rated.
        movie_ratings (list) : Corresponding ratings the user gave to those movie_ids.
        top_n (int) : Number of recommendations to return.
        LAMBDA, GAMMA, TAU  (float): Hyperparameters matching your training setup.
    """    

    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)
    
    model_path = os.path.join(experiment_folder, 'model_files')
    map_movie_to_id = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_movie_to_id', log=False)
    map_movie_to_title = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'),'map_movie_to_title', log=False)
    movie_embeddings = load_numpy_array(model_path, 'movie_embeddings', log=False)
    map_id_to_movie = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_id_to_movie', log=False)
    movie_biases = load_numpy_array(model_path, 'movie_biases', log=False)

    k = movie_embeddings.shape[1]   # embedding dimension

    # Build matrices for the rated movies
    # Map movie_ids to internal indices
    rated_indices = [map_movie_to_id[m_id] for m_id in movie_ids]

    # Stack embeddings and biases
    E = movie_embeddings[rated_indices]    
    b = movie_biases[rated_indices]       
    r = np.array(movie_ratings, dtype=float)

    n_rated = E.shape[0]

    # Initialize user embedding and bias
    user_embedding = np.zeros((k,))
    user_bias = 0.0

    # Iterative closed-form updates
    I_k = np.eye(k)

    for _ in range(5):
        # Predicted ratings with current user_embedding & bias (for rated items)
        pred = E @ user_embedding + b + user_bias
        residual = r - pred                        

        # Update user_embedding:
        inverse_term = LAMBDA * (E.T @ E) + TAU * I_k   
        other_term = E.T @ (LAMBDA * (r - (b + user_bias)))
        user_embedding = np.linalg.solve(inverse_term, other_term)

        # Update user_bias
        pred_no_ubias = E @ user_embedding + b
        residual_no_ubias = r - pred_no_ubias
        user_bias = LAMBDA * residual_no_ubias.sum() / (LAMBDA * n_rated + GAMMA)

    # Compute scores for all movies 
    # shape: (num_movies,)
    scores = movie_embeddings @ user_embedding + user_bias + 0.05 * movie_biases

    # Don't recommend the movies the user already rated
    mask = np.ones_like(scores, dtype=bool)
    mask[rated_indices] = False

    scores_filtered = scores.copy()
    scores_filtered[~mask] = -np.inf  # so they never appear in top-N

    # Top-N indices
    top_indices = np.argsort(scores_filtered)[-top_n:][::-1]

    # Map to movie IDs and titles
    top_movie_ids = [map_id_to_movie[i] for i in top_indices]
    item_titles = [map_movie_to_title[m_id] for m_id in top_movie_ids]

    # Pretty print
    liked_titles = [map_movie_to_title[m_id] for m_id in movie_ids]
    print("Since you liked:")
    for t, r_ in zip(liked_titles, movie_ratings):
        print(f"  - '{t}' (rating: {r_})")
    print("\nYou may also like:\n")
    for i, title in enumerate(item_titles):
        print(f"{i+1}. {title}")


def recommend_movies_bias_only(base_path: str, experiment_path: str, dataset_name: str,
    movie_ids, movie_ratings, top_n: int, LAMBDA: float, GAMMA: float):
    """
    Recommend movies for a *new* user based on rated movies.
    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        movie_ids (list) : List of movieId values that the user has rated.
        movie_ratings (list) : Corresponding ratings the user gave to those movie_ids.
        top_n (int) : Number of recommendations to return.
        LAMBDA, GAMMA (float): Hyperparameters matching your training setup.
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    # load mappings and model parameters
    model_path = os.path.join(experiment_folder, 'model_files')
    map_movie_to_id = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_movie_to_id', log=False)
    map_movie_to_title = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'),'map_movie_to_title', log=False)
    map_id_to_movie = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_id_to_movie', log=False)
    movie_biases = load_numpy_array(model_path, 'movie_biases', log=False)

    # movie_biases
    movie_biases = load_numpy_array(model_path, "movie_biases_b", log=False)
    # global_mean: scalar
    ratings = load_data_from_pickle(preprocess_folder, 'ratings', log=False)
    global_mean = ratings['rating'].mean()

    # build arrays for rated movies
    rated_indices = [map_movie_to_id[m_id] for m_id in movie_ids]
    r = np.array(movie_ratings, dtype=float)          # (n,)
    b_i = movie_biases[rated_indices]                 # (n,)
    n_rated = len(rated_indices)

    # compute new user bias b_u 
    residuals = r - (global_mean + b_i)               # (n,)
    user_bias = LAMBDA * residuals.sum() / (LAMBDA * n_rated + GAMMA)

    # predict score for all movies
    scores = global_mean + movie_biases + user_bias   # (num_movies,)

    # don't recommend already-rated movies
    scores[rated_indices] = -np.inf

    # top-N
    top_indices = np.argsort(scores)[-top_n:][::-1]

    top_movie_ids = [map_id_to_movie[i] for i in top_indices]
    top_titles = [map_movie_to_title[m_id] for m_id in top_movie_ids]

    # Pretty print

    liked_titles = [map_movie_to_title[m_id] for m_id in movie_ids]

    print("Since you liked:")

    for t, r_ in zip(liked_titles, movie_ratings):

        print(f"  - '{t}' (rating: {r_})")

    print("\nYou may also like:\n")

    for i, title in enumerate(top_titles):

        print(f"{i+1}. {title}")



def recommend_movies_bias_only_streamlit(base_path: str, experiment_path: str, dataset_name: str,
    movie_ids, movie_ratings, top_n: int, LAMBDA: float, GAMMA: float):
    """
    Recommend movies for a *new* user based on rated movies.
    Args:
        base_path (str): Base directory where the ratings file would be.
        experiment_path (str): Directory where experiment results will be saved.
        dataset_name (str): Should be the name of the data as seen on the MovieLens website (ml-latest-small, ml-25m, ml-32m etc).
        movie_ids (list) : List of movieId values that the user has rated.
        movie_ratings (list) : Corresponding ratings the user gave to those movie_ids.
        top_n (int) : Number of recommendations to return.
        LAMBDA, GAMMA (float): Hyperparameters matching your training setup.
    """
    # Create directory for the experiment (small, 25m, 32m)
    dataset_folder = os.path.join(base_path, dataset_name)       # ./data/ml-32m
    preprocess_folder = os.path.join(dataset_folder, "preprocess")
    experiment_folder = os.path.join(experiment_path, dataset_name)    # ./experiments/ml-32m
    image_folder = os.path.join(experiment_folder, "images")     # ./experiments/ml-32m/images

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    # load mappings and model parameters
    model_path = os.path.join(experiment_folder, 'model_files')
    map_movie_to_id = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_movie_to_id', log=False)
    map_movie_to_title = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'),'map_movie_to_title', log=False)
    map_id_to_movie = load_data_from_pickle(os.path.join(dataset_folder, 'preprocess'), 'map_id_to_movie', log=False)
    movie_biases = load_numpy_array(model_path, 'movie_biases', log=False)

    # movie_biases
    movie_biases = load_numpy_array(model_path, "movie_biases_b", log=False)
    # global_mean: scalar
    ratings = load_data_from_pickle(preprocess_folder, 'ratings', log=False)
    global_mean = ratings['rating'].mean()

    # build arrays for rated movies
    rated_indices = [map_movie_to_id[m_id] for m_id in movie_ids]
    r = np.array(movie_ratings, dtype=float)          # (n,)
    b_i = movie_biases[rated_indices]                 # (n,)
    n_rated = len(rated_indices)

    # compute new user bias b_u 
    residuals = r - (global_mean + b_i)               # (n,)
    user_bias = LAMBDA * residuals.sum() / (LAMBDA * n_rated + GAMMA)

    # predict score for all movies
    scores = global_mean + movie_biases + user_bias   # (num_movies,)

    # don't recommend already-rated movies
    scores[rated_indices] = -np.inf

    # top-N
    top_indices = np.argsort(scores)[-top_n:][::-1]

    top_movie_ids = [map_id_to_movie[i] for i in top_indices]
    top_titles = [map_movie_to_title[m_id] for m_id in top_movie_ids]

    return top_titles

