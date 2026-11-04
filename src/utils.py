#!/usr/bin/env python3

"""
Utilities for the MovieLens alternating least squares recommender.

This module provides helpers for:

* saving and loading model artifacts;
* downloading and preprocessing MovieLens ratings and metadata;
* creating and preprocessing interaction-level train/test splits;
* visualizing dataset statistics and model diagnostics; and
* generating recommendations for the users not present during training.

The public function signatures are retained for compatibility with 
existing experiments. Artifacts use the project's existing directory layout:

    data/<dataset>/raw/
    data/<dataset>/preprocess
    experiments/<dataset>/data_split/
    experiments/<dataset>/model_files/
    experiments/<dataset>/images/
"""

from __future__ import annotations

import logging
import pickle
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Plot settings: applied locally to avoid changing matplotlib's global
# configuration merely by importing this utility module.
FONT_SIZE = 18
PLOT_STYLE = {
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE
}

# Default settings
DEFAULT_EMBEDDING_MOVIE_IDS = (
    1,      # Toy Story (1995)
    2116,   # The Lord of the Rings (1978)
    3114,   # Toy Story 2 (1999)
    4993,   # The Lord of the Rings: The Fellowship of the Ring (2001)
    5952,   # The Lord of the Rings: The Two Towers (2002)
    45722,  # Pirates of the Caribbean: Dead Man's Chest (2006)
    63992,  # Twilight (2008)
    72407,  # The Twilight Saga: New Moon (2009)
    78772,  # The Twilight Saga: Eclipse (2010)
    78499,  # Toy Story 3 (2010)
    91104,  # The Twilight Saga: Breaking Dawn - Part 1 (2011)
    61160,  # Star Wars: The Clone Wars (2008)
    4886,   # Monsters, Inc. (2001)
)

# Weight assigned to the movie bias when ranking recommendations.
# A value below 1 reduces the influence of generally popular movies.
DEFAULT_BIAS_WEIGHT = 0.05

PICKLE_PROTOCOL = 5

#################### Paths and experiment Logging ########################

def _dataset_paths(base_path: str | Path, dataset_name: str) -> tuple[Path, Path]:
    """
    Return the dataset's raw-data and preprocessing directories.
    """
    dataset_dir = Path(base_path) / dataset_name
    return dataset_dir / "raw", dataset_dir / "preprocess"

def _experiment_paths(experiment_path: str | Path, dataset_name: str) -> tuple[Path, Path, Path]:
    """
    Return the data-split, model, and image directories.
    """
    experiment_dir = Path(experiment_path) /dataset_name
    return (
        experiment_dir / "data_split",
        experiment_dir / "model_files",
        experiment_dir / "images"
    )

def setup_logging(experiment_folder: str, filename: str) -> logging.Logger:
    """
    Configure file and console logging for an experiment.

    Parameters
    ----------
    experiment_folder : str
        Directory where the log file will be created.
    filename : str
        Log-file name, including any extension.

    Returns
    -------
    logging.Logger
        Configured root logger, also used by existing model code.
    """
    log_dir = Path(experiment_folder)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate experiment handlers without deleting handlers installed
    # by notebooks or other libraries.
    for handler in logger.handlers[:]:
        if getattr(handler, "_movielens_experiment_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_dir / filename, mode="w")
    console_handler = logging.StreamHandler(sys.stdout)

    for handler in (file_handler, console_handler):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        setattr(handler, "_movielens_experiment_handler", True)
        logger.addHandler(handler)  
    return logger

############################## Serialization #################################

def save_data_to_pickle(save_path: str, data: Any, filename: str, log: bool = True) -> None:
    """
    Serialize a Python object using pickle protocol 5.
    
    Parameters
    ----------
    save_path : str
        Destination directory; created if necessary.
    data : Any
        Object to serialize.
    filename : str
        Output name without the ``.pkl`` extension.
    log : bool, default=True
        Whether to print a confirmation message.
    """
    directory = Path(save_path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{filename}.pkl").open("wb") as file:
        pickle.dump(data, file, protocol=PICKLE_PROTOCOL)
    if log:
        print(f'Saved {filename} to .pkl file at {directory}.')

def load_data_from_pickle(save_path: str, filename: str, log: bool = True) -> Any:
    """
    Load a previously serialized Python object from a pickle file.
    
    Parameters
    ----------
    save_path : str
        Directory containing the serialized object.
    filename : str
        Filename without the ``.pkl`` extension.
    log : bool, default=True
        Whether to print a confirmation message.
    
    Returns
    -------
    Any
        Deserialized Python object.
    """
    path = Path(save_path) / f"{filename}.pkl"
    with path.open("rb") as file:
        data = pickle.load(file)

    if log:
        print(f"Loaded {filename} from .pkl file at {save_path}")
    return data

def save_numpy_array(save_path: str, data: np.ndarray, filename: str, log: bool = True):
    """
    Save an array to a Numpy ``.npy`` file.
    
    Parameters
    ----------
    save_path : str
        Destination directory; created if necessary.
    data : np.ndarray
        Array to save. Object arrays are supported for the project's 
        variable-length user/movie rating list.
    filename : str
        Output name without the ``.npy`` extension.
    log : bool, default=True
        Whether to print a confirmation message.
    """
    directory = Path(save_path)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f'{filename}.npy', data, allow_pickle=True)

    if log:
        print(f'Saved {filename} to .npy file at {directory}')

def load_numpy_array(save_path: str, filename: str, log: bool = True) -> np.ndarray:
    """
    Load a NumPy array from file.
    
    Parameters
    ----------
    save_path : str
        Directory containing the save array.
    filename : str
        Output name without the ``.npy`` extension.
    log : bool, default=True
        Whether to print a confirmation message.
    
    Returns
    -------
    numpy.ndarray
        Loaded array.
    """
    path = Path(save_path) / f"{filename}.npy"
    data = np.load(path, allow_pickle=True)

    if log:
        print(f'Loaded {filename} from .npy file at {save_path}')
    return data

###################### MovieLens Data Loading and Preprocessing ########################

def _download_movielens_dataset(raw_folder: Path, dataset_name: str) -> None:
    """
    Download and safely unpack the names MovieLens ZIP archive.

    Only files directly within the archive's ``<dataset_name>/`` directory 
    are extracted. This avoids arbitrary path traversal during extraction.
    """
    raw_folder.mkdir(parents=True, exist_ok=True)
    url = f"https://files.grouplens.org/datasets/movielens/{dataset_name}.zip"
    archive_path = raw_folder / f"{dataset_name}.zip"

    print(f"Downloading {dataset_name} from MovieLens website...")
    urllib.request.urlretrieve(url, archive_path)

    print("Download complete. Extracting files...")
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member.is_dir():
                continue
            if (
                len(member_path.parts) != 2
                or member_path.parts[0] != dataset_name
                or member_path.parts[1] in (".", "..")
            ):
                continue

            destination = raw_folder / member_path.parts[1]
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1

    if not extracted or not (raw_folder / "ratings.csv").is_file():
        raise FileNotFoundError(
            f"The archive for {dataset_name!r} did not contain ratings.csv "
            "in the expected directory structure."
        )

def load_ratings_data(base_path: str, dataset_name: str) -> None:
    """
    Prepare MovieLens ratings and create internal identifier mappings.

    If ``ratings.csv`` is absent, the corresponding MovieLens archive is 
    downloaded from GroupLens. Both the original ratings and four mappings 
    are saved as pickle files under ``data/<dataset>/preprocess``.

    Internal IDs are zero-based and follow the first-occurrence order of users 
    and movies in the input data.

    Parameters
    ----------
    base_path : str
        Root directory for downloaded and preprocessed datsets.
    dataset_name : str
        Movielens archive name, e.g. ``ml-latest-small`` or ``ml-32m``.

    Returns
    -------
    None
        Prepared artifacts are written to disk.
    """
    raw_folder, preprocess_folder = _dataset_paths(base_path, dataset_name)
    raw_folder.mkdir(parents=True, exist_ok=True)
    preprocess_folder.mkdir(parents=True, exist_ok=True)

    ratings_file = raw_folder / "ratings.csv"
    if not ratings_file.is_file():
        _download_movielens_dataset(raw_folder, dataset_name)

    print(f"Loading ratings from {ratings_file}...")
    ratings = pl.read_csv(ratings_file)

    # Preserve existing first-seen ordering to keep stored model indexes stable.
    # A stable unique scan is cheaper than building grouped rating lists just
    # to discover distinct IDs, especially for MovieLens 32M.
    user_ids = ratings["userId"].unique(maintain_order=True).to_list()
    movie_ids = ratings["movieId"].unique(maintain_order=True).to_list()

    map_user_to_id = {user_id: idx for idx, user_id in enumerate(user_ids)}
    map_movie_to_id = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    map_id_to_user = {idx: user_id for idx, user_id in enumerate(user_ids)}
    map_id_to_movie = {idx: movie_id for idx, movie_id in enumerate(movie_ids)}

    for name, data in (
        ("map_user_to_id", map_user_to_id),
        ("map_movie_to_id", map_movie_to_id),
        ("map_id_to_user", map_id_to_user),
        ("map_id_to_movie", map_id_to_movie),
        ("ratings", ratings)
    ):
        save_data_to_pickle(preprocess_folder, data, name)

    # Delete files to avoid overloading memory
    del ratings,  map_id_to_movie, map_id_to_user, map_user_to_id, map_movie_to_id

def load_movie_data(base_path: str, dataset_name: str) -> None:
    """
    Save MovieLens movie metadata and ID-to-title mapping.

    Parameters
    ----------
    base_path : str
        Root data directory.
    dataset_name : str
        Name of the MovieLens dataset.

    Returns
    -------
    None
        Prepared artifacts are written to disk.

    Notes
    -----
    Expects ``movies.csv`` and ``links.csv`` in the dataset's ``raw`` folder.
    Missing metadata files trigger a fresh MovieLens archive download.
    """
    raw_folder, preprocess_folder = _dataset_paths(base_path, dataset_name)
    preprocess_folder.mkdir(parents=True, exist_ok=True)

    if not (raw_folder / "movies.csv").is_file() or not (
        raw_folder / "links.csv"
    ).is_file():
        _download_movielens_dataset(raw_folder, dataset_name)

    print(f"Loading movie metadata from {raw_folder}...")
    movies = pl.read_csv(raw_folder / "movies.csv")
    links = pl.read_csv(raw_folder / "links.csv")
    map_movie_to_title = dict(
        zip(movies["movieId"].to_list(), movies["title"].to_list())
    )

    for name, data in (
        ("map_movie_to_title", map_movie_to_title),
        ("movies", movies),
        ("links", links)
    ):
        save_data_to_pickle(preprocess_folder, data, name)

    del movies, map_movie_to_title, links

def _group_ratings(
    ratings: pl.DataFrame, group_column: str, item_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group ratings, returning group IDs  and paired object-array lists.

    The returned second array has two columns: lists of neighbor IDs and the corresponding 
    rating lists, retaining the format expected by ``models.py``.
    """
    grouped = ratings.group_by(group_column, maintain_order=True).agg(
        pl.col(item_column), pl.col("rating")
    )
    return grouped[:, 0].to_numpy(), grouped[:, 1 :].to_numpy()

def split_into_train_test(
    base_path: str, experiment_path: str, dataset_name: str,
    seed: int = 42, test_size: float = .2
) -> None:
    """
    Split rating interactions into training and test artifacts.

    Each rating is assigned independently using a NumPy random generator.
    This is **not** a per-user stratified split: a user or movie can have no
    training ratings, particularly in sparse datasets. The expected test
    fraction is ``test_size``; the realized fraction may differ slightly.

    Parameters
    ----------
    base_path : str
        Root directory containing preprocessed ratings and ID mappings.
    experiment_path : str
        Root directory for experiment results.
    dataset_name : str
        Name of the MovieLens dataset.
    seed : int, default=42
        Seed for reproducible interaction assignments.
    test_size : float, default=0.2
        Probability that a rating is assigned to the test set. Must be
        strictly between 0 and 1.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")

    _, preprocess_folder = _dataset_paths(base_path, dataset_name)
    data_split_folder, _, _ = _experiment_paths(experiment_path, dataset_name)
    data_split_folder.mkdir(parents=True, exist_ok=True)

    ratings = load_data_from_pickle(preprocess_folder, "ratings", log=False)
    movie_id_map = load_data_from_pickle(preprocess_folder, "map_movie_to_id", log=False)
    user_id_map = load_data_from_pickle(preprocess_folder, "map_user_to_id", log=False)

    ratings = ratings.with_columns(
        pl.col("userId").replace_strict(user_id_map).alias("userIdx"),
        pl.col("movieId").replace_strict(movie_id_map).alias("movieIdx")
    )

    # One independent draw per interaction, independent of its user/movie ID.
    rng = np.random.default_rng(seed)
    ratings = ratings.with_columns(
        pl.Series("split", rng.uniform(size=ratings.height))
    )
    train_df = ratings.filter(pl.col("split") > test_size).drop("split")
    test_df = ratings.filter(pl.col("split") <= test_size).drop("split")

    users_indices_train, users_train = _group_ratings(train_df, "userIdx", "movieIdx")
    movies_indices_train, movies_train = _group_ratings(train_df, "movieIdx", "userIdx")
    users_indices_test, users_test = _group_ratings(test_df, "userIdx", "movieIdx")
    movies_indices_test, movies_test = _group_ratings(test_df, "movieIdx", "userIdx")
    train_global_mean = train_df["rating"].mean()

    for name, array in (
        ("users_train", users_train),
        ("users_test", users_test),
        ("movies_train", movies_train),
        ("movies_test", movies_test),
        ("users_indices_train", users_indices_train),
        ("users_indices_test", users_indices_test),
        ("movies_indices_train", movies_indices_train),
        ("movies_indices_test", movies_indices_test),
        ("train_global_mean", np.array(train_global_mean))
    ):
        save_numpy_array(data_split_folder, array, name)

    # Delete variables to free memory
    del ratings, test_df, train_df, users_test, users_train, rng
    del movie_id_map, user_id_map, movies_train, movies_test
    del users_indices_train, users_indices_test, movies_indices_test, movies_indices_train

######################## Data and Model Visualization #############################

def _save_and_show(fig: plt.Figure, filename: Path, *, show: bool = False) -> None:
    """
    Save a figure as PDF and PNG, optionally display it, and close it.
    """
    pdf_path = Path(filename).with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
    finally:
        plt.close(fig)

def plot_distribution(
    base_path: str, experiment_path: str,
    dataset_name: str, title: str,
    xlabel: str, ylabel: str,
    color: str = "g", kind: str = "small",
    show: bool = False
) -> None:
    """
    Plot the distribution of average ratings across movies.

    Parameters
    ----------
    base_path : str
        Root data directory.
    experiment_path : str
        Root experiment directory.
    dataset_name : str
        MovieLens dataset name.
    title, xlabel, ylabel : str
        Figure title and axis labels.
    color : str, default='g'
        Color of the histogram and kernel density estimate.
    kind : str, default='small'
        Selects KDE smoothing: ``small`` uses bandwidth adjustment 2;
        any other value uses 3, matching the original implementation.
    show : bool, default=False
        Display the figure if True. Otherwise, save without displaying.

    Notes
    -----
    The observations are *per-movie mean ratings*, not individual ratings.
    The PDF is stored in the experiment's ``images`` directory.
    """
    _, preprocess_folder = _dataset_paths(base_path, dataset_name)
    _, _, image_folder = _experiment_paths(experiment_path, dataset_name)
    ratings = load_data_from_pickle(preprocess_folder, "ratings", log=False)

    mean_ratings = (
        ratings.group_by("movieId")
        .agg(pl.col("rating").mean().alias("avg_rating"))
        .select("avg_rating")
        .to_numpy()
        .ravel()
    )
    bw_adjust = 2 if kind == "small" else 3

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            mean_ratings, bins=10, density=True, color=color,
            edgecolor="black", alpha=0.7
        )
        sns.kdeplot(mean_ratings, bw_adjust=bw_adjust, color=color, ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        _save_and_show(fig, image_folder / f"ratings_distribution_{dataset_name}", show=show)

def plot_power_law_distribution(
    base_path: str, experiment_path: str,
    dataset_name: str, title: str,
    xlabel: str, ylabel: str,
    show: bool = False
) -> None:
    """
    Plot empirical user and movie rating-count frequencies on log-log axes.

    Parameters
    ----------
    base_path : str
        Root data directory.
    experiment_path : str
        Root experiment directory.
    dataset_name : str
        MovieLens dataset name.
    title, xlabel, ylabel : str
        Figure title and axis labels.
    show : bool, default=False
        Display the figure if True. Otherwise, save without displaying.
    """
    _, preprocess_folder = _dataset_paths(base_path, dataset_name)
    _, _, image_folder = _experiment_paths(experiment_path, dataset_name)
    ratings = load_data_from_pickle(preprocess_folder, "ratings", log=False)

    movie_degree = (
        ratings.group_by("movieId")
        .agg(pl.len().alias("movie_ratings"))
        .get_column("movie_ratings")
        .to_numpy()
    )
    user_degree = (
        ratings.group_by("userId")
        .agg(pl.len().alias("user_ratings"))
        .get_column("user_ratings")
        .to_numpy()
    )
    movie_degrees, movie_freq = np.unique(movie_degree, return_counts=True)
    user_degrees, user_freq = np.unique(user_degree, return_counts=True)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(movie_degrees, movie_freq, ".", color="g", label="Movie")
        ax.loglog(user_degrees, user_freq, ".", color="b", label="User")
        ax.axvline(20, linestyle="--", color="r")
        ax.text(21, max(movie_freq), "Number of ratings = 20", color="r")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        _save_and_show(fig, image_folder / f"power_law_{dataset_name}", show=show)

def plot_loss_and_rmse(
    experiment_path: str, dataset_name: str,
    loss_history: Sequence[float] | np.ndarray,
    rmse_history: dict[str, Sequence[float]],
    model_type: str, add_title: bool = False,
    show: bool = False
) -> None:
    """
    Save training loss and train/test RMSE curves as separate PDF figures.

    Parameters
    ----------
    experiment_path : str
        Root experiment directory.
    dataset_name : str
        MovieLens dataset name.
    loss_history : sequence of float or numpy.ndarray
        Training loss at each iteration.
    rmse_history : dict
        Dictionary containing ``'train'`` and ``'test'`` RMSE sequences.
    model_type : str
        ``'bias'`` uses ``_bias`` file-name suffix; other
        values use filenames without that suffix.
    add_title : bool, default=False
        Whether to display descriptive figure titles.
    show : bool, default=False
            Display the figure if True. Otherwise, save without displaying.
    """
    _, _, image_folder = _experiment_paths(experiment_path, dataset_name)
    suffix = "_bias" if model_type == "bias" else ""

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = np.arange(1, len(loss_history) + 1)

        ax.plot(epochs, loss_history, label="Train loss", color="blue")
        ax.set(xlabel="Epoch", ylabel="Loss")

        if add_title:
            ax.set_title("Training Loss over Iterations")
        ax.legend()
        _save_and_show(
            fig, image_folder / f"train_loss_plot_{dataset_name}{suffix}",
            show=show
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, rmse_history["train"], label="Train RMSE", color="red")
        ax.plot(epochs, rmse_history["test"], label="Test RMSE", color="blue")
        ax.set(xlabel="Epoch", ylabel="RMSE")

        if add_title:
            ax.set_title("Root Mean Squared Error (RMSE) over Iterations")
        ax.legend()
        _save_and_show(
            fig, image_folder / f"train_rmse_plot_{dataset_name}{suffix}",
            show=show
        )

def plot_embeddings(
    base_path: str,
    experiment_path: str,
    dataset_name: str,
    use_pca: bool = False,
    movie_ids: Sequence[int] = DEFAULT_EMBEDDING_MOVIE_IDS,
    show: bool = False
) -> None:
    """
    Visualize selected learned movie embeddings in two dimensions.

    Parameters
    ----------
    base_path : str
        Root data directory with movie-ID mappings and titles.
    experiment_path : str
        Root experiment directory with saved model embeddings.
    dataset_name : str
        MovieLens dataset name.
    use_pca : bool, default=False
        Apply principal component analysis if True, otherwise use t-SNE.
    movie_ids : sequence of int, optional
        External MovieLens IDs to label. Defaults to a fixed illustrative
        selection from the original experiment.
    show : bool, default=False
        Display the figure if True. Otherwise, save without displaying.
    """
    _, preprocess_folder = _dataset_paths(base_path, dataset_name)
    _, model_folder, image_folder = _experiment_paths(
        experiment_path, dataset_name
    )
    movie_id_map = load_data_from_pickle(
        preprocess_folder, "map_movie_to_id", log=False
    )
    movie_titles = load_data_from_pickle(
        preprocess_folder, "map_movie_to_title", log=False
    )
    embeddings = load_numpy_array(model_folder, "movie_embeddings", log=False)

    movie_ids = tuple(movie_ids)
    movie_indices = [movie_id_map[movie_id] for movie_id in movie_ids]
    selected = embeddings[movie_indices]

    if len(movie_indices) < 2:
        raise ValueError("At least two movies are needed for a 2D plot.")

    if use_pca:
        projected = (
            selected.copy() if selected.shape[1] == 2
            else PCA(n_components=2).fit_transform(selected)
        )
        method = "pca"
    else:
        projected = TSNE(
            n_components=2,
            perplexity=min(10, len(movie_indices) - 1),
            learning_rate="auto",
            init="random",
            random_state=42,
        ).fit_transform(selected)
        method = "tsne"

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(projected[:, 0], projected[:, 1], color="blue", s=10)
        ax.set(xlabel="Embedding Dimension 1", ylabel="Embedding Dimension 2")
        for index, movie_id in enumerate(movie_ids):
            x_coord, y_coord = projected[index]
            ax.text(x_coord, y_coord, movie_titles[movie_id], fontsize=13)

        _save_and_show(
            fig,
            image_folder / f"2D_embedding_plot_{dataset_name}_{method}.pdf",
            show=show
        )

################### Cold-start Recommendations for New Users #######################

def _validate_rated_movies(
    movie_ids: Sequence[int], movie_ratings: Sequence[float], top_n: int
) -> np.ndarray:
    """
    Validate aligned input ratings and return a numeric ratings vector.
    """
    if len(movie_ids) == 0:
        raise ValueError("At least one movie rating is required.")
    if len(movie_ids) != len(movie_ratings):
        raise ValueError("movie_ids and movie_ratings must have equal lengths.")
    if len(set(movie_ids)) != len(movie_ids):
        raise ValueError("Each movie ID should occur only once.")
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")
    ratings = np.asarray(movie_ratings, dtype=float)
    if not np.all(np.isfinite(ratings)):
        raise ValueError("All movie ratings must be finite numbers.")
    return ratings

def _select_unrated_top_n(
    scores: np.ndarray, rated_indices: Sequence[int], top_n: int
) -> np.ndarray:
    """
    Return descending score indices excluding movies already rated.
    """
    candidates = np.asarray(scores, dtype=float).copy()
    candidates[list(rated_indices)] = -np.inf
    n_available = len(candidates) - len(set(rated_indices))
    if n_available == 0:
        return np.empty(0, dtype=int)
    return np.argsort(candidates)[-min(top_n, n_available):][::-1]

def _recommend_factor_titles(
    base_path: str, experiment_path: str,
    dataset_name: str, movie_ids: Sequence[int],
    movie_ratings: Sequence[float], top_n: int,
    lambda_reg: float, gamma: float, tau: float,
) -> list[str]:
    """
    Estimate a new user factor/bias and rank previously unseen movies.

    With item factors E, item biases b, observed ratings r, new-user factor
    vector u, and new-user bias a, alternate these closed-form updates five
    times (matching the original implementation)::

        u <- solve(lambda_reg * E.T @ E + tau * I,
                   lambda_reg * E.T @ (r - b - a))
        a <- lambda_reg * sum(r - E @ u - b) / (lambda_reg * n + gamma)
    """
    ratings = _validate_rated_movies(movie_ids, movie_ratings, top_n)
    _, preprocess_folder = _dataset_paths(base_path, dataset_name)
    _, model_folder, _ = _experiment_paths(experiment_path, dataset_name)

    movie_id_map = load_data_from_pickle(preprocess_folder, "map_movie_to_id", log=False)
    movie_titles = load_data_from_pickle(preprocess_folder, "map_movie_to_title", log=False)
    internal_id_map = load_data_from_pickle(preprocess_folder, "map_id_to_movie", log=False)
    embeddings = load_numpy_array(model_folder, "movie_embeddings", log=False)
    movie_biases = load_numpy_array(model_folder, "movie_biases", log=False).reshape(-1)

    rated_indices = [movie_id_map[movie_id] for movie_id in movie_ids]
    rated_embeddings = embeddings[rated_indices]
    rated_biases = movie_biases[rated_indices]
    n_rated, n_factors = rated_embeddings.shape

    user_embedding = np.zeros(n_factors)
    user_bias = 0.0
    system_matrix = lambda_reg * (rated_embeddings.T @ rated_embeddings) + (
        tau * np.eye(n_factors)
    )

    for _ in range(5):
        rhs = lambda_reg * rated_embeddings.T @ (ratings - rated_biases - user_bias)
        user_embedding = np.linalg.solve(system_matrix, rhs)
        residuals = ratings - rated_embeddings @ user_embedding - rated_biases
        user_bias = lambda_reg * residuals.sum() / (lambda_reg * n_rated + gamma)

    scores = (
        embeddings @ user_embedding
        + user_bias
        + DEFAULT_BIAS_WEIGHT * movie_biases
    )
    top_indices = _select_unrated_top_n(scores, rated_indices, top_n)
    return [movie_titles[internal_id_map[int(idx)]] for idx in top_indices]

def _recommend_bias_titles(
    base_path: str,
    experiment_path: str,
    dataset_name: str,
    movie_ids: Sequence[int],
    movie_ratings: Sequence[float],
    top_n: int,
    lam: float,
    gamma: float,
) -> list[str]:
    """
    Estimate a new user bias using the original bias-only model formula.
    """
    ratings_vector = _validate_rated_movies(movie_ids, movie_ratings, top_n)
    _, preprocess_folder = _dataset_paths(base_path, dataset_name)
    data_split_folder, model_folder, _ = _experiment_paths(experiment_path, dataset_name)

    movie_id_map = load_data_from_pickle(preprocess_folder, "map_movie_to_id", log=False)
    movie_titles = load_data_from_pickle(preprocess_folder, "map_movie_to_title", log=False)
    internal_id_map = load_data_from_pickle(preprocess_folder, "map_id_to_movie", log=False)
    movie_biases = load_numpy_array(model_folder, "movie_biases_b", log=False).reshape(-1)
    global_mean = float(load_numpy_array(data_split_folder, "train_global_mean", log=False))

    rated_indices = [movie_id_map[movie_id] for movie_id in movie_ids]
    residuals = ratings_vector - (global_mean + movie_biases[rated_indices])
    user_bias = lam * residuals.sum() / (lam * len(rated_indices) + gamma)

    scores = global_mean + movie_biases + user_bias
    top_indices = _select_unrated_top_n(scores, rated_indices, top_n)
    return [movie_titles[internal_id_map[int(idx)]] for idx in top_indices]

def _print_recommendations(
    movie_ids: Sequence[int],
    movie_ratings: Sequence[float],
    titles: Sequence[str],
    base_path: str,
    dataset_name: str,
) -> None:
    """Print ratings supplied by the user and ranked movie titles."""
    _, preprocess_folder = _dataset_paths(base_path, dataset_name)
    movie_titles = load_data_from_pickle(
        preprocess_folder, "map_movie_to_title", log=False
    )
    print("Since you liked:")
    for movie_id, rating in zip(movie_ids, movie_ratings):
        print(f"  - '{movie_titles[movie_id]}' (rating: {rating})")
    print("\nYou may also like:\n")
    for index, title in enumerate(titles, start=1):
        print(f"{index}. {title}")

def recommend_movies_streamlit(
    base_path: str,
    experiment_path: str,
    dataset_name: str,
    movie_ids: list,
    movie_ratings: list,
    top_n: int,
    lambda_reg: float,
    gamma: float,
    tau: float,
) -> list[str]:
    """
    Return top-N latent-factor recommendations for a new user.

    Parameters
    ----------
    base_path, experiment_path : str
        Root data and experiment directories, respectively.
    dataset_name : str
        MovieLens dataset name.
    movie_ids : list of int
        External MovieLens IDs rated by the new user.
    movie_ratings : list of float
        Ratings in the same order as ``movie_ids``.
    top_n : int
        Maximum number of new movie titles to recommend.
    lambda_reg : float
        Data-fit precision/weight in the ALS update.
    gamma : float
        Regularization weight for the new user bias.
    tau : float
        Regularization weight for the new user factor vector.

    Returns
    -------
    list[str]
        Titles in descending predicted-score order, excluding rated movies.
    """
    return _recommend_factor_titles(
        base_path, experiment_path, dataset_name,
        movie_ids, movie_ratings, top_n, lambda_reg, gamma, tau,
    )

def recommend_movies(
    base_path: str,
    experiment_path: str,
    dataset_name: str,
    movie_ids: list,
    movie_ratings: list,
    top_n: int,
    lambda_reg: float,
    gamma: float,
    tau: float,
) -> None:
    """
    Print latent-factor recommendations for a new user.

    Uses the same closed-form estimation as
    :func:`recommend_movies_streamlit`, but prints the results rather than
    returning the titles. Parameters have the same meanings.
    """
    titles = _recommend_factor_titles(
        base_path, experiment_path, dataset_name,
        movie_ids, movie_ratings, top_n, lambda_reg, gamma, tau,
    )
    _print_recommendations(
        movie_ids, movie_ratings, titles, base_path, dataset_name
    )

def recommend_movies_bias_only(
    base_path: str,
    experiment_path: str,
    dataset_name: str,
    movie_ids: list,
    movie_ratings: list,
    top_n: int,
    lambda_reg: float,
    gamma: float,
) -> None:
    """Print recommendations from the fitted bias-only model.

    Parameters are analogous to :func:`recommend_movies_streamlit`, without
    ``tau`` because this model has no latent factor vector. See
    :func:`recommend_movies_bias_only_streamlit` for its return-value variant.
    """
    titles = _recommend_bias_titles(
        base_path, experiment_path, dataset_name,
        movie_ids, movie_ratings, top_n, lambda_reg, gamma,
    )
    _print_recommendations(
        movie_ids, movie_ratings, titles, base_path, dataset_name
    )

def recommend_movies_bias_only_streamlit(
    base_path: str,
    experiment_path: str,
    dataset_name: str,
    movie_ids: list,
    movie_ratings: list,
    top_n: int,
    lambda_reg: float,
    gamma: float,
) -> list[str]:
    """
    Return top-N recommendations from the fitted bias-only model.

    Parameters
    ----------
    base_path, experiment_path : str
        Root data and experiment directories, respectively.
    dataset_name : str
        MovieLens dataset name.
    movie_ids : list of int
        External MovieLens IDs rated by the new user.
    movie_ratings : list of float
        Ratings corresponding to ``movie_ids``.
    top_n : int
        Maximum number of titles to return.
    lambda_reg : float
        Data-fit weight used in the new-user bias estimate.
    gamma : float
        Regularization weight for the new-user bias.

    Returns
    -------
    list[str]
        Ranked unrated movie titles.
    """
    return _recommend_bias_titles(
        base_path, experiment_path, dataset_name,
        movie_ids, movie_ratings, top_n, lambda_reg, gamma,
    )
