#!/usr/bin/env python3
"""
Command-line training pipeline for the MovieLens ALS recommender.

The pipeline prepares MovieLens data, trains a bias-only baseline and/or a
bias-plus-latent-factors model, evaluates the observed-rating RMSE, and saves
model parameters and training histories.

Run from the repository root::

    python main.py --model both --dataset ml-latest-small

If this file is stored inside ``src/``, use::

    python -m src.main --model both --dataset ml-latest-small

Both training models intentionally follow the equations in ``src.models``:
there is no separately fitted global-mean intercept. Hyperparameters may be
supplied as explicit arguments or through saved Optuna ``best_params`` files.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src import models, utils

# Resolve default storage paths relative to the repository, not the terminal's
# current working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "src" else _SCRIPT_DIR

SPLIT_ARTIFACTS = (
    "users_train", "users_test", "movies_train", "movies_test",
    "users_indices_train", "users_indices_test",
    "movies_indices_train", "movies_indices_test",
)

@dataclass(frozen=True)
class SplitData:
    """
    Grouped user/movie ratings and their original mapping dimensions.

    Each grouped ``users_*`` or ``movies_*`` array has two columns containing
    neighbor-index arrays and aligned rating arrays. The associated
    ``*_indices_*`` arrays identify which row corresponds to which global ID.
    """

    users_train: np.ndarray
    users_test: np.ndarray
    movies_train: np.ndarray
    movies_test: np.ndarray
    users_indices_train: np.ndarray
    users_indices_test: np.ndarray
    movies_indices_train: np.ndarray
    movies_indices_test: np.ndarray
    n_users: int
    n_movies: int

class RecommendationModel:
    """
    Coordinate preprocessing and ALS training for a MovieLens dataset.

    Parameters
    ----------
    n_factors : int, default=16
        Latent-factor dimension for matrix factorization.
    lambda_reg : float, default=0.216
        Positive coefficient weighting squared rating errors.
    gamma : float, default=0.0093
        Nonnegative L2 penalty on user and movie biases.
    tau : float, default=0.5092
        Positive L2 penalty on user and movie latent vectors.
    n_epochs : int, default=20
        Number of alternating update sweeps.
    jitter : float, default=1e-8
        Legacy argument retained for compatibility. It is *not* applied to
        the ALS equations; positive ``tau`` stabilizes the factor solves.
    seed : int, default=42
        Random seed for both the interaction split and factor initialization.
    dataset_name : str, default="ml-latest-small"
        Name of the MovieLens archive used by ``utils.py``.
    dataset_dir : str or pathlib.Path, optional
        Root folder containing ``<dataset_name>/raw`` and ``preprocess``.
    experiment_dir : str or pathlib.Path, optional
        Root folder for data splits, logs, model files, and plots.
    test_size : float, default=0.2
        Probability of assigning an interaction to the test set.
    """

    def __init__(
        self,
        n_factors: int = 16,
        lambda_reg: float = 0.216,
        gamma: float = 0.0093,
        tau: float = 0.5092,
        n_epochs: int = 20,
        jitter: float = 1e-8,
        seed: int = 42,
        dataset_name: str = "ml-latest-small",
        dataset_dir: str | Path = PROJECT_ROOT / "data",
        experiment_dir: str | Path = PROJECT_ROOT / "experiments",
        test_size: float = 0.2,
    ) -> None:
        if n_factors < 1 or n_epochs < 1:
            raise ValueError("n_factors and n_epochs must be positive integers.")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be strictly between 0 and 1.")
        if jitter < 0:
            raise ValueError("jitter must be nonnegative.")
        self._validate_parameters(lambda_reg, gamma, tau)

        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.tau = tau
        self.n_epochs = n_epochs
        self.jitter = jitter
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.test_size = test_size
        self.logger = logging.getLogger(__name__)
        self._split: SplitData | None = None

    @staticmethod
    def _validate_parameters(lambda_reg: float, gamma: float, tau: float) -> None:
        """
        Reject parameters that violate the assumed ALS objectives.
        """
        if not (np.isfinite(lambda_reg) and lambda_reg > 0):
            raise ValueError("LAMBDA must be positive and finite.")
        if not (np.isfinite(gamma) and gamma >= 0):
            raise ValueError("GAMMA must be nonnegative and finite.")
        if not (np.isfinite(tau) and tau > 0):
            raise ValueError("TAU must be positive and finite.")

    @property
    def experiment_path(self) -> Path:
        """Dataset-specific experiment output folder."""
        return self.experiment_dir / self.dataset_name

    @property
    def split_path(self) -> Path:
        """
        Directory containing the grouped training and test arrays.
        """
        return self.experiment_path / "data_split"

    @property
    def model_path(self) -> Path:
        """
        Directory containing learned parameters and training histories.
        """
        return self.experiment_path / "model_files"

    def preprocess_data(
        self, *, force: bool = False, plot_data: bool = True
    ) -> None:
        """
        Prepare rating mappings, movie metadata, and a reproducible split.

        Existing complete artifacts are reused. ``force=True`` rebuilds all
        artifacts together so that data mappings and split IDs stay aligned.
        A different requested seed or test fraction also regenerates the split
        when a recorded split configuration is available.

        Parameters
        ----------
        force : bool, default=False
            Recreate downloaded-data-derived artifacts and the data split.
        plot_data : bool, default=False
            Save exploratory dataset plots to ``experiments/<dataset>/images``.
            Plotting is optional to keep CLI training usable in headless runs.
        """
        start = time.perf_counter()
        preprocess_path = self.dataset_dir / self.dataset_name / "preprocess"
        ratings_outputs = (
            "ratings", "map_user_to_id", "map_movie_to_id",
            "map_id_to_user", "map_id_to_movie",
        )
        movie_outputs = ("movies", "links", "map_movie_to_title")

        rebuild_ratings = force or any(
            not (preprocess_path / f"{name}.pkl").is_file()
            for name in ratings_outputs
        )
        if rebuild_ratings:
            self.logger.info("Preparing ratings and index mappings...")
            utils.load_ratings_data(str(self.dataset_dir), self.dataset_name)

        if force or any(
            not (preprocess_path / f"{name}.pkl").is_file()
            for name in movie_outputs
        ):
            self.logger.info("Preparing movie metadata...")
            utils.load_movie_data(str(self.dataset_dir), self.dataset_name)

        split_config_path = self.split_path / "split_config.json"
        split_files_missing = any(
            not (self.split_path / f"{name}.npy").is_file()
            for name in SPLIT_ARTIFACTS
        )
        prior_config: dict[str, Any] | None = None
        if split_config_path.is_file():
            with split_config_path.open("r", encoding="utf-8") as file:
                prior_config = json.load(file)

        split_changed = prior_config is not None and (
            prior_config.get("seed") != self.seed
            or prior_config.get("test_size") != self.test_size
        )
        rebuild_split = (
            force or rebuild_ratings or split_files_missing
            or split_changed or prior_config is None
        )
        if rebuild_split:
            self.logger.info(
                "Creating interaction split (seed=%d, test_size=%.3f)...",
                self.seed, self.test_size,
            )
            utils.split_into_train_test(
                str(self.dataset_dir), str(self.experiment_dir),
                self.dataset_name, seed=self.seed, test_size=self.test_size,
            )
            self.split_path.mkdir(parents=True, exist_ok=True)
            with split_config_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {"seed": self.seed, "test_size": self.test_size},
                    file, indent=2,
                )

        if plot_data:
            self.logger.info("Generating exploratory data plots...")
            utils.plot_distribution(
                str(self.dataset_dir), str(self.experiment_dir),
                self.dataset_name, title="", xlabel="Mean movie rating",
                ylabel="Density",
                kind="small" if self.dataset_name == "ml-latest-small" else "large",
            )
            utils.plot_power_law_distribution(
                str(self.dataset_dir), str(self.experiment_dir),
                self.dataset_name, title="", xlabel="Number of ratings",
                ylabel="Frequency",
            )

        self._split = None  # Invalidate previously loaded arrays after any split.
        self.logger.info(
            "Data preparation completed in %.2f seconds.",
            time.perf_counter() - start,
        )

    def _load_split(self) -> SplitData:
        """
        Load the eight grouped split arrays and full-size ID mappings.
        """
        if self._split is not None:
            return self._split

        missing = [
            str(self.split_path / f"{name}.npy") for name in SPLIT_ARTIFACTS
            if not (self.split_path / f"{name}.npy").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Data split is missing. Run with --preprocess or prepare data "
                "first. Missing files: " + ", ".join(missing)
            )

        arrays = {
            name: utils.load_numpy_array(str(self.split_path), name, log=False)
            for name in SPLIT_ARTIFACTS
        }
        preprocess_path = self.dataset_dir / self.dataset_name / "preprocess"
        n_users = len(utils.load_data_from_pickle(
            str(preprocess_path), "map_user_to_id", log=False
        ))
        n_movies = len(utils.load_data_from_pickle(
            str(preprocess_path), "map_movie_to_id", log=False
        ))
        self._split = SplitData(**arrays, n_users=n_users, n_movies=n_movies)

        if (len(self._split.users_indices_train) == 0
                or len(self._split.users_indices_test) == 0):
            raise ValueError("Training and test splits must both have ratings.")
        self.logger.info(
            "Loaded grouped split for %d users and %d movies.",
            n_users, n_movies,
        )
        return self._split

    def train_model(
        self, model: str, *, params: Mapping[str, float] | None = None,
        plot_training: bool = True,
    ) -> dict[str, Any]:
        """
        Train an ALS model, log metrics, and save all resulting artifacts.

        Parameters
        ----------
        model : {"Bias-Only", "Matrix Factorization"}
            Model family to train. Uses the exact model equations in
            ``src.models``, without a separate global-mean intercept.
        params : mapping, optional
            Optional model-specific values under the keys ``lambda_ref``,
            ``gamma``, and (for matrix factorization) ``tau``.
        plot_training : bool, default=False
            Save loss and RMSE plots after training.

        Returns
        -------
        dict
            Final training/test RMSE, loss, elapsed time, and configuration.
        """
        if model not in ("Bias-Only", "Matrix Factorization"):
            raise ValueError(f"Unknown model type: {model!r}.")
        config = {
            "lambda_reg": float(self.lambda_reg),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
        }
        if params is not None:
            unknown = set(params).difference(config)
            if unknown:
                raise ValueError(f"Unknown hyperparameter keys: {sorted(unknown)}")
            config.update({key: float(value) for key, value in params.items()})
        self._validate_parameters(config["lambda_reg"], config["gamma"], config["tau"])

        split = self._load_split()
        self.model_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        self.logger.info("Training %s for %d epochs...", model, self.n_epochs)
        self.logger.info("Hyperparameters: %s", config)

        if model == "Bias-Only":
            history, arrays = self._train_bias_only(split, config)
            suffix = "bias"
        else:
            history, arrays = self._train_factor_model(split, config)
            suffix = "bias_factors"

        elapsed = time.perf_counter() - start
        final_metrics: dict[str, Any] = {
            "model": model,
            "dataset": self.dataset_name,
            "n_epochs": self.n_epochs,
            "n_factors": self.n_factors if model == "Matrix Factorization" else None,
            "seed": self.seed,
            "hyperparameters": {
                key: value for key, value in config.items()
                if key != "tau" or model == "Matrix Factorization"
            },
            "final_loss": history["loss"][-1],
            "train_rmse": history["rmse"]["train"][-1],
            "test_rmse": history["rmse"]["test"][-1],
            "training_time_seconds": round(elapsed, 3),
        }

        # Preserve filenames consumed by the recommendation and plotting code.
        for filename, array in arrays.items():
            utils.save_numpy_array(str(self.model_path), array, filename, log=False)
        utils.save_data_to_pickle(
            str(self.model_path), history["loss"],
            f"{suffix}_loss_history", log=False,
        )
        utils.save_data_to_pickle(
            str(self.model_path), history["rmse"],
            f"{suffix}_rmse_history", log=False,
        )
        with (self.model_path / f"training_summary_{suffix}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(final_metrics, file, indent=2)

        if plot_training:
            utils.plot_loss_and_rmse(
                str(self.experiment_dir),
                self.dataset_name, history["loss"], history["rmse"],
                model_type="bias" if model == "Bias-Only" else "bias_factors",
                add_title=True,
            )

        self.logger.info(
            "%s: train RMSE=%.5f, test RMSE=%.5f, "
            "loss=%.5f, training time=%.2fs",
            model, final_metrics["train_rmse"], final_metrics["test_rmse"],
            final_metrics["final_loss"], elapsed,
        )
        return final_metrics

    def _train_bias_only(
        self, split: SplitData, config: Mapping[str, float]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """
        Perform bias-only alternating updates and collect diagnostics.
        """
        user_biases = np.zeros(split.n_users, dtype=np.float64)
        movie_biases = np.zeros(split.n_movies, dtype=np.float64)
        losses: list[float] = []
        rmse: dict[str, list[float]] = {"train": [], "test": []}

        for epoch in range(1, self.n_epochs + 1):
            models.update_biases(
                split.users_train, split.movies_train,
                split.users_indices_train, split.movies_indices_train,
                user_biases, movie_biases, config["lambda_reg"], config["gamma"],
            )
            losses.append(float(models.compute_loss(
                split.users_train, split.users_indices_train,
                user_biases, movie_biases, config["lambda_reg"], config["gamma"],
            )))
            rmse["train"].append(float(models.compute_rmse(
                split.users_indices_train, split.users_train,
                user_biases, movie_biases,
            )))
            rmse["test"].append(float(models.compute_rmse(
                split.users_indices_test, split.users_test,
                user_biases, movie_biases,
            )))
            self.logger.info(
                "Bias-Only %d/%d | loss %.5f | train RMSE %.5f | test RMSE %.5f",
                epoch, self.n_epochs, losses[-1],
                rmse["train"][-1], rmse["test"][-1],
            )

        return (
            {"loss": losses, "rmse": rmse},
            {"user_biases_b": user_biases, "movie_biases_b": movie_biases}
        )

    def _train_factor_model(
        self, split: SplitData, config: Mapping[str, float]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """
        Perform factor-model ALS sweeps from reproducible initial factors.
        """
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.n_factors)
        user_biases = np.zeros(split.n_users, dtype=np.float64)
        movie_biases = np.zeros(split.n_movies, dtype=np.float64)
        # Both factor matrices need nonzero starting values: all-zero factors
        # would leave the embedding updates trapped at zero.
        user_embeddings = rng.normal(
            0.0, scale, size=(split.n_users, self.n_factors)
        )
        movie_embeddings = rng.normal(
            0.0, scale, size=(split.n_movies, self.n_factors)
        )
        # An interaction-level split can leave some users or movies without
        # training ratings. Do not use arbitrary random factors at test time
        # for those cold-start entities: fall back to zero bias/factor values.
        seen_users = np.zeros(split.n_users, dtype=bool)
        seen_movies = np.zeros(split.n_movies, dtype=bool)
        seen_users[split.users_indices_train] = True
        seen_movies[split.movies_indices_train] = True
        user_embeddings[~seen_users] = 0.0
        movie_embeddings[~seen_movies] = 0.0
        losses: list[float] = []
        rmse: dict[str, list[float]] = {"train": [], "test": []}

        for epoch in range(1, self.n_epochs + 1):
            models.update_user_biases_and_embeddings(
                split.users_train, split.users_indices_train,
                user_biases, user_embeddings, movie_embeddings, movie_biases,
                config["lambda_reg"], config["gamma"], config["tau"]
            )
            models.update_movie_biases_and_embeddings(
                split.movies_train, split.movies_indices_train,
                user_biases, user_embeddings, movie_embeddings, movie_biases,
                config["lambda_reg"], config["gamma"], config["tau"]
            )
            losses.append(float(models.compute_loss_with_embeddings(
                split.users_train, split.users_indices_train,
                user_biases, user_embeddings, movie_biases, movie_embeddings,
                config["lambda_reg"], config["gamma"], config["tau"],
            )))
            rmse["train"].append(float(models.compute_rmse_with_embeddings(
                split.users_train, split.users_indices_train,
                user_biases, user_embeddings, movie_biases, movie_embeddings,
            )))
            rmse["test"].append(float(models.compute_rmse_with_embeddings(
                split.users_test, split.users_indices_test,
                user_biases, user_embeddings, movie_biases, movie_embeddings,
            )))
            self.logger.info(
                "Matrix Factorization %d/%d | loss %.5f | "
                "train RMSE %.5f | test RMSE %.5f",
                epoch, self.n_epochs, losses[-1],
                rmse["train"][-1], rmse["test"][-1]
            )

        return (
            {"loss": losses, "rmse": rmse},
            {
                "user_biases": user_biases,
                "movie_biases": movie_biases,
                "user_embeddings": user_embeddings,
                "movie_embeddings": movie_embeddings
            }
        )

def _load_parameter_file(file_path: Path) -> dict[str, float]:
    """
    Read saved hyperparameters from a trusted Optuna trial/study pickle.

    Accepts a mapping of parameters or an object exposing ``best_params``.
    """
    if file_path.suffix != ".pkl":
        raise ValueError("Parameter file must have a .pkl extension.")
    parameter_object = utils.load_data_from_pickle(
        str(file_path.parent), file_path.stem, log=False
    )
    values = getattr(parameter_object, "best_params", parameter_object)
    if not isinstance(values, Mapping):
        raise TypeError("Parameter file must contain a dict or .best_params dict.")
    return {key: float(value) for key, value in values.items()}

def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line options for a reproducible ALS experiment.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens data and train ALS recommendation models."
    )
    parser.add_argument(
        "--model", choices=("bias-only", "matrix-factorization", "both"),
        default="both", help="Model family to train (default: both).",
    )
    parser.add_argument(
        "--dataset", default="ml-latest-small",
        help="MovieLens dataset name (default: ml-latest-small).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data",
        help="Root directory for MovieLens raw and preprocessed data.",
    )
    parser.add_argument(
        "--experiment-dir", type=Path, default=PROJECT_ROOT / "experiments",
        help="Root directory for output artifacts.",
    )
    parser.add_argument("--n-factors", type=int, default=16)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lambda-reg", type=float, default=0.216)
    parser.add_argument("--gamma", type=float, default=0.0093)
    parser.add_argument("--tau", type=float, default=0.5092)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--bias-params", type=Path,
        help="Trusted .pkl file containing bias-only Optuna best_params.",
    )
    parser.add_argument(
        "--factor-params", type=Path,
        help="Trusted .pkl file containing factor-model Optuna best_params.",
    )
    parser.add_argument(
        "--force-preprocess", action="store_true",
        help="Recreate preprocessing and train/test split artifacts.",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Use previously generated split and mapping artifacts as-is.",
    )
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the requested experiments and report total elapsed wall time.
    """
    args = _parse_arguments(argv)
    if args.skip_preprocessing and args.force_preprocess:
        raise ValueError("Cannot combine --skip-preprocessing and --force-preprocess.")

    experiment = RecommendationModel(
        n_factors=args.n_factors,
        lambda_reg=args.lambda_reg,
        gamma=args.gamma,
        tau=args.tau,
        n_epochs=args.n_epochs,
        seed=args.seed,
        dataset_name=args.dataset,
        dataset_dir=args.data_dir,
        experiment_dir=args.experiment_dir,
        test_size=args.test_size,
    )
    logger = utils.setup_logging(
        str(experiment.experiment_path), "experiment.log"
    )
    started = time.perf_counter()
    logger.info("Starting MovieLens ALS experiment for %s", args.dataset)

    try:
        if args.skip_preprocessing:
            logger.info("Using existing preprocessing artifacts (not rebuilding).")
        else:
            experiment.preprocess_data(
                force=args.force_preprocess
            )

        if args.model in ("bias-only", "both"):
            bias_params = (
                _load_parameter_file(args.bias_params)
                if args.bias_params else None
            )
            experiment.train_model(
                "Bias-Only", params=bias_params
            )

        if args.model in ("matrix-factorization", "both"):
            factor_params = (
                _load_parameter_file(args.factor_params)
                if args.factor_params else None
            )
            experiment.train_model(
                "Matrix Factorization", params=factor_params
            )
        logger.info("Experiment completed successfully.")
    except Exception:
        logger.exception("Experiment failed.")
        raise
    finally:
        logger.info(
            "Total execution time: %.2f seconds", time.perf_counter() - started
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())