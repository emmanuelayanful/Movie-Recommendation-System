# Movie Recommendation System

**Collaborative filtering with alternating least squares (ALS), from a bias-only baseline to latent-factor matrix factorization.**

This project implements, trains, evaluates, and analyzes an explicit-feedback movie recommender using the [MovieLens datasets](https://grouplens.org/datasets/movielens/). The core algorithms are implemented in Python rather than delegated to a recommender-system library. **Polars** handles data preparation, **Numba** accelerates numerical updates, and **NumPy** supports model fitting and inference.

The repository serves two complementary purposes: a **reproducible experiment pipeline** in `src/main.py` and an **analysis notebook** in `analysis.ipynb` for examining learned movie representations and recommendation behavior. Hyperparameter tuning is documented separately in `hyperparameters.ipynb`.

> **For evaluators:** Start with [How to reproduce an experiment](#how-to-reproduce-an-experiment), then open [`analysis.ipynb`](analysis.ipynb) to examine movie similarities and recommendations. Training and preprocessing happen in `main.py`; the analysis notebook uses the saved results and does not retrain the models.

## Project at a glance

| Component | Purpose |
| --- | --- |
| **Bias-Only ALS** | A baseline that models user and movie rating tendencies without latent factors. |
| **Bias + Factors ALS** | Learns user and movie vectors to capture preference-dependent interactions. |
| **Experiment runner** | Downloads data if needed, preprocesses, trains selected models, evaluates them, and saves artifacts and plots. |
| **Hyperparameter search** | Uses Optuna and a validation subset of the training data to select regularization/data-fit parameters. |
| **Embedding analysis** | Compares movies in the learned factor space using cosine similarity, Euclidean distance, PCA, and t-SNE. |
| **Recommendation analysis** | Estimates preferences for a new user, ranks unseen movies, and compares model behavior and ranking heuristics. |

## Repository organization

```text
Movie-Recommendation-System/
├── src/
│   ├── main.py               # Command-line experiment runner
│   ├── models.py             # ALS updates, losses, and RMSE calculations
│   └── utils.py              # Data preparation, persistence, plots, recommendations
├── analysis.ipynb            # Post-training embedding and recommendation analysis
├── hyperparameters.ipynb     # Optuna hyperparameter experiments
├── environment.yaml          # Conda environment specification
├── data/                     # Created/used during data preparation
│   └── <dataset>/
│       ├── raw/              # Downloaded MovieLens files
│       └── preprocess/       # Ratings, metadata, and ID mappings
├── experiments/              # Created/used by training and analysis
│   └── <dataset>/
│       ├── data_split/       # Grouped train/test observations
│       ├── model_files/      # Trained parameters, histories, summaries
│       ├── images/           # Dataset and training diagnostic plots
│       │   └── analysis/     # Notebook-generated embedding figures
│       └── experiment.log.   # Stores experiment logs including time taken.
└── README.md
```

The `data/` and `experiments/` folders contain generated artifacts; they may be absent from a fresh clone. Their contents are created when the corresponding workflows run. The **movie-ID mappings and model arrays must be generated from the same preprocessing run**, because the models use internal zero-based indices rather than the original MovieLens IDs.

## Installation

**Prerequisites:** Conda and Python 3.10 or newer. Use the dependencies specified in `environment.yaml`.

```bash
git clone https://github.com/emmanuelayanful/Movie-Recommendation-System.git
cd Movie-Recommendation-System

conda env create -f environment.yaml
conda activate movie_rec
```

## How to reproduce an experiment

Run the following commands below from the repository root so the `src` package is importable.

### 1. Run the training pipeline

To download or reuse the small MovieLens dataset and train both models:

```bash
python -m src.main --model both --dataset ml-latest-small
```

To train just one model:

```bash
python -m src.main --model bias-only --dataset ml-latest-small
python -m src.main --model matrix-factorization --dataset ml-latest-small
```

For the larger MovieLens dataset, change the dataset argument to `ml-32m`:

```bash
python -m src.main --model both --dataset ml-32m
```

The default configuration uses **16 latent factors**, **20 epochs**, a **random seed of 42**, and an **approximately 80/20 interaction-level train/test split**. These can be changed with `--n-factors`, `--n-epochs`, `--seed`, and `--test-size`.

On a normal run, `main.py`:

1. Downloads missing MovieLens files and creates/reuses the rating and movie-ID mappings.
2. Builds/reuses the interaction-level training and test artifacts.
3. Saves exploratory plots of per-movie average ratings and rating-count frequencies.
4. Trains the requested model or both models using ALS.
5. Records training loss and train/test RMSE at each epoch.
6. Saves model arrays, histories, configuration summaries, logs, and loss/RMSE plots.

**Figures are saved automatically** as PDF and PNG files; they are not opened as blocking plot windows. Existing complete preprocessing artifacts are reused unless they need to be regenerated. Use `--force-preprocess` to rebuild the derived data and split. Running with `--skip-preprocessing` uses the existing artifacts as-is and skips the preprocessing-stage exploratory plots.

### 2. Inspect the saved results

Outputs are organized under `experiments/<dataset>/`:

| File or folder | Description |
| --- | --- |
| `experiment.log` | Training progress, model configurations, and elapsed time. |
| `model_files/user_biases_b.npy`, `movie_biases_b.npy` | Bias-Only model parameters. |
| `model_files/user_biases.npy`, `movie_biases.npy` | User/movie biases for the factor model. |
| `model_files/user_embeddings.npy`, `movie_embeddings.npy` | Learned latent-factor matrices. |
| `model_files/training_summary_bias.json` | Bias-Only final metrics and parameters. |
| `model_files/training_summary_bias_factors.json` | Factor-model final metrics and parameters. |
| `model_files/*_loss_history.pkl`, `*_rmse_history.pkl` | Per-epoch diagnostics. |
| `images/` | Dataset distributions and training loss/RMSE figures. |

### 3. Explore the trained models in Jupyter

```bash
jupyter lab
```

Open [`analysis.ipynb`](analysis.ipynb), set `DATASET_NAME` to the dataset already trained, and run the notebook cells. No training is performed in this notebook.

The analysis includes:

- **Nearest neighbors:** Compare movies by cosine similarity and Euclidean distance in the original factor space.
- **Similarity heatmap:** Examine relationships among selected films and series.
- **PCA:** Project the learned movie factors onto two principal components.
- **t-SNE:** Visualize a reproducible sample of movies and highlighted titles.
- **Recommendation scenarios:** Compare Bias-Only and Matrix Factorization recommendations for new users who rate selected movies.
- **Ranking sensitivity:** Examine the effect of movie-bias weighting and changes in supplied ratings.

Analysis figures are saved under `experiments/<dataset>/images/analysis/`. The notebook reads model parameters and summaries created by `main.py`; run the relevant model training first.

### 4. Optional: tune hyperparameters

Open [`hyperparameters.ipynb`](hyperparameters.ipynb) to run Optuna studies separately for each model. The studies minimize **validation RMSE** using a subset of the training observations, keeping the held-out test set separate from hyperparameter selection.

The parameter names are consistently **`lambda_reg`, `gamma`, and `tau`**. The notebook saves dictionaries such as:

```python
# Bias-Only
{"lambda_reg": ..., "gamma": ...}

# Bias + Factors
{"lambda_reg": ..., "gamma": ..., "tau": ...}
```

To train using the selected parameters:

```bash
python -m src.main \
  --model both \
  --dataset ml-latest-small \
  --bias-params experiments/ml-latest-small/model_files/params_bias.pkl \
  --factor-params experiments/ml-latest-small/model_files/params_bias_factors.pkl
```

After tuning, perform the final training run and report its held-out test metrics. The test RMSE curves saved during training are diagnostic; they should **not** be used to select hyperparameters.

## Methodology

### Dataset and evaluation protocol

MovieLens provides explicit ratings, with each observed interaction identifying a user, a movie, and a rating. Polars is used to read and transform these records. User and movie IDs are mapped to contiguous, zero-based indices for efficient array operations.

The default **interaction-level** split independently assigns ratings to training or test data using a seeded random generator. **It is not stratified by user or movie**: some entities may have no training observations, and the realized test fraction may differ slightly from the requested value. The implementation handles unseen-in-training factor rows by setting their initially random factors to zero, but this split is not designed as a dedicated cold-start evaluation.

For both models, prediction quality is reported using root mean squared error over observed ratings:

$$
\operatorname{RMSE}=\sqrt{\frac{1}{|\Omega|}\sum_{(u,i)\in\Omega}(r_{ui}-\hat r_{ui})^2}.
$$

### Bias-Only baseline

The baseline predicts a user's rating of movie $i$ as

$$
\hat r_{ui}=b_u+b_i,
$$

where $b_u$ and $b_i$ are learned user and movie biases. **There is no separately fitted global-mean intercept** in the current implementation. With $\Omega_{\mathrm{train}}$ denoting observed training ratings, the model minimizes

$$
\mathcal L_{\mathrm{bias}}=
\frac{\lambda}{2}\sum_{(u,i)\in\Omega_{\mathrm{train}}}
(r_{ui}-b_u-b_i)^2
+\frac{\gamma}{2}\left(\sum_u b_u^2+\sum_i b_i^2\right).
$$

ALS alternately updates user and movie biases while holding the other group fixed. Here, `lambda_reg` controls the squared-error term, and `gamma` controls bias regularization.

### Matrix Factorization: Bias + Factors

The factor model extends the baseline with $K$-dimensional user vectors $p_u$ and movie vectors $q_i$:

$$
\hat r_{ui}=b_u+b_i+p_u^\top q_i.
$$

Its regularized objective is

$$
\begin{aligned}
\mathcal L_{\mathrm{MF}}={}&
\frac{\lambda}{2}\sum_{(u,i)\in\Omega_{\mathrm{train}}}
(r_{ui}-b_u-b_i-p_u^\top q_i)^2\\
&+\frac{\gamma}{2}\left(\sum_u b_u^2+\sum_i b_i^2\right)\\
&+\frac{\tau}{2}\left(\sum_u\|p_u\|_2^2+\sum_i\|q_i\|_2^2\right).
\end{aligned}
$$

The algorithm alternates closed-form, regularized least-squares updates of user-side and movie-side parameters. `tau` controls latent-factor regularization. Numerical kernels are JIT-compiled with Numba; the factor updates solve their corresponding regularized linear systems.

### New-user recommendation and ranking

For a user not present in the training data, the existing **movie parameters are held fixed**. The new user's bias and (for Matrix Factorization) latent vector are estimated from a small set of movie ratings.

Once the user parameters are estimated, previously rated movies are excluded and the remaining movies are ranked:

| Model | Ranking score |
| --- | --- |
| Bias-Only | $s_{ui}=b_i$ |
| Bias + Factors | $s_{ui}=p_u^\top q_i+0.05\,b_i$ (default heuristic) |

The factor model's **0.05 movie-bias weight is a recommendation-ranking heuristic**, intended to reduce the influence of general movie-rating tendencies. It is **not** used for training or for computing rating-prediction RMSE, which uses the full $b_u+b_i+p_u^\top q_i$ prediction equation. The analysis notebook compares alternative bias weights instead of assuming one is universally best.

A limitation of the Bias-Only model is that a new user's estimated bias adds the same constant to every candidate score. Therefore, **changing that user's rating values does not change the ranking** (provided the set of already-rated titles stays the same). Matrix Factorization can change recommendations based on the pattern of ratings.

## Interpreting the embedding analysis

Movie factors are learned from **rating interactions**, not from plot descriptions, genres, or other movie content. The notebook therefore treats closeness in embedding space as an empirical pattern to investigate, not as proof of semantic or genre equivalence.

- **Cosine similarity and Euclidean distance** are computed in the original $K$-dimensional space for quantitative comparisons.
- **PCA** provides a linear two-dimensional projection; the displayed axes do not capture all variation in the factors.
- **t-SNE** emphasizes local neighborhoods and depends on sampling and configuration. Distances in a t-SNE plot should not be interpreted as exact distances in the original embedding space.

## Interactive web demonstration

A separate [Streamlit deployment repository](https://github.com/emmanuelayanful/Movie-Recommendation-Project) provides a lightweight interface for selecting movies, submitting ratings, and viewing recommendations. Its purpose is **inference only**: the deployed application uses saved model files and does not expose or execute the ALS training pipeline.

[Open the hosted movie recommendation app](https://emmanuelayanful-movie-recommendation-project-app-kv1nci.streamlit.app/)

The web demonstration is optional for reproducing the experiments in this repository; all model training and analysis can be performed locally.

## Reproducibility and limitations

- **Deterministic configuration:** Splitting and factor initialization use an explicit seed (default `42`). Record the dataset, split configuration, model, factor dimension, epochs, and hyperparameters when comparing runs.
- **Artifact consistency:** Never mix movie-ID mappings and factor/bias arrays from different preprocessing runs; their internal index order must agree.
- **Hyperparameter selection:** Use validation observations for search. Reserve the test set for final reporting rather than choosing parameters or epochs based on its RMSE curve.
- **Cold-start limitations:** An interaction-level split can contain users or movies absent from training. New-user inference is supported using supplied ratings, but unseen movies do not have learned collaborative factors.
- **Similarity interpretation:** Learned proximity reflects collaborative rating patterns; 2D projections are exploratory visualizations, not ground-truth genre classification.
- **Saved files:** Only load pickle files (`.pkl`) from trusted sources. Large raw data, grouped arrays, and model artifacts may consume substantial storage, especially for `ml-32m`.

## Data and acknowledgments

Ratings and movie metadata come from the [GroupLens MovieLens project](https://grouplens.org/datasets/movielens/). This repository documents the implementation and analysis of ALS-based collaborative filtering; dataset ownership and licensing remain with the original provider. When using MovieLens results in academic work, follow the dataset's attribution guidance.
