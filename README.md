# Movie Recommendation System

## Overview
This project implements a highly efficient Collaborative Filtering Recommendation System from scratch. It features custom implementations of Matrix Factorization algorithms, leveraging **Numba** for JIT-compiled performance acceleration and **Polars** for lightning-fast data manipulation.

The system is designed to handle MovieLens datasets (e.g., `ml-latest-small`, `ml-32m`) and provides robust tools for training, evaluation, and visualization of latent feature spaces.

## Key Features

- **High-Performance Modeling**:
  - **Bias-Only Model**: A baseline model capturing global averages, user biases, and movie biases.
  - **Matrix Factorization (Bias + Embeddings)**: A complete latent factor model that learns user and item embeddings to capture complex preferences.
  - **Numba Optimization**: Critical update loops are compiled with `@jit(nopython=True)` for C-like performance.

- **Data Pipeline**:
  - Automatic download and extraction of MovieLens datasets.
  - Efficient preprocessing and train-test splitting (stratified) using **Polars**.
  - Fast storage and retrieval using Pickle and NumPy formats.

- **Visualization & Analysis**:
  - Rating distribution and Power Law (Long Tail) analysis.
  - Training progress tracking (Loss and RMSE curves).
  - **Latent Space Visualization**: Dimensionality reduction (t-SNE, PCA) to visualize movie similarities.

## Directory Structure

```text
.
├── deploy/               # Deployment artifacts (Flask app)
│   ├── app.py            # Streamlit application entry point
├── data/                 # Data storage (Raw downloads and preprocessed pickles)
├── experiments/          # Experiment artifacts (Plots, logs, saved models, data splits)
├── src/                  # Core source code
│   ├── main.py           # Main entry point class `RecommendationModel`
│   ├── models.py         # Mathematical implementations of optimization algorithms
│   └── utils.py          # Utilities for loading data, plotting, and splitting
├── environment.yaml      # Conda environment dependency file
├── example.ipynb         # Jupyter notebook demonstrating the full recommendation flow
├── hyperparameters.ipynb # Conda environment dependency file
└── README.md             # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up the Conda Environment**:
   Ensure you have Conda installed, then create and activate the environment:
   ```bash
   conda env create -f environment.yaml
   conda activate movie_rec
   ```

## Usage

### Jupyter Notebook (Recommended)
The **`example.ipynb`** notebook is the best place to start. It provides a complete walkthrough:
1. Loading and exploring the dataset.
2. Visualizing data distributions (Power Law, Ratings).
3. Training both Bias and Embedding models.
4. Analyzing Training/Test RMSE.
5. Visualizing Movie Embeddings.
6. Generating personalized recommendations.

### Streamlit App
The **`app.py`** file provides a web interface for users to interact with the recommendation system. It allows users to:
1. Select one or more movies from the dataset.
2. Provide ratings for the selected movies.
3. Choose between the Bias-Only and Bias + Factors models.
4. View personalized recommendations based on the selected movies and ratings.

## Methodology

### 1. Data Splitting
The data is split into training and testing sets to evaluate generalization. A custom stratified split ensures users appear in both sets, mimicking realistic evaluation scenarios.

### 2. Bias-Only Model
Minimizes the error by finding the optimal User Bias ($b_u$) and Movie Bias ($b_i$):
\[ \hat{r}_{ui} = \mu + b_u + b_i \]
Solved via alternating least squares (ALS) updates.

### 3. Matrix Factorization (Embeddings)
Adds latent factors ($p_u$ for users, $q_i$ for items) to capture "hidden" features (e.g., genre preferences):
\[ \hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i^T \]
Includes regularization terms ($\lambda$, $\gamma$, $\tau$) to prevent overfitting.

## Results & Visualization
The project includes tools to visualize the learned embeddings. By projecting high-dimensional movie vectors into 2D space (using t-SNE), we can see semantic clustering—similar movies appear closer together.

---
*Created for CS 533 Project.*
