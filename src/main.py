#!/usr/bin/env python

from src.utils import *


# ============= Class Definition ===============

class RecommendationModel:
    def __init__(self, n_factors=25, lambda_reg=1, gamma=0.1, tau=10, n_epochs=20, jitter=1e-8, seed=42, dataset_name="ml-latest-small", dataset_dir="../data", experiment_dir="../experiments"):
        self.n_factors = n_factors
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.tau = tau
        self.n_epochs = n_epochs
        self.jitter = jitter
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.experiment_dir = experiment_dir

    def preprocess_data(self):
        load_ratings_data(self.dataset_dir ,self.dataset_name)
        load_movie_data(self.dataset_dir ,self.dataset_name)
        plot_distribution(self.dataset_dir, self.experiment_dir, self.dataset_name, title='', xlabel='Rating', ylabel='Frequency', color='g', kind='small')
        plot_power_law_distribution(self.dataset_dir, self.experiment_dir, self.dataset_name, title='', xlabel='Rating', ylabel='Frequency')
        split_into_train_test(self.dataset_dir, self.experiment_dir, self.dataset_name)
    
    def train_model(self,  model):
        if model == "Bias-Only":
            bias_loss_history = []
            bias_rmse_history = {'train': [], 'test': []}

            with tqdm(range(n_iterations), unit='epochs') as tepochs:
                tepochs.set_description('Training')
                for _ in tepochs:
                    user_biases, movie_biases = update_biases(users_train, movies_train, users_indices_train, movies_indices_train,
                    user_biases, movie_biases, LAMBDA, GAMMA)
                    loss = compute_loss(users_train, users_indices_train, user_biases, movie_biases, LAMBDA, GAMMA)
                    train_rmse = compute_rmse(users_indices_train, users_train, user_biases, movie_biases)
                    test_rmse = compute_rmse(users_indices_test, users_test, user_biases, movie_biases)
                    bias_loss_history.append(loss)
                    bias_rmse_history['train'].append(train_rmse)
                    bias_rmse_history['test'].append(test_rmse)
                    tepochs.set_postfix(rmse_train=bias_rmse_history['train'][-1], rmse_test=bias_rmse_history['test'][-1])

            os.makedirs(save_path, exist_ok=True)
            save_numpy_array(save_path, user_biases, 'user_biases_b')
            save_numpy_array(save_path, movie_biases, 'movie_biases_b')
            save_data_to_pickle(save_path, bias_loss_history, 'bias_loss_history')
            save_data_to_pickle(save_path, bias_rmse_history, 'bias_rmse_history')
        
        elif model == "Matrix Factorization":
            bias_factors_loss_history = []
            bias_factors_rmse_history = {'train': [], 'test': []}

            n_iterations = 50
            K = 16
            np.random.seed(self.seed)

            # Initialize biases
            n_users, n_movies = len(map_user_to_id), len(map_movie_to_id)
            user_biases = np.zeros(n_users)
            movie_biases = np.zeros(n_movies)
            user_embeddings = np.random.normal(0, 1/np.sqrt(K), size=(n_users, K))
            movie_embeddings = np.random.normal(0, 1/np.sqrt(K), size=(n_movies, K))

            with tqdm(range(n_iterations), unit='epochs') as tepochs:
                tepochs.set_description('Training')
                for _ in tepochs:
                    user_biases, user_embeddings = update_user_biases_and_embeddings(users_train, users_indices_train, user_biases, 
                                                                        user_embeddings, movie_embeddings, movie_biases, LAMBDA, GAMMA, TAU)
                    movie_biases, movie_embeddings = update_movie_biases_and_embeddings(movies_train, movies_indices_train, user_biases,
                                                                        user_embeddings, movie_embeddings, movie_biases, LAMBDA, GAMMA, TAU)
                    loss = compute_loss_with_embeddings(users_train, users_indices_train, user_biases, user_embeddings, 
                                 movie_biases, movie_embeddings, LAMBDA, GAMMA, TAU)
                    train_rmse = compute_rmse_with_embeddings(users_train, users_indices_train, user_biases, user_embeddings, 
                                 movie_biases, movie_embeddings)
                    test_rmse = compute_rmse_with_embeddings(users_test, users_indices_test, user_biases, user_embeddings, 
                                 movie_biases, movie_embeddings)
                    bias_factors_loss_history.append(loss)
                    bias_factors_rmse_history['train'].append(train_rmse)
                    bias_factors_rmse_history['test'].append(test_rmse)
                    tepochs.set_postfix(rmse_train=bias_factors_rmse_history['train'][-1], rmse_test=bias_factors_rmse_history['test'][-1])
        
            os.makedirs(save_path, exist_ok=True)
            save_numpy_array(save_path, user_biases, 'user_biases')
            save_numpy_array(save_path, user_embeddings, 'user_embeddings')
            save_numpy_array(save_path, movie_biases, 'movie_biases')
            save_numpy_array(save_path, movie_embeddings,'movie_embeddings')
            save_data_to_pickle(save_path, bias_factors_loss_history, 'bias_factors_loss_history')
            save_data_to_pickle(save_path, bias_factors_rmse_history, 'bias_factors_rmse_history')
            



# ============= Main Script ===================
if __name__ == "__main__":
    save_path = "../experiments"
    bias_factors_params = load_data_from_pickle(save_path, 'params_bias_factors').best_params
    LAMBDA, GAMMA, TAU = bias_factors_params['LAMBDA'], bias_factors_params['GAMMA'], bias_factors_params['TAU']
    model = RecommendationModel()
    model.preprocess_data()
