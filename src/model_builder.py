# src/model_builder.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import logging
import warnings

# Configure a logger for this module
logger = logging.getLogger(__name__)

class ModelBuilder:
    """
    Handles all machine learning model training.
    Takes clean DataFrames and produces all necessary model components
    (matrices, mappings, fitted models).
    """
    def __init__(self, n_components=50):
        """
        Initializes the ModelBuilder.

        Args:
            n_components (int): The number of latent factors for TruncatedSVD.
        """
        self.n_components = n_components
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.tfidf = TfidfVectorizer(stop_words='english', min_df=2)
        self.svd_score_scaler = MinMaxScaler()

    def _build_content_model(self, films_df: pd.DataFrame):
        """Builds the content-based model components."""
        if films_df.empty:
            logger.warning("films_df is empty. Skipping content-based model build.")
            return None, None
        
        logger.info("Building content-based model...")
        films_df_copy = films_df.copy()
        
        # Combine text features, handle NaNs
        films_df_copy['content'] = films_df_copy.apply(
            lambda row: ' '.join(filter(None, [
                str(row.get('name', '')),
                str(row.get('details', '')),
                str(row.get('language', '')),
                str(row.get('type_name', ''))
            ])), axis=1
        )
        
        tfidf_matrix = self.tfidf.fit_transform(films_df_copy['content'].fillna(''))
        content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create a mapping from film ID to its index in the matrix
        film_id_to_content_idx = pd.Series(films_df_copy.index, index=films_df_copy['id'])
        
        logger.info(f"-> Content similarity matrix created with shape: {content_sim_matrix.shape}")
        return content_sim_matrix, film_id_to_content_idx

    def _build_collaborative_model(self, reviews_df: pd.DataFrame):
        """Builds the collaborative filtering model components."""
        if reviews_df.empty or len(reviews_df) < 5:
            logger.warning("reviews_df is empty or has too few reviews. Skipping collaborative model build.")
            return None, None, None, None, None, None

        logger.info("Building collaborative filtering model (sklearn TruncatedSVD)...")
        
        unique_clients = reviews_df['client_id'].unique()
        unique_films = reviews_df['film_id'].unique()

        user_map = pd.Series(range(len(unique_clients)), index=unique_clients)
        film_map = pd.Series(range(len(unique_films)), index=unique_films)
        
        user_codes = reviews_df['client_id'].map(user_map)
        film_codes = reviews_df['film_id'].map(film_map)
        
        user_item_matrix = csr_matrix((reviews_df['rate'].astype(float), (user_codes, film_codes)),
                                      shape=(len(user_map), len(film_map)))

        # Fit SVD and get user/item factors
        user_factors = self.svd_model.fit_transform(user_item_matrix)
        item_factors = self.svd_model.components_

        # Pre-fit the scaler for score normalization
        self._fit_score_scaler(user_factors, item_factors)
        
        logger.info(f"-> Collaborative model built. User factors shape: {user_factors.shape}")
        return self.svd_model, user_factors, item_factors, user_map, film_map, self.svd_score_scaler

    def _fit_score_scaler(self, user_factors, item_factors):
        """Reconstructs a sample of ratings to fit the MinMaxScaler."""
        logger.info("Fitting score scaler for normalization...")
        try:
            all_predicted_ratings = user_factors @ item_factors
            sample_size = min(1_000_000, all_predicted_ratings.size)
            sample_indices = np.random.choice(all_predicted_ratings.size, sample_size, replace=False)
            sample_predictions = all_predicted_ratings.flat[sample_indices]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.svd_score_scaler.fit(sample_predictions.reshape(-1, 1))
            logger.info("-> Score scaler fitted successfully.")
        except Exception as e:
            logger.error(f"Could not fit score scaler: {e}. Normalization may not work as expected.")

    def build_all_models(self, dataframes: dict):
        """
        Takes a dictionary of DataFrames and builds all necessary models.

        Args:
            dataframes (dict): A dict containing 'films' and 'reviews' DataFrames.

        Returns:
            dict: A dictionary containing all trained model assets.
        """
        films_df = dataframes.get("films")
        reviews_df = dataframes.get("reviews")

        content_sim_matrix, film_id_to_content_idx = self._build_content_model(films_df)
        svd_model, user_factors, item_factors, user_map, film_map, svd_score_scaler = self._build_collaborative_model(reviews_df)

        return {
            "content_sim_matrix": content_sim_matrix,
            "film_id_to_content_idx": film_id_to_content_idx,
            "svd_model": svd_model,
            "user_factors": user_factors,
            "item_factors": item_factors,
            "user_map": user_map,
            "film_map": film_map,
            "svd_score_scaler": svd_score_scaler,
        }