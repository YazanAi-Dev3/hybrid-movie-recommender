# src/model_builder_scipy.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import logging
import warnings

logger = logging.getLogger(__name__)

# Note: The class name is changed to avoid conflicts
class ModelBuilderSciPy:
    """
    ModelBuilder specifically for the SciPy SVD implementation.
    """
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(stop_words='english', min_df=2)
        self.svd_score_scaler = MinMaxScaler()

    def _build_content_model(self, films_df: pd.DataFrame):
        # This method is identical to the sklearn version and can be reused
        if films_df.empty:
            logger.warning("films_df is empty. Skipping content-based model build.")
            return None, None
        
        logger.info("Building content-based model...")
        films_df_copy = films_df.copy()
        films_df_copy['content'] = films_df_copy.apply(
            lambda row: ' '.join(filter(None, [
                str(row.get('name', '')), str(row.get('details', '')),
                str(row.get('language', '')), str(row.get('type_name', ''))
            ])), axis=1
        )
        
        tfidf_matrix = self.tfidf.fit_transform(films_df_copy['content'].fillna(''))
        content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        film_id_to_content_idx = pd.Series(films_df_copy.index, index=films_df_copy['id'])
        
        logger.info(f"-> Content similarity matrix created with shape: {content_sim_matrix.shape}")
        return content_sim_matrix, film_id_to_content_idx

    # <<< THIS IS THE MODIFIED PART FOR SCIPY >>>
    def _build_collaborative_model(self, reviews_df: pd.DataFrame):
        """Builds the collaborative filtering model using scipy.sparse.linalg.svds."""
        if reviews_df.empty or len(reviews_df) < 5:
            logger.warning("reviews_df is empty or has too few reviews. Skipping SciPy collaborative model build.")
            return {}

        logger.info("Building collaborative filtering model (SciPy SVD)...")
        
        unique_clients = reviews_df['client_id'].unique()
        unique_films = reviews_df['film_id'].unique()
        user_map = pd.Series(range(len(unique_clients)), index=unique_clients)
        film_map = pd.Series(range(len(unique_films)), index=unique_films)
        
        user_codes = reviews_df['client_id'].map(user_map)
        film_codes = reviews_df['film_id'].map(film_map)
        
        user_item_matrix = csr_matrix((reviews_df['rate'].astype(float), (user_codes, film_codes)),
                                      shape=(len(user_map), len(film_map)))

        k = min(self.n_components, min(user_item_matrix.shape) - 1)
        if k <= 0:
            logger.error(f"Cannot perform SVD: k={k} is not valid for matrix shape {user_item_matrix.shape}")
            return {}

        U, s, Vt = svds(user_item_matrix.astype(float), k=k)
        
        # Sort singular values
        sort_indices = np.argsort(s)[::-1]
        s = s[sort_indices]
        U = U[:, sort_indices]
        Vt = Vt[sort_indices, :]
        
        sigma = np.diag(s)
        
        self._fit_score_scaler(U, sigma, Vt)

        logger.info(f"-> SciPy collaborative model built. U shape: {U.shape}")
        
        return {
            "user_factors": U, "sigma": sigma, "item_factors_t": Vt,
            "user_map": user_map, "film_map": film_map, "svd_score_scaler": self.svd_score_scaler
        }

    def _fit_score_scaler(self, U, sigma, Vt):
        """Reconstructs a sample of ratings to fit the MinMaxScaler for SciPy."""
        logger.info("Fitting score scaler for SciPy model...")
        try:
            all_predicted_ratings = U @ sigma @ Vt
            sample_size = min(1_000_000, all_predicted_ratings.size)
            sample_indices = np.random.choice(all_predicted_ratings.size, sample_size, replace=False)
            sample_predictions = all_predicted_ratings.flat[sample_indices]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.svd_score_scaler.fit(sample_predictions.reshape(-1, 1))
            logger.info("-> Score scaler fitted successfully for SciPy model.")
        except Exception as e:
            logger.error(f"Could not fit score scaler for SciPy model: {e}")

    def build_all_models(self, dataframes: dict):
        """Builds all models for the SciPy implementation."""
        films_df = dataframes.get("films")
        reviews_df = dataframes.get("reviews")

        content_sim_matrix, film_id_to_content_idx = self._build_content_model(films_df)
        collab_assets = self._build_collaborative_model(reviews_df)

        # Combine all assets into one dictionary
        final_assets = {
            "content_sim_matrix": content_sim_matrix,
            "film_id_to_content_idx": film_id_to_content_idx,
        }
        final_assets.update(collab_assets)
        return final_assets