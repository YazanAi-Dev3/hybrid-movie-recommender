# src/model_builder_surprise.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class ModelBuilderSurprise:
    """
    ModelBuilder for the Surprise SVD implementation.
    """
    def __init__(self, n_factors=50, reg_all=0.05, n_epochs=20):
        self.svd_model_surprise = SVD(n_factors=n_factors, reg_all=reg_all, n_epochs=n_epochs, random_state=42)
        self.tfidf = TfidfVectorizer(stop_words='english', min_df=2)

    def _build_content_model(self, films_df: pd.DataFrame):
        # This method is identical across all versions
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

    # <<< THIS IS THE MODIFIED PART FOR SURPRISE >>>
    def _build_collaborative_model(self, reviews_df: pd.DataFrame):
        """Builds the collaborative filtering model using the Surprise library."""
        if reviews_df.empty or len(reviews_df) < 5:
            logger.warning("reviews_df is empty or has too few reviews. Skipping Surprise collaborative model build.")
            return None

        logger.info("Building collaborative filtering model (Surprise SVD)...")
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(reviews_df[['client_id', 'film_id', 'rate']], reader)
        trainset = data.build_full_trainset()
        
        self.svd_model_surprise.fit(trainset)
        
        logger.info("-> Surprise SVD model built successfully.")
        return self.svd_model_surprise

    def build_all_models(self, dataframes: dict):
        """Builds all models for the Surprise implementation."""
        films_df = dataframes.get("films")
        reviews_df = dataframes.get("reviews")

        content_sim_matrix, film_id_to_content_idx = self._build_content_model(films_df)
        svd_model = self._build_collaborative_model(reviews_df)

        return {
            "content_sim_matrix": content_sim_matrix,
            "film_id_to_content_idx": film_id_to_content_idx,
            "svd_model_surprise": svd_model
        }