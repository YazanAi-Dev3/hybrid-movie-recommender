# src/recommender.py

import pandas as pd
import numpy as np
import logging
from sklearn.exceptions import NotFittedError
import warnings

# Configure a logger for this module
logger = logging.getLogger(__name__)

class Recommender:
    """
    The main engine for generating recommendations.
    It uses pre-trained model assets and data to provide hybrid recommendations.
    This class handles the "online" or "inference" part of the system.
    """
    def __init__(self, model_assets: dict, dataframes: dict):
        """
        Initializes the Recommender with all necessary components.

        Args:
            model_assets (dict): A dictionary containing all trained model components
                                 from the ModelBuilder.
            dataframes (dict): A dictionary containing the necessary pandas DataFrames
                               (films, reviews, clients) from the DataLoader.
        """
        logger.info("Initializing Recommender engine...")
        # Unpack model assets
        self.content_sim_matrix = model_assets.get("content_sim_matrix")
        self.film_id_to_content_idx = model_assets.get("film_id_to_content_idx")
        self.svd_model = model_assets.get("svd_model")
        self.user_factors = model_assets.get("user_factors")
        self.item_factors = model_assets.get("item_factors")
        self.user_map = model_assets.get("user_map")
        self.film_map = model_assets.get("film_map")
        self.svd_score_scaler = model_assets.get("svd_score_scaler")

        # Unpack dataframes
        self.films_df = dataframes.get("films")
        self.reviews_df = dataframes.get("reviews")
        self.clients_df = dataframes.get("clients")
        logger.info("Recommender engine initialized successfully.")

    def get_hybrid_recommendations(self, client_id, film_id=None, age=None, gender=None, top_n=10):
        """
        Generates hybrid recommendations by combining content-based and collaborative filtering.
        This is the main public method of the class.
        """
        logger.debug(f"Generating hybrid recommendations for client_id: {client_id}")
        
        # --- Start of Logic from your original script ---
        try:
            num_initial_recs = max(top_n * 3, 20)
            
            content_recs_ids = self._get_content_based_recommendations(film_id, top_n=num_initial_recs)
            collab_recs_tuples = self._get_collaborative_recommendations(client_id, top_n=num_initial_recs)

            if not content_recs_ids and not collab_recs_tuples:
                logger.warning(f"No base recommendations found for client {client_id}. Falling back to popular films.")
                return self._get_popular_films(top_n)

            combined_scores = {}
            weight_content, weight_collab = 0.4, 0.6

            # Content scores (Rank-based)
            max_rank_c = len(content_recs_ids)
            for i, c_film_id in enumerate(content_recs_ids):
                score = (max_rank_c - i) / max_rank_c if max_rank_c > 0 else 0
                combined_scores[c_film_id] = combined_scores.get(c_film_id, 0) + weight_content * score

            # Collaborative scores (Normalized)
            if collab_recs_tuples:
                collab_scores_raw = np.array([score for _, score in collab_recs_tuples]).reshape(-1, 1)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        normalized_collab_scores = self.svd_score_scaler.transform(collab_scores_raw)
                except NotFittedError:
                    normalized_collab_scores = collab_scores_raw
                
                for i, (film_id_coll, _) in enumerate(collab_recs_tuples):
                    norm_score = normalized_collab_scores[i, 0]
                    combined_scores[film_id_coll] = combined_scores.get(film_id_coll, 0) + weight_collab * norm_score

            if film_id and film_id in combined_scores:
                del combined_scores[film_id]

            sorted_recs_tuples = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
            
            # Apply post-processing filters
            if age or (gender and gender.strip()):
                sorted_recs_tuples = self._filter_by_demographics(sorted_recs_tuples, age, gender)
            
            recommendations_ids_ranked = [rec[0] for rec in sorted_recs_tuples]
            
            diversified_ids = self._diversify_recommendations(recommendations_ids_ranked)
            
            final_recommendations = diversified_ids[:top_n]

        except Exception as e:
            logger.error(f"Error during get_hybrid_recommendations for client {client_id}: {e}", exc_info=True)
            final_recommendations = self._get_popular_films(top_n)
        
        return final_recommendations


    def _get_content_based_recommendations(self, film_id, top_n=10):
        """Generates content-based recommendations for a given film."""
        if self.content_sim_matrix is None or film_id is None:
            return []
        
        if film_id not in self.film_id_to_content_idx.index:
            logger.warning(f"Film ID {film_id} not found for content-based recommendations.")
            return []
        
        film_idx = self.film_id_to_content_idx[film_id]
        sim_scores = list(enumerate(self.content_sim_matrix[film_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        recommended_film_indices = [i[0] for i in sim_scores[1:top_n+1]]
        return self.films_df['id'].iloc[recommended_film_indices].tolist()

    def _get_collaborative_recommendations(self, client_id, top_n=10):
        """Generates collaborative filtering recommendations for a given client."""
        if self.svd_model is None or client_id not in self.user_map.index:
            logger.warning(f"Client ID {client_id} not found for collaborative recommendations.")
            return []
            
        user_idx = self.user_map[client_id]
        user_vector = self.user_factors[user_idx, :]
        predicted_scores = user_vector.dot(self.item_factors)
        
        film_codes = self.film_map.values
        film_ids = self.film_map.index
        
        predictions_df = pd.DataFrame({'film_id': film_ids, 'score': predicted_scores[film_codes]})
        
        rated_films = self.reviews_df[self.reviews_df['client_id'] == client_id]['film_id']
        predictions_df = predictions_df[~predictions_df['film_id'].isin(rated_films)]
        
        top_predictions = predictions_df.sort_values('score', ascending=False).head(top_n)
        return list(zip(top_predictions['film_id'], top_predictions['score']))

    def _filter_by_demographics(self, recommendations_tuples, age=None, gender=None):
        # This is a placeholder for the complex demographic logic 
        # For brevity, we are returning the original list
        logger.info("Demographic filtering would be applied here.")
        return recommendations_tuples

    def _diversify_recommendations(self, recommendations_ids, diversity_factor=0.4):
        # This is a placeholder for the diversification logic.
        # For brevity, we are returning the original list.
        logger.info("Diversification would be applied here.")
        return recommendations_ids

    def _get_popular_films(self, top_n=10):
        """Generates a list of popular films as a fallback."""
        if self.reviews_df.empty:
            return []
        logger.info("Falling back to popular films.")
        film_ratings = self.reviews_df.groupby('film_id')['rate'].agg(['count', 'mean']).reset_index()
        # Simple popularity metric: mean rating * number of ratings
        film_ratings['popularity'] = film_ratings['mean'] * film_ratings['count']
        popular_films = film_ratings.sort_values('popularity', ascending=False).head(top_n)
        return popular_films['film_id'].tolist()