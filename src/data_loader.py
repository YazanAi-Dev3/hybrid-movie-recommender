# src/data_loader.py

import pandas as pd
from database_manager import DatabaseManager
import logging

# Configure a logger for this module
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Responsible for fetching data from the database using a DatabaseManager
    and loading it into pandas DataFrames.
    """
    def __init__(self, db_manager: DatabaseManager):
        """
        Initializes the DataLoader with a DatabaseManager instance.

        Args:
            db_manager (DatabaseManager): An instance of DatabaseManager to handle DB communication.
        """
        if not isinstance(db_manager, DatabaseManager):
            raise TypeError("db_manager must be an instance of DatabaseManager")
        self.db_manager = db_manager

    def _fetch_and_create_df(self, query, table_name):
        """Helper function to fetch data and convert to a DataFrame."""
        logger.info(f"Fetching data for {table_name}...")
        try:
            data = self.db_manager.execute_query(query, fetch='all')
            df = pd.DataFrame(data) if data else pd.DataFrame()
            logger.info(f"-> Loaded {len(df)} rows into {table_name}_df.")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {table_name}: {e}")
            # Return an empty DataFrame on error
            return pd.DataFrame()

    def load_all_data(self):
        """
        Loads all necessary tables (films, types, reviews, clients) from the database
        and performs initial cleaning.

        Returns:
            dict: A dictionary containing all the loaded and cleaned DataFrames.
        """
        query_films = "SELECT f.id, f.name, f.details, f.language, f.type_id, t.name as type_name FROM films f LEFT JOIN types t ON f.type_id = t.id"
        query_types = "SELECT id, name FROM types"
        query_reviews = "SELECT id, film_id, client_id, rate, content FROM reviews"
        query_clients = "SELECT id, name, gender, age FROM clients"

        films_df = self._fetch_and_create_df(query_films, "films")
        types_df = self._fetch_and_create_df(query_types, "types")
        reviews_df = self._fetch_and_create_df(query_reviews, "reviews")
        clients_df = self._fetch_and_create_df(query_clients, "clients")

        # --- Data Cleaning ---
        if not reviews_df.empty:
            logger.info("Cleaning reviews_df...")
            reviews_df['rate'] = pd.to_numeric(reviews_df['rate'], errors='coerce')
            reviews_df.dropna(subset=['rate', 'film_id', 'client_id'], inplace=True)
            logger.info(f"-> reviews_df shape after cleaning: {reviews_df.shape}")

        return {
            "films": films_df,
            "types": types_df,
            "reviews": reviews_df,
            "clients": clients_df
        }