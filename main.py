# main.py

import time
import logging
import sys
import argparse

# Add the 'src' directory to the Python path
sys.path.append('src')

# Import all our refactored classes for all versions
from database_manager import DatabaseManager
from data_loader import DataLoader

from model_builder import ModelBuilder as ModelBuilderSklearn
from recommender import Recommender as RecommenderSklearn

from model_builder_scipy import ModelBuilderSciPy
from recommender_scipy import RecommenderSciPy

from model_builder_surprise import ModelBuilderSurprise
from recommender_surprise import RecommenderSurprise

# --- Application-wide Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the Hybrid Movie Recommender Service.")
    parser.add_argument(
        '--version', 
        type=str, 
        choices=['sklearn', 'scipy', 'surprise'], 
        default='sklearn', 
        help='The model version to run (default: sklearn)'
    )
    args = parser.parse_args()

    logger.info(f" Starting the Hybrid Recommendation Service (Version: {args.version.upper()})...")

    if args.version == 'sklearn':
        ModelBuilderClass = ModelBuilderSklearn
        RecommenderClass = RecommenderSklearn
    elif args.version == 'scipy':
        ModelBuilderClass = ModelBuilderSciPy
        RecommenderClass = RecommenderSciPy
    elif args.version == 'surprise':
        ModelBuilderClass = ModelBuilderSurprise
        RecommenderClass = RecommenderSurprise

    db_manager = DatabaseManager()
    try:
        logger.info("--- Phase 1: Initial Data Load and Model Build ---")
        db_manager.connect()
        data_loader = DataLoader(db_manager)
        dataframes = data_loader.load_all_data()

        model_builder = ModelBuilderClass() 
        model_assets = model_builder.build_all_models(dataframes)
        
        recommender_engine = RecommenderClass(model_assets, dataframes)
        
        logger.info(f" Startup complete. Recommender ({args.version.upper()}) is ready.")
        
        logger.info("--- Phase 2: Entering continuous service loop ---")
        while True:
            logger.info(f"Loop ({args.version.upper()}): Checking for new requests or data updates...")
            time.sleep(30)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received (Ctrl+C).")
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        logger.info("Shutting down the service.")
        db_manager.close()


if __name__ == "__main__":
    main()