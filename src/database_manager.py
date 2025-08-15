# src/database_manager.py

import pymysql
import os
from dotenv import load_dotenv
import logging
import time

# Configure a logger for this module
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages all interactions with the MySQL database.
    Handles connection, disconnection, and query execution.
    """
    def __init__(self):
        """
        Initializes the DatabaseManager by loading DB configuration
        from environment variables.
        """
        load_dotenv()
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
            'connect_timeout': 10,
            'autocommit': False
        }
        self.connection = None

    def connect(self):
        """Establishes a connection to the database."""
        if self.connection and self.connection.open:
            logger.debug("Connection already established.")
            return

        logger.debug("Attempting to connect to the database...")
        try:
            self.connection = pymysql.connect(**self.db_config)
            logger.info(f"Successfully connected to DB '{self.db_config['database']}'")
            # Set transaction isolation level upon connection
            with self.connection.cursor() as cursor:
                cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;")
            logger.info("Session transaction isolation level set to READ COMMITTED.")
        except pymysql.err.OperationalError as e:
            logger.error(f"Failed to connect to the database: {e}")
            self.connection = None
            raise # Re-raise the exception to be handled by the caller

    def close(self):
        """Closes the database connection if it is open."""
        if self.connection and self.connection.open:
            try:
                self.connection.close()
                logger.info("Database connection closed successfully.")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
        self.connection = None

    def _ensure_connection(self):
        """Checks if the connection is open, and reconnects if not."""
        if not self.connection or not self.connection.open:
            logger.warning("DB connection is not open. Reconnecting...")
            self.connect()

    def execute_query(self, query, params=None, fetch=None):
        """
        Executes a given SQL query with optional parameters.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): The parameters to substitute into the query.
            fetch (str, optional): Type of fetch ('one' or 'all'). Default is None for
                                   INSERT/UPDATE/DELETE.

        Returns:
            Result of the query (dict or list of dicts) or number of affected rows.
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                
                if fetch == 'one':
                    result = cursor.fetchone()
                    logger.debug(f"Query returned 1 row.")
                    return result
                elif fetch == 'all':
                    result = cursor.fetchall()
                    logger.debug(f"Query returned {len(result)} rows.")
                    return result
                else: # This is an INSERT, UPDATE, or DELETE
                    self.connection.commit()
                    affected_rows = cursor.rowcount
                    logger.debug(f"Query affected {affected_rows} rows. Committed.")
                    return affected_rows
        except pymysql.err.Error as e:
            logger.error(f"Database query failed: {e}. Rolling back.")
            if self.connection and self.connection.open:
                self.connection.rollback()
            raise # Re-raise to allow higher-level error handling