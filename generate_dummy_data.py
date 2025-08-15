# --- generate_dummy_data.py ---
# This script populates the database with realistic dummy data for the recommender system.

import pymysql
import os
import random
from faker import Faker
from tqdm import tqdm
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'cursorclass': pymysql.cursors.DictCursor
}

# --- Constants for Data Generation ---
NUM_TYPES = 10
NUM_CLIENTS = 200
NUM_FILMS = 500
NUM_REVIEWS = 5000
NUM_REQUESTS = 50

# Initialize Faker for data generation
fake = Faker()

def insert_data(connection, query, data):
    """Helper function to insert multiple rows of data."""
    with connection.cursor() as cursor:
        cursor.executemany(query, data)
    connection.commit()

def generate_types(conn):
    """Generates and inserts film types/genres."""
    print("Generating film types...")
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Thriller', 'Romance', 'Animation', 'Documentary', 'Fantasy']
    # Ensure we don't try to insert more than we have
    data_to_insert = [(genre,) for genre in genres[:NUM_TYPES]]
    query = "INSERT INTO types (name) VALUES (%s)"
    insert_data(conn, query, data_to_insert)
    print(f"-> Inserted {len(data_to_insert)} types.")
    return [i for i in range(1, len(data_to_insert) + 1)]

def generate_clients(conn):
    """Generates and inserts clients."""
    print("Generating clients...")
    data_to_insert = []
    for _ in tqdm(range(NUM_CLIENTS), desc="Clients"):
        data_to_insert.append((
            fake.name(),
            random.randint(13, 70),
            random.choice(['male', 'female', 'other'])
        ))
    query = "INSERT INTO clients (name, age, gender) VALUES (%s, %s, %s)"
    insert_data(conn, query, data_to_insert)
    print(f"-> Inserted {NUM_CLIENTS} clients.")
    return [i for i in range(1, NUM_CLIENTS + 1)]

def generate_films(conn, type_ids):
    """Generates and inserts films."""
    print("Generating films...")
    data_to_insert = []
    for _ in tqdm(range(NUM_FILMS), desc="Films"):
        data_to_insert.append((
            ' '.join(fake.words(nb=random.randint(2, 5))).title(), # Fake movie title
            fake.paragraph(nb_sentences=3),
            fake.language_code(),
            random.choice(type_ids)
        ))
    query = "INSERT INTO films (name, details, language, type_id) VALUES (%s, %s, %s, %s)"
    insert_data(conn, query, data_to_insert)
    print(f"-> Inserted {NUM_FILMS} films.")
    return [i for i in range(1, NUM_FILMS + 1)]

def generate_reviews(conn, client_ids, film_ids):
    """Generates and inserts reviews, ensuring no duplicate reviews."""
    print("Generating reviews...")
    data_to_insert = []
    # Use a set to track existing (client, film) pairs to prevent duplicates
    existing_reviews = set()
    
    for _ in tqdm(range(NUM_REVIEWS), desc="Reviews"):
        # Keep trying until a unique review pair is found
        while True:
            client_id = random.choice(client_ids)
            film_id = random.choice(film_ids)
            if (client_id, film_id) not in existing_reviews:
                existing_reviews.add((client_id, film_id))
                break
        
        data_to_insert.append((
            film_id,
            client_id,
            round(random.uniform(1.0, 5.0), 1),
            fake.sentence()
        ))
    query = "INSERT INTO reviews (film_id, client_id, rate, content) VALUES (%s, %s, %s, %s)"
    insert_data(conn, query, data_to_insert)
    print(f"-> Inserted {NUM_REVIEWS} unique reviews.")

def generate_requests(conn, client_ids):
    """Generates and inserts pending recommendation requests."""
    print("Generating requests...")
    # Select a subset of clients to make requests
    requesting_clients = random.sample(client_ids, k=min(NUM_REQUESTS, len(client_ids)))
    
    data_to_insert = []
    for client_id in tqdm(requesting_clients, desc="Requests"):
        data_to_insert.append((
            client_id,
            0, # Status 0 for pending
        ))
    query = "INSERT INTO requests (client_id, status) VALUES (%s, %s)"
    insert_data(conn, query, data_to_insert)
    print(f"-> Inserted {len(requesting_clients)} requests.")

def main():
    """Main function to connect to DB and run all data generation steps."""
    connection = None
    try:
        print("Connecting to the database...")
        connection = pymysql.connect(**DB_CONFIG)
        print("Connection successful.")
        
        # The order is important due to foreign key constraints
        type_ids = generate_types(connection)
        client_ids = generate_clients(connection)
        film_ids = generate_films(connection, type_ids)
        generate_reviews(connection, client_ids, film_ids)
        generate_requests(connection, client_ids)
        
        print("\n✅ Database has been populated successfully!")

    except pymysql.err.OperationalError as e:
        print(f"Error connecting to database: {e}")
        print("Please check your .env file and ensure the MySQL server is running.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if connection:
            connection.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()