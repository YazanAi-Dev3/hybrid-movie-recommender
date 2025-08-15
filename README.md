#  Hybrid Movie Recommender System

This project presents a hybrid movie recommender system, designed to operate as a live service. The system is capable of providing personalized recommendations to users by combining Content-Based Filtering and Collaborative Filtering techniques.

The key feature of this project is its implementation in **three distinct ways**, showcasing flexibility and a deep understanding of different data science tools and libraries.

---

##  Key Features

* **Hybrid Model**: Integrates the power of Collaborative Filtering (what did similar users watch?) and Content-Based Filtering (what are similar movies?) to provide accurate and diverse recommendations.
* **Three Distinct Implementations**: The collaborative filtering engine was built using:
    1.  **Surprise**: A high-level library specialized for recommender systems.
    2.  **Scikit-learn**: The most popular machine learning library in Python.
    3.  **SciPy**: A low-level scientific library for fine-grained control over computations.
* **Professional Architecture**: The project is designed with a clean separation of concerns (database, data loading, model building, recommender engine) for easy maintenance and scalability.
* **Live Service Mode**: Designed to run as a backend service (`main.py`) that periodically processes recommendation requests.
* **Demonstration Notebook**: Includes a Jupyter Notebook (`demo.ipynb`) to interactively showcase and analyze the system's performance.

---

##  Implementation Comparison

| Feature           | Surprise Version                             | Scikit-learn Version                           | SciPy Version                                      |
| :---------------- | :------------------------------------------- | :--------------------------------------------- | :------------------------------------------------- |
| **Library** | Specialized for Recommenders                 | General-Purpose Machine Learning               | General-Purpose Scientific (Low-Level)             |
| **Abstraction Level** | **Very High** (Black Box)                    | **Medium** (Convenient API)                    | **Very Low** (Full Control)                        |
| **Control** | Limited                                      | Very Good                                      | Complete                                           |
| **Best For** | Rapid prototyping, comparing algorithms.     | Integrating recommendations into a larger ML system, balancing ease of use and control. | Custom systems that require precise control and deep understanding. |

---

##  Project Structure

```
/hybrid-movie-recommender
|
├── /data
│   └── db_schema.sql           # Database schema
├── /notebooks
│   └── demo.ipynb              # Demonstration notebook
├── /src
│   ├── database_manager.py     # Manages DB connection
│   ├── data_loader.py          # Loads and cleans data
│   ├── model_builder.py        # Builds models (sklearn)
│   ├── model_builder_scipy.py  # Builds models (scipy)
│   ├── model_builder_surprise.py # Builds models (surprise)
│   ├── recommender.py          # Recommender engine (sklearn)
│   ├── recommender_scipy.py    # Recommender engine (scipy)
│   └── recommender_surprise.py # Recommender engine (surprise)
|
├── .env.example                # Example environment variables file
├── .gitignore                  # Files to be ignored by Git
├── app.log                     # Application log file
├── generate_dummy_data.py      # Script to generate fake data
├── main.py                     # Entry point to run the service
├── requirements.txt            # Required Python libraries
└── README.md                   # This file
```

---

##  Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

* Python 3.8+
* A running MySQL or MariaDB server

### 2. Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YazanAi-Dev3/hybrid-movie-recommender.git](https://github.com/YazanAi-Dev3/hybrid-movie-recommender.git)
    cd hybrid-movie-recommender
    ```

2.  **Set Up the Database:**
    * Create a new database in MySQL (e.g., `movie_recommender_db`).
    * Execute the database schema to create the tables:
    ```bash
    mysql -u your_user -p movie_recommender_db < data/db_schema.sql
    ```

3.  **Configure Environment Variables:**
    * Copy the example `.env.example` file to `.env`:
        ```bash
        cp .env.example .env
        ```
    * Open the `.env` file and fill in your actual database credentials.

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **(Optional) Generate Dummy Data:**
    To populate the database with usable fake data, run:
    ```bash
    python generate_dummy_data.py
    ```

### 3. Usage

#### Running the Live Service

You can run any version of the recommender system from the command line:

```bash
# Run the default version (sklearn)
python main.py

# Explicitly run the SciPy version
python main.py --version scipy

# Explicitly run the Surprise version
python main.py --version surprise
```

#### Running the Demonstration Notebook

For an interactive demonstration, launch Jupyter:

```bash
jupyter notebook
```

Then, open the `notebooks/demo.ipynb` file and run the cells. You can change the model version to use directly inside the notebook.