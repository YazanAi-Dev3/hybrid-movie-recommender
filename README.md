<div align="center">

# Hybrid Movie Recommender System

### Collaborative + content-based filtering, implemented three ways

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Surprise](https://img.shields.io/badge/Surprise-Recommenders-1f77b4)](https://surpriselib.com)
[![MySQL](https://img.shields.io/badge/MySQL-Database-4479A1?logo=mysql&logoColor=white)](https://mysql.com)

**A hybrid recommender that blends "what did similar users watch?" with "what are similar movies?" — and ships the collaborative engine in three separate implementations to compare abstraction vs. control.**

</div>

---

## Design

```mermaid
flowchart TD
    DB[("MySQL: users · movies · ratings")] --> DL["data_loader.py<br/>load + clean"]
    DL --> CF["Collaborative filtering<br/>(similar users)"]
    DL --> CB["Content-based filtering<br/>(similar movies)"]
    CF --> HYB["Hybrid blend"]
    CB --> HYB
    HYB --> SVC["main.py live service<br/>periodic recommendation processing"]

    CF -. "3 interchangeable backends" .- IMPL["Surprise · scikit-learn · SciPy"]
```

## Three Implementations, One Interface

| Version | Library | Abstraction | Best For |
|---|---|---|---|
| Surprise | specialized recommender lib | very high (black box) | rapid prototyping, algorithm comparison |
| scikit-learn | general-purpose ML | medium | integrating into a larger ML system |
| SciPy | low-level scientific | very low (full control) | custom systems needing fine-grained control |

Selectable at runtime: `python main.py --version {sklearn|scipy|surprise}`.

---

## Structure

```
hybrid-movie-recommender/
├── src/
│   ├── database_manager.py         # DB connection
│   ├── data_loader.py              # load + clean
│   ├── model_builder{,_scipy,_surprise}.py
│   └── recommender{,_scipy,_surprise}.py
├── data/db_schema.sql              # MySQL schema
├── demo.ipynb                      # interactive demo
├── generate_dummy_data.py
└── main.py                         # live service entry point
```

## Setup

```bash
git clone https://github.com/YazanAi-Dev3/hybrid-movie-recommender.git
cd hybrid-movie-recommender

mysql -u <user> -p movie_recommender_db < data/db_schema.sql
cp .env.example .env      # DB credentials
pip install -r requirements.txt
python generate_dummy_data.py   # optional: populate with fake data
python main.py                  # default (sklearn)
```

## Tech Stack

`Python` · `scikit-learn` · `Surprise` · `SciPy` · `MySQL` · `Pandas`

## License

MIT — see [LICENSE](LICENSE).
