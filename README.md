# Scalable Movie Recommendation System with CASM

A thesis-based recommender system project built with Python, focusing on scalable collaborative filtering, confidence-aware similarity, matrix factorization, incremental updates, and recommender-system evaluation.

This project is based on my MSc thesis on scalable recommender systems using incremental collaborative filtering. The goal is to transform the academic implementation into a clean, modular, GitHub-ready data science project.

---

## Project Overview

Recommender systems are widely used in digital platforms to reduce information overload and personalize user experience. Traditional collaborative filtering methods, however, face challenges when the data is large, sparse, and dynamic.

This project implements a hybrid recommender system based on:

- Collaborative Filtering
- Confidence-Aware Similarity Measure (CASM)
- Matrix Factorization with SGD
- Hybrid CF + MF rating prediction
- Baseline bias prediction
- Incremental update logic
- Rating prediction evaluation
- Ranking quality evaluation
- Experiment tracking
- Result visualizations
- Streamlit demo

The system supports MovieLens 100K, MovieLens 1M, and MovieLens 10M through a configurable dataset setup.

---

## Main Features

- Loads and preprocesses multiple MovieLens datasets
- Supports configurable execution on MovieLens 100K, 1M, and 10M
- Implements Confidence-Aware Similarity Measure (CASM)
- Implements Matrix Factorization using Stochastic Gradient Descent
- Combines collaborative filtering and matrix factorization in a hybrid prediction model
- Uses user-item rating dictionaries for efficient recommendation logic
- Generates Top-N movie recommendations
- Supports incremental update logic for new user-item interactions
- Evaluates rating prediction using RMSE and MAE
- Evaluates ranking quality using Precision@K, Recall@K, NDCG@K, MRR@K, and F1@K
- Saves experiment results into `results/metrics.csv`
- Generates metric visualizations in `results/figures`
- Includes an interactive Streamlit recommendation demo
- Keeps raw and processed datasets out of GitHub using `.gitignore`

---

## Project Structure

```text
movie-recommender-system/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   ├── ml-100k/
│   │   ├── ml-1m/
│   │   └── ml-10m/
│   │
│   └── processed/
│
├── notebooks/
│
├── results/
│   ├── figures/
│   │   ├── rating_metrics.png
│   │   └── ranking_metrics_at_10.png
│   │
│   └── metrics.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── matrix_factorization.py
│   ├── prepare_datasets.py
│   ├── preprocessing.py
│   ├── recommender.py
│   ├── similarity.py
│   ├── utils.py
│   └── visualization.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## Dataset

This project uses the MovieLens datasets published by GroupLens.

Supported datasets:

- MovieLens 100K
- MovieLens 1M
- MovieLens 10M

The datasets are not included in this repository because they are stored locally and excluded using `.gitignore`.

Download the datasets from the official GroupLens website:

```text
https://grouplens.org/datasets/movielens/
```

After downloading and extracting them, place the files as follows:

```text
data/raw/ml-100k/u.data
data/raw/ml-1m/ratings.dat
data/raw/ml-10m/ratings.dat
```

Then prepare the datasets:

```bash
python src/prepare_datasets.py
```

This creates processed CSV files such as:

```text
data/processed/ratings_100k.csv
data/processed/ratings_1m.csv
data/processed/ratings_10m.csv
```

---

## Methodology

### 1. Collaborative Filtering

The recommender uses user-item interactions to estimate user preferences. Similar users are identified based on shared rated items.

### 2. CASM Similarity

The Confidence-Aware Similarity Measure combines several components:

- Pearson correlation
- Support confidence
- Jaccard overlap
- User expertise weight

This reduces the impact of unreliable similarities in sparse data.

### 3. Matrix Factorization

The project includes matrix factorization trained with Stochastic Gradient Descent. The model learns latent vectors for users and items and predicts ratings based on:

- Global mean rating
- User bias
- Item bias
- User latent factors
- Item latent factors

### 4. Hybrid Prediction

The final rating prediction combines:

- Baseline bias prediction
- CASM-based collaborative filtering prediction
- Matrix factorization prediction

The hybrid prediction is controlled through configurable weights in `src/config.py`.

### 5. Incremental Update Logic

The project includes an incremental update mechanism. When a new rating is added, the system updates:

- User-item interaction dictionary
- Item-user interaction dictionary
- CASM similarity cache for the affected user
- Matrix factorization latent factors for the affected user-item pair

---

## Evaluation Metrics

### Rating Prediction Metrics

- RMSE
- MAE

### Ranking Metrics

- Precision@K
- Recall@K
- NDCG@K
- MRR@K
- F1@K

For ranking evaluation, the candidate set is built using relevant test items plus sampled negative candidates. This prevents ranking evaluation from becoming artificially zero when relevant items are not sampled.

---

## Current Experimental Results

The system was evaluated on MovieLens 100K, MovieLens 1M, and MovieLens 10M.

For MovieLens 10M, the full dataset statistics are reported, while training and evaluation are performed in scalable sampled mode to keep the experiment feasible on a local laptop.

### Dataset Statistics

| Dataset | Ratings | Users | Items | Sparsity | Train Rows Used | Test Rows Used |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens 100K | 100,000 | 943 | 1,682 | 93.69% | 80,000 | 20,000 |
| MovieLens 1M | 1,000,209 | 6,040 | 3,706 | 95.53% | 800,167 | 200,042 |
| MovieLens 10M | 10,000,054 | 69,878 | 10,677 | 98.66% | 1,000,000 | 50,000 |

### Rating Prediction Results

| Dataset | RMSE | MAE | Sample Size |
|---|---:|---:|---:|
| MovieLens 100K | 0.9061 | 0.7195 | 1,000 |
| MovieLens 1M | 0.8974 | 0.7124 | 2,000 |
| MovieLens 10M | 0.9181 | 0.7099 | 1,000 |

### Ranking Results at K=10

| Dataset | Precision@10 | Recall@10 | NDCG@10 | MRR@10 | F1@10 | Evaluated Users |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens 100K | 0.0340 | 0.0238 | 0.0481 | 0.1560 | 0.0242 | 50 |
| MovieLens 1M | 0.0100 | 0.0206 | 0.0222 | 0.0540 | 0.0111 | 50 |
| MovieLens 10M | 0.0650 | 0.3381 | 0.2201 | 0.2342 | 0.0998 | 20 |

### MovieLens 10M Scalable Evaluation Setup

MovieLens 10M is highly sparse, with approximately 98.66% sparsity.

To make the experiment feasible on a local laptop, this project uses a scalable sampled training and evaluation setup for 10M:

- Full dataset loaded for statistics
- 1,000,000 training rows used
- 50,000 test rows used
- 3 matrix factorization epochs
- Ranking evaluation uses relevant test items plus sampled negative candidates
- 17,120 eligible users identified for ranking evaluation
- 20 users sampled for final ranking evaluation

This setup demonstrates that the implementation can handle large-scale MovieLens data while remaining executable in a portfolio environment.

---

## Result Visualizations

Rating prediction metrics:

![Rating Metrics](results/figures/rating_metrics.png)

Ranking metrics at K=10:

![Ranking Metrics at K=10](results/figures/ranking_metrics_at_10.png)

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Zahra-ziaee/movie-recommender-system.git
cd movie-recommender-system
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the datasets

Make sure the MovieLens files are placed in the correct folders, then run:

```bash
python src/prepare_datasets.py
```

### 5. Run the project

Run on MovieLens 100K:

```bash
python main.py --dataset 100K
```

Run on MovieLens 1M:

```bash
python main.py --dataset 1M
```

Run on MovieLens 10M:

```bash
python main.py --dataset 10M
```

---

## Streamlit Demo

Run the interactive recommendation demo:

```bash
streamlit run app/streamlit_app.py
```

The demo allows users to enter a user ID and generate Top-N movie recommendations using the hybrid CASM-CF and Matrix Factorization recommender.

---

## Experiment Tracking

Each experiment run is saved into:

```text
results/metrics.csv
```

The file includes:

- Timestamp
- Dataset name
- Model name
- RMSE
- MAE
- Ranking metrics
- Evaluated users

This makes the project easier to reproduce, compare, and extend.

---

## Current Status

Completed:

- Project structure
- GitHub setup
- MovieLens 100K, 1M, and 10M support
- Data preparation scripts
- Data loading and train/test split
- Configurable dataset selection
- CASM similarity engine
- Matrix Factorization with SGD
- Hybrid CF + MF prediction
- Rating prediction evaluation
- Ranking metrics evaluation
- Incremental update logic
- Experiment tracking with CSV logging
- Result visualizations
- Streamlit recommendation demo

Planned next steps:

- Add more advanced candidate generation
- Add item metadata and movie titles to the Streamlit demo
- Add notebook-based exploratory data analysis
- Add model comparison with baseline recommenders
- Add optional FastAPI endpoint
- Add Docker support

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Git
- GitHub

---

## Author

Zahra Ziaee

MSc Computer Science - Data Mining  
Focus: Recommender Systems, Collaborative Filtering, Incremental Learning, Matrix Factorization, and Scalable Machine Learning