# Sentiment Analysis Engine

An end-to-end **binary sentiment classifier** for English text, trained on the **IMDB 50K Movie Reviews** dataset. Raw reviews go through an NLTK preprocessing pipeline, are turned into **sublinear TF-IDF n-gram features**, and are scored by three linear models plus a **hard-voting ensemble** of all three.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154F5B)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-success)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Project Architecture](#project-architecture)
- [Pipeline Overview](#pipeline-overview)
- [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
- [Models](#models)
- [Performance Metrics](#performance-metrics)
- [Quick Start](#quick-start)
- [Interactive Real-Time Testing](#interactive-real-time-testing)
- [Programmatic Usage](#programmatic-usage)
- [Retraining](#retraining)
- [Known Limitations & Roadmap](#known-limitations--roadmap)

---

## Executive Summary

| | |
|---|---|
| **Task** | Binary sentiment classification (Positive / Negative) |
| **Dataset** | IMDB 50K Movie Reviews (25K positive, 25K negative) |
| **NLP stack** | BeautifulSoup, NLTK (tokenization, stopwords, WordNet lemmatization) |
| **Features** | TF-IDF, word 1–3 grams, 75,000 features, sublinear TF |
| **Models** | Logistic Regression, Linear SVC, Passive Aggressive, hard Voting Classifier |
| **Best single model** | Linear SVC, **~91.17%** held-out accuracy |
| **Persistence** | `joblib` artifacts in `models/`, loaded on startup |
| **Interface** | Interactive CLI (`python3 main.py`) |

On first run the system **trains itself** if no saved models exist. Later runs load the saved artifacts and are ready for predictions within seconds.

---

## Project Architecture

```
Sentiment_Analysis_Model/
├── data/
│   └── Movie_Reviews.csv          # IMDB 50K dataset (columns: review, sentiment)
├── models/                        # Generated artifacts (created by training)
│   ├── vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── linear_support_vector_svc.joblib
│   ├── passive_aggressive.joblib
│   └── ensemble_voting_model.joblib
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # TextPreprocessor: HTML strip, normalize, lemmatize
│   ├── model.py                   # Model definitions and hyperparameters
│   ├── train.py                   # Load → clean → split → vectorize → fit → evaluate → save
│   └── predict.py                 # Load-or-train logic and real-time inference
├── main.py                        # Interactive CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

| Module | Responsibility |
|---|---|
| `src/preprocess.py` | `TextPreprocessor.clean_text()` turns raw review text into normalized tokens. Downloads the required NLTK corpora on import. |
| `src/model.py` | `get_multi_models()` returns the three base classifiers and the `VotingClassifier`, with hyperparameters set. |
| `src/train.py` | `train_and_save_all_models()` runs the full training pipeline, prints test accuracy for every model, and saves all artifacts. |
| `src/predict.py` | `load_or_train_pipeline()` loads artifacts, or trains if any are missing. `predict_realtime_text()` scores one input with every model. |
| `main.py` | Terminal interface for real-time testing. |

---

## Pipeline Overview

```
                ┌──────────────────────── TRAINING ────────────────────────┐
 Movie_Reviews  │  TextPreprocessor  →  80/20 stratified split  →  TF-IDF  │
     .csv  ───► │  (HTML, punct,         (random_state=42)       (1–3 gram,│
                │   lowercase, stop-                              75K feat)│
                │   words, lemmatize)                                 │    │
                │                                                     ▼    │
                │        ┌──────────────┬───────────────┬─────────────────┐│
                │        │ LogisticReg  │  LinearSVC    │ PassiveAggress. ││
                │        └──────┬───────┴───────┬───────┴────────┬────────┘│
                │               └──── VotingClassifier (hard) ───┘         │
                │                              │                           │
                │                     joblib → models/                     │
                └──────────────────────────────────────────────────────────┘

                ┌──────────────────────── INFERENCE ───────────────────────┐
  User text ──► │  TextPreprocessor → vectorizer.transform → 4 predictions │
                └──────────────────────────────────────────────────────────┘
```

At inference time the same `TextPreprocessor` and the **fitted** vectorizer are reused, so text is cleaned and vectorized exactly as it was during training.

---

## Preprocessing & Feature Engineering

### Text Cleaning (`src/preprocess.py`)

Each review passes through these steps, in order:

1. **HTML stripping.** `BeautifulSoup(..., "html.parser").get_text()` removes markup such as the `<br />` tags common in IMDB reviews.
2. **Punctuation removal.** Every character in `string.punctuation` is dropped.
3. **Lowercasing.** Done in the same pass as punctuation removal.
4. **Tokenization.** NLTK `word_tokenize`.
5. **Stopword filtering.** Tokens in the NLTK English stopword list are removed, **except negations** (`not`, `no`, `nor`, `never`, `neither`, `cannot`), which flip sentiment and are kept.
6. **WordNet lemmatization.** `WordNetLemmatizer` reduces inflected forms to base forms (`movies → movie`).

```text
Input : "This movie was <br /><br />NOT good. I would never watch it again!"
Output: "movie not good would never watch"
```

### Vectorization (`src/train.py`)

```python
TfidfVectorizer(
    max_features=75000,   # vocabulary capped at the 75K most frequent terms
    ngram_range=(1, 3),   # unigrams, bigrams and trigrams
    sublinear_tf=True,    # tf → 1 + log(tf), damps repeated terms
)
```

- **Word 1–3 grams** capture short phrases such as *"waste of time"* or *"highly recommend"* that single words miss.
- **Sublinear TF** keeps long reviews that repeat one word from dominating the feature space.
- **75K-feature cap** limits memory use and noise while keeping most of the discriminative vocabulary.
- The vectorizer is **fit on the training split only**, so no information leaks from the test set.

---

## Models

All models are linear, which suits high-dimensional, sparse TF-IDF features: they train fast and are hard to overfit.

| Model | Hyperparameters | Role |
|---|---|---|
| `LogisticRegression` | `C=3.0`, `max_iter=2000` | Probabilistic linear baseline, mild regularization |
| `LinearSVC` | `C=0.4`, `max_iter=3000` | Max-margin classifier; strongest single model |
| `PassiveAggressiveClassifier` | `C=0.3`, `max_iter=1500` | Online, margin-based learner; adds diversity to the ensemble |
| `VotingClassifier` | `voting="hard"` over the three above | Majority vote: a label wins when at least 2 of 3 models agree |

All models use `random_state=42` for reproducibility.

---

## Performance Metrics

Accuracy on a held-out **20% stratified test split** (10,000 reviews):

| Model | Test Accuracy |
|---|:---:|
| Logistic Regression | **~91.04%** |
| Linear Support Vector (SVC) | **~91.17%** |
| Passive Aggressive Classifier | **~90.13%** |
| Ensemble Voting Classifier | **~91.15%** |

**How to read these numbers**

- All three base models land within about 1 point of each other. With good TF-IDF features, the choice of linear model matters less than the feature engineering.
- The **ensemble** smooths over individual model errors. Because it votes on hard labels, its main value is **robustness**: a single model's mistake on an unusual input is outvoted when the other two agree. When all four outputs match in the CLI, the prediction is a high-agreement one.
- Exact figures can vary slightly across `scikit-learn` versions. Retrain to reproduce them on your machine (see [Retraining](#retraining)).

---

## Quick Start

### Prerequisites

- **Python 3.11**
- About 1 GB of free RAM for training (inference needs far less)
- The IMDB dataset at `data/Movie_Reviews.csv` with columns `review` and `sentiment` (`positive` / `negative`). It is available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### 1. Clone the repository

```bash
git clone <your-repo-url> Sentiment_Analysis_Model
cd Sentiment_Analysis_Model
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The required NLTK corpora (`stopwords`, `wordnet`, `punkt`, `punkt_tab`) download automatically on first import.

### 4. Run the interactive pipeline

```bash
python3 main.py
```

- **First run** (no files in `models/`): the full training pipeline runs, test accuracies are printed, and all artifacts are saved.
- **Later runs**: the saved models load right away and the prompt appears.

---

## Interactive Real-Time Testing

`main.py` takes any free-form text (a movie review, a product review, a news snippet) and returns a verdict from **every** model, so you can see where they agree.

### Example: positive review

```text
$ python3 main.py
Loaded saved models and vectorizer from models/

==================================================
Test Real_time Movie Review / Sentiment Article
==================================================

Enter Movie Review / Text: The cinematography was breathtaking and the performances felt honest. One of the best films I have seen this year.

--- Model Predictions ---
Logistic Regression: Positive Sentiment
Linear Support Vector (SVC): Positive Sentiment
Passive Aggressive: Positive Sentiment
Ensemble Voting Model: Positive Sentiment
```

### Example: negative review

```text
Enter Movie Review / Text: A tedious, predictable mess. The plot dragged and the acting was wooden. I want my two hours back.

--- Model Predictions ---
Logistic Regression: Negative Sentiment
Linear Support Vector (SVC): Negative Sentiment
Passive Aggressive: Negative Sentiment
Ensemble Voting Model: Negative Sentiment
```

### Example: first run (training)

```text
Saved models not found. Initiating full ensemble training pipeline...

[+] Loading dataset from: data/Movie_Reviews.csv
[+] Preprocessing text reviews...
[+] Splitting dataset into Train (80%) and Test (20%)...
[+] Extracting TF-IDF Features (Word N-Grams (1 to 3), 75,000 max features)...

==================================================
      ENSEMBLE & MULTI-MODEL ACCURACY RESULTS
==================================================
[*] Training Logistic Regression...
 -> Logistic Regression Accuracy: 91.04%

[*] Training Linear Support Vector (SVC)...
 -> Linear Support Vector (SVC) Accuracy: 91.17%
...
[✔] All ensemble models and vectorizer saved in 'models/'.
```

Empty input exits cleanly, and `Ctrl+C` ends the session without a traceback.

---

## Programmatic Usage

The inference API can be used from your own code without the CLI:

```python
from src.predict import load_or_train_pipeline, predict_realtime_text

models, vectorizer = load_or_train_pipeline("data/Movie_Reviews.csv", "models")

results = predict_realtime_text(
    "An absolute masterpiece with a moving soundtrack.", models, vectorizer
)
# {'Logistic Regression': 'Positive Sentiment',
#  'Linear Support Vector (SVC)': 'Positive Sentiment',
#  'Passive Aggressive': 'Positive Sentiment',
#  'Ensemble Voting Model': 'Positive Sentiment'}
```

---

## Retraining

Artifacts are only rebuilt when one is missing. To force a full retrain, for example after changing hyperparameters in `src/model.py` or the vectorizer settings in `src/train.py`:

```bash
rm -f models/*.joblib
python3 main.py
```

Or call the trainer directly:

```bash
python3 -c "from src.train import train_and_save_all_models; train_and_save_all_models()"
```

> Artifacts saved with `joblib` depend on the `scikit-learn` version that created them. If you upgrade scikit-learn, retrain.

---

## Known Limitations & Roadmap

| Area | Limitation | Possible improvement |
|---|---|---|
| **Confidence** | Hard voting and `LinearSVC` give labels, not probabilities. | Wrap the SVC in `CalibratedClassifierCV` and use `voting="soft"`. |
| **Domain** | Trained on movie reviews; accuracy may drop on other domains. | Fine-tune on in-domain data or add domain adaptation. |
| **Library deprecation** | `PassiveAggressiveClassifier` is deprecated in scikit-learn 1.8 and slated for removal in 1.10. `requirements.txt` pins `scikit-learn<1.10`. | Switch to `SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0)` to lift the pin. |
| **Reproducibility** | Only scikit-learn has a version bound. | Pin exact versions (`pip freeze > requirements.txt`). |
| **Evaluation** | Only accuracy is reported. | Add precision, recall, F1 and a confusion matrix. |
| **Serving** | CLI only. | Add a REST API (FastAPI) and a Dockerfile. |

---

## Tech Stack

`Python 3.11` · `scikit-learn` · `NLTK` · `BeautifulSoup4` · `pandas` · `NumPy` · `joblib`

## Author

**Abdullah Haroon Khan**
