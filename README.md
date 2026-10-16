# Sentiment Analysis Model

A modular Machine Learning pipeline that classifies movie reviews and user text as **Positive** or **Negative** sentiment. Reviews are vectorized with TF-IDF and four Scikit-Learn classifiers (Logistic Regression, Linear Support Vector / SVC, Passive Aggressive, and an Ensemble Voting Classifier) are trained and compared. Models reach **~91% accuracy** on the held-out test set. Trained models are saved to disk, so later runs load them in seconds instead of retraining.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python 3.11+ |
| Data processing / ML | Scikit-Learn, Pandas, NumPy |
| NLP | NLTK (stopwords, tokenizer, WordNet lemmatizer), BeautifulSoup |
| Vectorization | TF-IDF (`TfidfVectorizer`, word n-grams 1–3, `sublinear_tf=True`, 75,000 features) |
| Models evaluated | Logistic Regression, Linear SVC, Passive Aggressive, Ensemble Voting Classifier |
| Persistence | joblib |

## Pipeline

1. **Loading and labeling** (`src/train.py`): reads `data/Movie_Reviews.csv` and encodes sentiment labels (`positive` = `1`, `negative` = `0`).
2. **Text cleaning** (`TextPreprocessor.clean_text` in `src/preprocess.py`): strips HTML tags with BeautifulSoup, removes punctuation, lowercases, filters stopwords while **keeping negations** (`not`, `no`, `nor`, `never`, `neither`, `cannot`) and applies WordNet lemmatization.
3. **Train/test split**: 80/20 stratified split (`random_state=42`).
4. **Feature extraction**: `TfidfVectorizer(max_features=75000, ngram_range=(1, 3), sublinear_tf=True)`, fit on the training set only, then applied to the test set.
5. **Model training and comparison** (`src/train.py`, models defined in `src/model.py`): trains the three base classifiers and a hard-voting ensemble of all three, and prints each model's test accuracy.
6. **Persistence**: saves the fitted vectorizer and every trained model to `models/`. On later runs, `src/predict.py` loads them directly.
7. **Real-time inference**: `predict_realtime_text` cleans and vectorizes raw input text, then returns each model's prediction.

### Results

Test-set accuracies on **10,000** held-out movie reviews:

| Model | Accuracy |
| --- | :---: |
| Logistic Regression | 91.04% |
| Linear Support Vector (SVC) | 91.17% |
| Passive Aggressive Classifier | 90.13% |
| Ensemble Voting Classifier | 91.15% |

The ensemble uses hard (majority) voting: a label wins when at least 2 of the 3 base models agree.

## Project Structure

```
Sentiment_Analysis_Model/
├── data/
│   └── Movie_Reviews.csv           # IMDB 50K Reviews (not tracked in git)
├── models/                         # Created on first run (not tracked in git)
│   ├── vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── linear_support_vector_svc.joblib
│   ├── passive_aggressive.joblib
│   └── ensemble_voting_model.joblib
├── src/
│   ├── __init__.py
│   ├── preprocess.py               # Text cleaning, HTML removal, negation-aware stopwords & lemmatization
│   ├── model.py                    # Multi-model definitions & VotingClassifier configuration
│   ├── train.py                    # Dataset loading, TF-IDF feature extraction, model training & persistence
│   └── predict.py                  # Load-or-train pipeline & real-time prediction helpers
├── main.py                         # Entry point: loads/trains models, then runs the interactive prompt
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repository-url>
cd Sentiment_Analysis_Model

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The required NLTK data (`stopwords`, `wordnet`, `punkt`, `punkt_tab`) downloads automatically on first run. `requirements.txt` pins `scikit-learn<1.10`, because `PassiveAggressiveClassifier` is removed in 1.10.

### Dataset

`data/*.csv` is excluded by `.gitignore`. Download the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle and save it as:

```
data/Movie_Reviews.csv
```

The file needs the columns `review` and `sentiment` (`positive` / `negative`).

## Usage

```bash
python3 main.py             # Load models from models/, or train and save them if missing
```

The first run trains all four models and saves them, which takes a few minutes. Later runs load the saved files in seconds.

To retrain, for example after changing `clean_text`, the vectorizer settings or model hyperparameters, delete the saved artifacts and run again:

```bash
rm -f models/*.joblib
python3 main.py
```

When the models are ready, the script asks for text:

```
Enter Movie Review / Text: This movie was not good at all.

--- Model Predictions ---
Logistic Regression: Negative Sentiment
Linear Support Vector (SVC): Negative Sentiment
Passive Aggressive: Negative Sentiment
Ensemble Voting Model: Negative Sentiment
```

### Using the modules directly

```python
from src.predict import load_or_train_pipeline, predict_realtime_text

models, vectorizer = load_or_train_pipeline()   # loads models/, or trains if any file is missing
print(predict_realtime_text("An absolute masterpiece.", models, vectorizer))
```

To train from scratch in code:

```python
from src.train import train_and_save_all_models

models, vectorizer = train_and_save_all_models("data/Movie_Reviews.csv", "models")
```

## Author

Abdullah Haroon Khan
