import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.model import get_multi_models
from src.preprocess import TextPreprocessor


def train_and_save_all_models(
    data_path: str = "data/Movie_Reviews.csv", models_dir: str = "models"
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at '{data_path}'.")

    os.makedirs(models_dir, exist_ok=True)

    print(f"\n[+] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path).dropna()

    print("[+] Preprocessing text reviews...")
    preprocessor = TextPreprocessor()
    X = df["review"].apply(preprocessor.clean_text)
    y = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

    print("[+] Splitting dataset into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[+] Extracting TF-IDF Features (Word N-Grams (1 to 3), 75,000 max features)...")
    vectorizer = TfidfVectorizer(
        max_features=75000, ngram_range=(1, 3), sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Save Vectorizer
    vectorizer_path = os.path.join(models_dir, "vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)

    models = get_multi_models()
    trained_models = {}

    print("\n==================================================")
    print("      ENSEMBLE & MULTI-MODEL ACCURACY RESULTS     ")
    print("==================================================")

    for name, model in models.items():
        print(f"[*] Training {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f" -> {name} Accuracy: {acc * 100:.2f}%\n")

        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_filename = f"{clean_name}.joblib"

        joblib.dump(model, os.path.join(models_dir, model_filename))
        trained_models[name] = model

    print(f"[✔] All ensemble models and vectorizer saved in '{models_dir}/'.")
    return trained_models, vectorizer
