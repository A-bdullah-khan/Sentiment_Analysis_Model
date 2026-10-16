import os

import joblib

from src.preprocess import TextPreprocessor


def load_or_train_pipeline(
    data_path: str = "data/Movie_Reviews.csv", models_dir: str = "models"
):
    expected_models = [
        "logistic_regression",
        "linear_support_vector_svc",
        "passive_aggressive",
        "ensemble_voting_model",
    ]
    vectorizer_path = os.path.join(models_dir, "vectorizer.joblib")

    all_exist = os.path.exists(vectorizer_path) and all(
        os.path.exists(os.path.join(models_dir, f"{m}.joblib")) for m in expected_models
    )

    if all_exist:
        print(f"Loaded saved models and vectorizer from {models_dir}/\n")
        vectorizer = joblib.load(vectorizer_path)
        models = {
            "Logistic Regression": joblib.load(
                os.path.join(models_dir, "logistic_regression.joblib")
            ),
            "Linear Support Vector (SVC)": joblib.load(
                os.path.join(models_dir, "linear_support_vector_svc.joblib")
            ),
            "Passive Aggressive": joblib.load(
                os.path.join(models_dir, "passive_aggressive.joblib")
            ),
            "Ensemble Voting Model": joblib.load(
                os.path.join(models_dir, "ensemble_voting_model.joblib")
            ),
        }
        return models, vectorizer
    else:
        print("Saved models not found. Initiating full ensemble training pipeline...\n")
        from src.train import train_and_save_all_models

        return train_and_save_all_models(data_path, models_dir)


def predict_realtime_text(text: str, models: dict, vectorizer) -> dict:
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    text_vec = vectorizer.transform([cleaned_text])

    predictions = {}
    for name, model in models.items():
        pred = model.predict(text_vec)[0]
        predictions[name] = "Positive Sentiment" if pred == 1 else "Negative Sentiment"

    return predictions
