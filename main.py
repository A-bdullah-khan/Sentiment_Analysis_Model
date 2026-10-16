import sys

from src.predict import load_or_train_pipeline, predict_realtime_text

if __name__ == "__main__":
    DATA_PATH = "data/Movie_Reviews.csv"
    MODELS_DIR = "models"

    # Loading saved artifacts or automatically training if missing
    models, vectorizer = load_or_train_pipeline(DATA_PATH, MODELS_DIR)

    # Displaying Terminal Interface
    print("==================================================")
    print("Test Real_time Movie Review / Sentiment Article")
    print("==================================================\n")

    try:
        user_input = input("Enter Movie Review / Text: ").strip()

        if not user_input:
            print("No text entered. Exiting.")
            sys.exit(0)

        # Step 3: Get Predictions across all models
        results = predict_realtime_text(user_input, models, vectorizer)

        print("\n--- Model Predictions ---")
        for model_name, prediction in results.items():
            print(f"{model_name}: {prediction}")

    except KeyboardInterrupt:
        print("\nSession interrupted.")
        sys.exit(0)
