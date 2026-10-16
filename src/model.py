from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC


def get_multi_models():
    """
    Returns individual high-performing classifiers and an
    Ensemble Voting Classifier combining all top models.
    """
    # Top 3 Individual Models
    lr = LogisticRegression(C=3.0, max_iter=2000, random_state=42)
    svc = LinearSVC(C=0.4, max_iter=3000, random_state=42)
    pac = PassiveAggressiveClassifier(C=0.3, max_iter=1500, random_state=42)

    # Combined Ensemble Voting Model
    ensemble = VotingClassifier(
        estimators=[
            ("logistic_regression", lr),
            ("linear_svc", svc),
            ("passive_aggressive", pac),
        ],
        voting="hard",  # Majority voting mechanism
    )

    models = {
        "Logistic Regression": lr,
        "Linear Support Vector (SVC)": svc,
        "Passive Aggressive": pac,
        "Ensemble Voting Model": ensemble,
    }
    return models
