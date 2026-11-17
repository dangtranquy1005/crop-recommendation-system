from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import BASE_DIR, RANDOM_STATE
from feature_engineering import build_crop_pipeline
from trainlda import load_training_data


def build_random_forest_model(use_engineered_features=True):
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )
    return build_crop_pipeline(classifier, use_engineered_features)


def main():
    X_train, X_val, y_train, y_val = load_training_data()
    model = build_random_forest_model().fit(X_train, y_train)

    for name, X, y in (("Train", X_train, y_train), ("Validation", X_val, y_val)):
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        macro_f1 = f1_score(y, predictions, average="macro")
        print(f"{name} ({len(y)} samples) | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

    model_path = BASE_DIR / "models" / "random_forest_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"Saved pipeline: {model_path}")


if __name__ == "__main__":
    main()
