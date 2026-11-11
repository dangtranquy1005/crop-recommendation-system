from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from config import BASE_DIR, RANDOM_STATE,PROCESSED_DATA_DIR, FEATURES
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.metrics import classification_report


def load_training_data():
    """Doc du lieu da chia tu preprocessing.py."""

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )
    X_val = pd.read_csv(
        PROCESSED_DATA_DIR / "X_val.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DATA_DIR / "y_train.csv"
    )["label"]

    y_val = pd.read_csv(
        PROCESSED_DATA_DIR / "y_val.csv"
    )["label"]

    # Dam bao thu tu dac trung giong nhau
    X_train = X_train[FEATURES]
    X_val = X_val[FEATURES]

    return X_train, X_val, y_train, y_val


def evaluate_model(model, X, y, name):
    """Danh gia mo hinh tren mot tap du lieu."""

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(
        y, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y, y_pred, average="macro", zero_division=0
    )

    print(f"\n===== {name} =====")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    return accuracy, f1



def build_logistic_model():
    """Tao Pipeline Logistic Regression."""

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=0.1,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ])

    return model


def main():
    #Doc du lieu
    X_train, X_val, y_train, y_val = load_training_data()

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    # Tao mo hinh
    model = build_logistic_model()

    # Huan luyen
    model.fit(X_train, y_train)

    print("\nHuan luyen Logistic Regression thanh cong!")

    # Danh gia tren Train
    train_acc, train_f1 = evaluate_model(
        model, X_train, y_train, "LOGISTIC TRAIN"
    )

    # Danh gia tren Validation
    val_acc, val_f1 = evaluate_model(
        model, X_val, y_val, "LOGISTIC VALIDATION"
    )

    # Kiem tra chenhlech
    print(
        "\nChenh lech Accuracy:",
        round(train_acc - val_acc, 4)
    )

    # Classification Report
    y_pred = model.predict(X_val)

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        zero_division=0
    ))

    # Luu mo hinh
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "logistic_model.joblib"
    dump(model, model_path)

    print("\nDa luu mo hinh tai:", model_path)


if __name__ == "__main__":
    main()