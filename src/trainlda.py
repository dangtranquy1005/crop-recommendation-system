
import pandas as pd
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    FEATURES
)


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


def build_model():
    """Tao Pipeline cho LDA."""

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LinearDiscriminantAnalysis())
    ])

    return model


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


def main():
    # Doc du lieu
    X_train, X_val, y_train, y_val = load_training_data()

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    # Tao mo hinh
    model = build_model()

    # Huan luyen LDA
    model.fit(X_train, y_train)

    print("\nHuan luyen LDA thanh cong!")

    # Danh gia tren Train
    train_acc, train_f1 = evaluate_model(
        model, X_train, y_train, "TRAIN"
    )

    # Danh gia tren Validation
    val_acc, val_f1 = evaluate_model(
        model, X_val, y_val, "VALIDATION"
    )

    # Kiem tra chenhlech train-validation
    print("\nChenh lech Accuracy:",
          round(train_acc - val_acc, 4))

    # Bao cao tung loai cay
    y_pred = model.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(
        y_val, y_pred, zero_division=0
    ))

    # Luu mo hinh
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lda_model.joblib"
    dump(model, model_path)

    print("\nDa luu mo hinh tai:", model_path)


if __name__ == "__main__":
    main()
