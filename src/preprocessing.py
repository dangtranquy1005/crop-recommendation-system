
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    FEATURES,
    TARGET,
    RANDOM_STATE,
)


def load_data():
    """Doc dataset goc."""
    df = pd.read_csv(RAW_DATA_PATH)

    print("So dong, so cot:", df.shape)
    print("\n5 dong dau:")
    print(df.head())

    return df


def inspect_data(df):
    """Kiem tra chat luong du lieu."""
    print("\n--- THONG TIN DATASET ---")
    df.info()

    print("\nGia tri thieu:")
    print(df.isnull().sum())

    print("\nSo dong trung lap:")
    print(df.duplicated().sum())

    print("\nPhan bo cac loai cay:")
    print(df[TARGET].value_counts())

    print("\nThong ke dac trung:")
    print(df[FEATURES].describe())


def clean_data(df):
    """Lam sach du lieu co ban."""
    df = df.copy()

    # Loai bo dong trung hoan toan
    df = df.drop_duplicates()

    # Chuyen dac trung sang kieu so
    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Loai bo dong thieu nhan
    df = df.dropna(subset=[TARGET])

    # Loai bo gia tri khong hop le ve vat ly
    valid = (
        df["N"].ge(0) &
        df["P"].ge(0) &
        df["K"].ge(0) &
        df["humidity"].between(0, 100) &
        df["ph"].between(0, 14) &
        df["rainfall"].ge(0)
    )

    # Giu lai dong thieu dac trung de
    # xu ly bang SimpleImputer khi huan luyen
    valid = valid | df[FEATURES].isna().any(axis=1)

    df = df[valid].copy()

    print("\nSo mau sau lam sach:", len(df))
    return df


def split_data(df):
    """Chia Train 70%, Validation 15%, Test 15%."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print("\n--- KICH THUOC DU LIEU ---")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data(splits):
    """Luu cac tap du lieu sau khi chia."""
    PROCESSED_DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    names = [
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test"
    ]

    for name, data in zip(names, splits):
        data.to_csv(
            PROCESSED_DATA_DIR / f"{name}.csv",
            index=False
        )

    print("\nDa luu du lieu vao data/processed/")


def main():
    df = load_data()
    inspect_data(df)

    df = clean_data(df)
    splits = split_data(df)
    save_data(splits)


if __name__ == "__main__":
    main()
