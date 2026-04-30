from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("WestRoxbury.csv")
OUTPUT_PATH = Path("WestRoxbury_preprocessed.csv")
TARGET_COLUMN = "TOTAL_VALUE"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dataframe column names."""
    return df.rename(columns=lambda column: column.strip().replace(" ", "_"))


def load_west_roxbury(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the West Roxbury housing dataset and apply non-leaky cleaning."""
    return (
        pd.read_csv(data_path)
        .pipe(clean_column_names)
        .drop(columns=["TAX"], errors="ignore")
        .assign(REMODEL=lambda df_: df_["REMODEL"].fillna("None"))
        .pipe(pd.get_dummies, columns=["REMODEL"], prefix="REMODEL", dtype=int)
    )


def print_data_diagnostics(df: pd.DataFrame) -> None:
    """Show basic defensive parsing checks after ingestion."""
    print("Data types:")
    print(df.dtypes)
    print("\nFirst five rows:")
    print(df.head())
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partition the data into 60% training and 40% test sets."""
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=1,
    )


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit the scaler on training data, then transform train and test features."""
    numeric_columns = X_train.select_dtypes(include="number").columns[
        ~X_train.select_dtypes(include="number").columns.str.startswith("REMODEL_")
    ]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    return X_train_scaled, X_test_scaled, scaler


def fit_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit a linear regression model to the training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def build_preprocessed_output(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Combine features and target for inspection/export."""
    return pd.concat([y.rename(TARGET_COLUMN), X], axis=1)


def main() -> None:
    cleaned_df = load_west_roxbury()
    print_data_diagnostics(cleaned_df)

    X_train, X_test, y_train, y_test = split_features_target(cleaned_df)
    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test)
    model = fit_linear_regression(X_train_scaled, y_train)

    preprocessed_df = build_preprocessed_output(
        pd.concat([X_train_scaled, X_test_scaled]).sort_index(),
        pd.concat([y_train, y_test]).sort_index(),
    )
    preprocessed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved preprocessed data to {OUTPUT_PATH}")
    print(f"Rows: {preprocessed_df.shape[0]}, columns: {preprocessed_df.shape[1]}")
    print(f"Training rows: {X_train_scaled.shape[0]}, test rows: {X_test_scaled.shape[0]}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Linear regression intercept: {model.intercept_:.4f}")
    print(f"Scaled numeric feature columns: {', '.join(scaler.feature_names_in_)}")
    print("Columns:")
    print(", ".join(preprocessed_df.columns))


if __name__ == "__main__":
    main()
