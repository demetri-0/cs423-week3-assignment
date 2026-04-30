from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("WestRoxbury.csv")
OUTPUT_PATH = Path("WestRoxbury_preprocessed.csv")
TARGET_COLUMN = "TOTAL_VALUE"
K_FOLDS = 5


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


def calculate_mse(
    model: LinearRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[float, float]:
    """Calculate mean squared error for train and test predictions."""
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))

    return train_mse, test_mse


def cross_validate_linear_regression(
    df: pd.DataFrame,
    k: int = K_FOLDS,
) -> tuple[np.ndarray, float, float]:
    """Run k-fold cross-validation and summarize RMSE scores."""
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]
    numeric_columns = X.select_dtypes(include="number").columns[
        ~X.select_dtypes(include="number").columns.str.startswith("REMODEL_")
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_numeric", StandardScaler(), numeric_columns),
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("linear_regression", LinearRegression()),
        ]
    )

    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=k,
        scoring="neg_root_mean_squared_error",
    )
    rmse_scores = -scores
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)

    return rmse_scores, mean_rmse, std_rmse


def build_preprocessed_output(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Combine features and target for inspection/export."""
    return pd.concat([y.rename(TARGET_COLUMN), X], axis=1)


def main() -> None:
    cleaned_df = load_west_roxbury()
    print_data_diagnostics(cleaned_df)

    X_train, X_test, y_train, y_test = split_features_target(cleaned_df)
    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test)
    model = fit_linear_regression(X_train_scaled, y_train)
    rmse_scores, mean_rmse, std_rmse = cross_validate_linear_regression(cleaned_df)
    train_mse, test_mse = calculate_mse(
        model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

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
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"{K_FOLDS}-fold CV RMSE scores: {rmse_scores.round(4)}")
    print(f"Mean CV RMSE: {mean_rmse:.4f}")
    print(f"CV RMSE standard deviation: {std_rmse:.4f}")
    print(f"Scaled numeric feature columns: {', '.join(scaler.feature_names_in_)}")
    print("Columns:")
    print(", ".join(preprocessed_df.columns))


if __name__ == "__main__":
    main()
