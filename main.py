from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("WestRoxbury.csv")
OUTPUT_PATH = Path("WestRoxbury_preprocessed.csv")


def preprocess_west_roxbury(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the West Roxbury housing dataset."""
    df = pd.read_csv(data_path)

    # Clean column names so they are easier to reference in code.
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

    if "TAX" in df.columns:
        df = df.drop(columns="TAX")

    numeric_columns = df.select_dtypes(include="number").columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    if "REMODEL" in df.columns:
        df["REMODEL"] = df["REMODEL"].fillna("None")
        df = pd.get_dummies(df, columns=["REMODEL"], prefix="REMODEL", dtype=int)

    return df


def main() -> None:
    preprocessed_df = preprocess_west_roxbury()
    preprocessed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved preprocessed data to {OUTPUT_PATH}")
    print(f"Rows: {preprocessed_df.shape[0]}, columns: {preprocessed_df.shape[1]}")
    print("Columns:")
    print(", ".join(preprocessed_df.columns))


if __name__ == "__main__":
    main()
