import pandas as pd
from typing import Tuple, List


def get_feature_columns(
    df: pd.DataFrame,
    target_column: str = "num_appointments",
    id_column: str = "patient_id",
) -> List[str]:
    """
    Return the list of feature columns given a dataframe.

    All columns except `target_column` and `id_column` are treated as features.
    """
    cols_to_exclude = {target_column}
    if id_column in df.columns:
        cols_to_exclude.add(id_column)
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    if not feature_cols:
        raise ValueError("No feature columns found. Check your CSV and column names.")
    return feature_cols


def load_training_data(
    csv_path: str,
    target_column: str = "num_appointments",
    id_column: str = "patient_id",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data from CSV.

    The CSV is expected to contain:
      - one column with the target number of appointments (`target_column`)
      - optionally an ID column (`id_column`)
      - all other columns are treated as features.

    Returns
    -------
    X : DataFrame
        Feature matrix.
    y : Series
        Target values (number of appointments).
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in CSV. "
            f"Columns: {list(df.columns)}"
        )

    feature_cols = get_feature_columns(
        df, target_column=target_column, id_column=id_column
    )

    X = df[feature_cols].copy()
    y = df[target_column].astype(float)
    return X, y


def load_features_for_prediction(
    csv_path: str,
    target_column: str = "num_appointments",
    id_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Load feature data from CSV for prediction.

    If `target_column` is present, it is dropped.
    If `id_column` is present, it is kept so the caller can join predictions back.

    Returns
    -------
    df_features : DataFrame
        DataFrame containing at least all feature columns and optionally ID.
        The model code is responsible for dropping any ID column before prediction.
    """
    df = pd.read_csv(csv_path)

    # Drop target if present
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    if df.empty:
        raise ValueError("No columns left after dropping target. Check your CSV.")

    return df
