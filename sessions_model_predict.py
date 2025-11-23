import sys
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline  # type hints only

from sessions_data import load_features_for_prediction, get_feature_columns


def load_model(model_path: str = "sessions_model.joblib") -> Pipeline:
    """Load a trained sessions model pipeline from disk."""
    model: Pipeline = joblib.load(model_path)
    return model


def predict_from_features_df(
    features_df: pd.DataFrame,
    model: Optional[Pipeline] = None,
    model_path: str = "sessions_model.joblib",
    target_column: str = "num_appointments",
    id_column: str = "patient_id",
) -> np.ndarray:
    """
    Predict number of appointments from a features DataFrame.

    Parameters
    ----------
    features_df : DataFrame
        DataFrame containing feature columns and optionally an ID column.
    model : Pipeline, optional
        Pre-loaded model. If None, will be loaded from `model_path`.
    model_path : str
        Path to the saved model pipeline (ignored if `model` is provided).
    target_column : str
        Name of target column (if present, it will be ignored).
    id_column : str
        Name of ID column (if present, it is ignored for prediction).

    Returns
    -------
    preds : np.ndarray
        1D array of predicted number of appointments.
    """
    if model is None:
        model = load_model(model_path)

    df = features_df.copy()
    # Drop target if somehow present
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    # Treat everything except ID as features
    feature_cols = get_feature_columns(df, target_column="__dummy__", id_column=id_column)
    X = df[feature_cols]

    preds = model.predict(X)
    return np.asarray(preds)


def predict_from_csv(
    csv_path: str,
    model_path: str = "sessions_model.joblib",
    target_column: str = "num_appointments",
    id_column: str = "patient_id",
    return_ids: bool = False,
):
    """
    Predict number of appointments given a CSV of features.

    If `return_ids` is True and an ID column is present, returns a tuple
    (patient_ids, predictions). Otherwise, returns only the predictions array.
    """
    df = load_features_for_prediction(
        csv_path=csv_path,
        target_column=target_column,
        id_column=id_column,
    )

    model = load_model(model_path)
    preds = predict_from_features_df(
        features_df=df,
        model=model,
        target_column=target_column,
        id_column=id_column,
    )

    if return_ids and id_column in df.columns:
        patient_ids = df[id_column].values
        return patient_ids, preds
    return preds


def main(args: Optional[list] = None):
    """
    CLI entrypoint.

    Usage:
        python sessions_model_predict.py features.csv [model_path] [id_column]
    """
    if args is None:
        args = sys.argv[1:]

    if not args:
        print(
            "Usage: python sessions_model_predict.py "
            "features.csv [model_path] [id_column]"
        )
        sys.exit(1)

    csv_path = args[0]
    model_path = args[1] if len(args) > 1 else "sessions_model.joblib"
    id_column = args[2] if len(args) > 2 else "patient_id"

    result = predict_from_csv(
        csv_path=csv_path,
        model_path=model_path,
        target_column="num_appointments",
        id_column=id_column,
        return_ids=True,
    )

    if isinstance(result, tuple):
        ids, preds = result
        for pid, p in zip(ids, preds):
            print(f"patient_id={pid}, predicted_num_appointments={p:.2f}")
    else:
        preds = result
        for i, p in enumerate(preds):
            print(f"row={i}, predicted_num_appointments={p:.2f}")


if __name__ == "__main__":
    main()
