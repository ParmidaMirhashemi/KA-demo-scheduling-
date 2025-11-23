import sys
from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from sessions_data import load_training_data


RANDOM_SEED = 42


def infer_column_types(X: pd.DataFrame) -> (List[str], List[str]):
    """
    Infer numeric vs categorical columns from a feature DataFrame.

    - Numeric: int, float, bool
    - Categorical: object, category, anything else
    """
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a sklearn Pipeline for predicting number of appointments."""
    numeric_features, categorical_features = infer_column_types(X)

    if not numeric_features and not categorical_features:
        raise ValueError("No features to train on. Check your CSV.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe


def train_model_from_csv(
    csv_path: str,
    model_path: str = "sessions_model.joblib",
    target_column: str = "num_appointments",
    id_column: str = "patient_id",
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """
    Train a model from a CSV file and save it.

    Parameters
    ----------
    csv_path : str
        Path to CSV containing features and target.
    model_path : str
        File path to save the trained model pipeline.
    target_column : str
        Name of the target column (number of appointments).
    id_column : str
        Name of optional ID column (ignored as feature).
    test_size : float
        Proportion of data to use as test set.
    random_state : int
        Random seed for train/test split.

    Returns
    -------
    model : Pipeline
        The fitted sklearn Pipeline.
    """
    X, y = load_training_data(
        csv_path,
        target_column=target_column,
        id_column=id_column,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_model_pipeline(X_train)

    print(f"[INFO] Training model on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[METRIC] Test RMSE: {rmse:.3f}")
    print(f"[METRIC] Test MAE:  {mae:.3f}")
    print(f"[METRIC] Test R^2:  {r2:.3f}")

    joblib.dump(model, model_path)
    print(f"[INFO] Trained model saved to {model_path}")

    return model


def main(args: Optional[list] = None):
    """
    CLI entrypoint.

    Usage:
        python sessions_model_train.py training_data.csv [model_path] [target_column] [id_column]
    """
    if args is None:
        args = sys.argv[1:]

    if not args:
        print(
            "Usage: python sessions_model_train.py "
            "training_data.csv [model_path] [target_column] [id_column]"
        )
        sys.exit(1)

    csv_path = args[0]
    model_path = args[1] if len(args) > 1 else "sessions_model.joblib"
    target_column = args[2] if len(args) > 2 else "num_appointments"
    id_column = args[3] if len(args) > 3 else "patient_id"

    train_model_from_csv(
        csv_path=csv_path,
        model_path=model_path,
        target_column=target_column,
        id_column=id_column,
    )


if __name__ == "__main__":
    main()
