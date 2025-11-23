"""
End-to-end script:
- Load a trained demand model.
- Predict number of appointments for new patients from a features CSV.
- Merge in already-booked sessions from a fixed schedule CSV.
- Build and solve the optimization model.
- Print/export the resulting schedule.
"""

import argparse
import math
import pandas as pd
import numpy as np

from sessions_model_predict import predict_from_features_df, load_model
from sessions_data import load_features_for_prediction, get_feature_columns
from Optimization_model import create_optimization_model, visualize_schedule
from schedule_printer import print_and_save_schedule


def build_fixed_x(P, T, fixed_df: pd.DataFrame) -> dict:
    """
    Build fixed_x dictionary from CSV with columns:
        patient_id, time_slot, value (0/1)
    Missing rows default to 0.
    """
    required_cols = {"patient_id", "time_slot"}
    if not required_cols.issubset(set(fixed_df.columns)):
        raise ValueError(
            "fixed schedule CSV must have columns: patient_id, time_slot, (optional) value"
        )

    if "value" not in fixed_df.columns:
        fixed_df["value"] = 1

    fixed_x = {(p, t): 0 for p in P for t in T}
    for _, row in fixed_df.iterrows():
        pid = row["patient_id"]
        t = int(row["time_slot"])
        val = int(row["value"])
        if pid not in P or t not in T:
            continue
        fixed_x[(pid, t)] = val
    return fixed_x


def infer_t_fixed(fixed_df: pd.DataFrame) -> int:
    """Return the last fixed slot + 1 (or 0 if none)."""
    if fixed_df.empty:
        return 0
    return int(fixed_df["time_slot"].max()) + 1


def build_capacity(T, capacity_df: pd.DataFrame | None, default_cap: int) -> dict:
    if capacity_df is None or capacity_df.empty:
        return {t: default_cap for t in T}
    if not {"time_slot", "capacity"}.issubset(set(capacity_df.columns)):
        raise ValueError("capacity CSV must have columns: time_slot, capacity")
    cap_map = {int(r.time_slot): int(r.capacity) for r in capacity_df.itertuples()}
    return {t: cap_map.get(t, default_cap) for t in T}


def build_costs(P, T, sigma, delta):
    alpha = {(p, j): max(0, 20 * (sigma[p] + 0.1) + 1 - 20 * j) for p in P for j in range(1, delta)}
    beta = {(p, t): (p.__hash__() % 10 + 1) * math.log(t + 2) for p in P for t in T}
    gamma = {}
    for p in P:
        for l in range(1, sigma[p] - 1):
            for i in range(0, l):
                gamma[(p, i, l)] = (l - i) * math.log(l - i + 1) / 50
    return alpha, beta, gamma


def main():
    parser = argparse.ArgumentParser(description="Predict demand and optimize scheduling.")
    parser.add_argument("features_csv", help="CSV with new patient features (must include patient_id).")
    parser.add_argument("fixed_schedule_csv", help="CSV with already booked sessions (patient_id,time_slot[,value]).")
    parser.add_argument(
        "--model-path", default="sessions_model.joblib", help="Path to trained model pipeline."
    )
    parser.add_argument(
        "--capacity-csv",
        default=None,
        help="Optional CSV with time_slot,capacity columns. Defaults to constant capacity.",
    )
    parser.add_argument("--default-capacity", type=int, default=4, help="Fallback capacity per time slot.")
    parser.add_argument("--horizon", type=int, default=20, help="Number of time slots in the planning horizon.")
    parser.add_argument("--tau", type=int, default=4, help="Slots after t_fixed that must respect fixed_x.")
    parser.add_argument("--binary", action="store_true", help="Use binary decision variables instead of continuous.")
    parser.add_argument("--output-csv", default="optimized_schedule.csv", help="Where to save the schedule matrix.")
    args = parser.parse_args()

    # Load features and model, predict demand
    features_df = load_features_for_prediction(args.features_csv, id_column="patient_id")
    model = load_model(args.model_path)
    preds = predict_from_features_df(features_df, model=model, id_column="patient_id")

    patient_ids = features_df["patient_id"].astype(str).tolist()
    P = patient_ids
    T = list(range(args.horizon))
    delta = args.horizon
    sigma = {pid: max(1, int(math.ceil(p))) for pid, p in zip(P, preds)}

    # Build fixed assignments and parameters
    fixed_df = pd.read_csv(args.fixed_schedule_csv)
    fixed_x = build_fixed_x(P, T, fixed_df)
    t_fixed = infer_t_fixed(fixed_df)

    capacity_df = pd.read_csv(args.capacity_csv) if args.capacity_csv else None
    kappa = build_capacity(T, capacity_df, args.default_capacity)

    alpha, beta, gamma = build_costs(P, T, sigma, delta)

    # Optimize
    model = create_optimization_model(
        P=P,
        T=T,
        sigma=sigma,
        kappa=kappa,
        delta=delta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        fixed_x=fixed_x,
        t_fixed=t_fixed,
        tau=args.tau,
        Binary=args.binary,
    )
    model.optimize()

    schedule_df = visualize_schedule(model, P, T)
    print_and_save_schedule(schedule_df, args.output_csv)


if __name__ == "__main__":
    main()
