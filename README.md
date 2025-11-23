# Scheduling Demo: Predict + Optimize

This folder contains two stages that work together:
1) Learn how many appointments each patient will likely need.
2) Schedule those appointments while respecting existing bookings and capacity limits.

The code is split between a small ML pipeline (for demand prediction) and a mathematical optimization model (for capacity-constrained scheduling).

## Repository layout
- `sessions_data.py` – CSV loaders and feature/target column helpers.
- `sessions_model_train.py` – trains a model that predicts `num_appointments` for each patient.
- `sessions_model_predict.py` – predicts appointment counts for new patients.
- `Optimization_model.py` – Gurobi model that schedules patients given demand, capacity (`kappa`), and already-fixed assignments (`fixed_x`).
- `run_prediction_static_optimization.py` – placeholder for wiring predictions into the optimizer.

## Prerequisites
- Python 3.10+.
- Packages: `pandas`, `numpy`, `scikit-learn`, `joblib`, `gurobipy`. Install via:
  ```
  pip install pandas numpy scikit-learn joblib gurobipy
  ```
- A working Gurobi license for solving the optimization model.

## Data expectations
Training CSV columns:
- `num_appointments` (target) – how many sessions the patient actually needed.
- `patient_id` (optional) – kept for traceability, dropped as a feature.
- All other columns are treated as features (mixed numeric/categorical supported).

Prediction CSV columns:
- Same feature columns as training.
- Optional `patient_id` for later joining predictions back to patients.
- `num_appointments` may be absent; if present, it is ignored for prediction.

## Stage 1: Train the demand model
```
python sessions_model_train.py training_data.csv [model_path] [target_column] [id_column]
```
- Saves a fitted pipeline (preprocessing + `RandomForestRegressor`) to `sessions_model.joblib` by default.
- Prints RMSE/MAE/R² on a hold-out split so you can track model quality.

## Stage 2: Predict appointment counts for new patients
```
python sessions_model_predict.py new_patients.csv [model_path] [id_column]
```
- Outputs `patient_id` (if present) and the predicted `num_appointments`.
- Use these predictions as the demand input to the scheduler.

## Stage 3: Schedule with capacity and fixed bookings
`Optimization_model.py` builds a Gurobi model with:
- Decision variables `x[p,t]` indicating whether patient `p` is scheduled at time `t`.
- Capacity constraints `∑_p x[p,t] ≤ kappa[t]` for each time slot.
- Respect for already-booked slots via `fixed_x` (and `t_fixed` horizon); assignments before `t_fixed` are locked, and assignments immediately after can be constrained with `tau`.
- Objective terms balancing wait times (`beta`), missed sessions (`alpha`), and spacing penalties (`gamma`).

Typical flow to use the optimizer:
1) Prepare sets of patients `P`, time slots `T`, predicted session counts `sigma[p]` (e.g., round predicted `num_appointments`), and capacity per slot `kappa[t]`.
2) Build or load `fixed_x[p,t]` for already scheduled sessions; set `t_fixed` to the last fully fixed time slot.
3) Construct cost coefficients `alpha`, `beta`, `gamma` to reflect your operational preferences.
4) Call `create_optimization_model(...)`, set Gurobi parameters as needed, then `optimize()`.
5) Use `visualize_schedule(model, P, T)` to return a `DataFrame` showing the schedule.

The `__main__` block in `Optimization_model.py` demonstrates a fully synthetic example; adapt it by replacing the synthetic data generation with your predicted demand and real capacities.

## Putting it together end-to-end
1) Train: `python sessions_model_train.py training_data.csv`.
2) Predict: `python sessions_model_predict.py new_patients.csv --model_path sessions_model.joblib --id_column patient_id`.
3) Map predictions to integers (e.g., `sigma[p] = ceil(pred)`), set capacities, and feed them into the Gurobi model along with any pre-booked sessions.
4) Inspect the resulting schedule DataFrame or export it to downstream systems.

## Notes and gaps
- `run_prediction_static_optimization.py` is empty; use it to glue prediction output into the optimizer steps above.
- The example in `Optimization_model.py` references `generate_data`, which is not included here; swap in your own data preparation that produces `P, d, nu, sigma_std, sigma`.
