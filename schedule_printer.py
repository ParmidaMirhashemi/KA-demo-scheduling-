import pandas as pd


def print_and_save_schedule(schedule_df: pd.DataFrame, output_path: str):
    """
    Pretty-print the schedule DataFrame and save to CSV.

    The DataFrame is expected to have patients as the index and time slots as columns.
    """
    print("\n=== Optimized Schedule (rows=patients, cols=time slots) ===")
    print(schedule_df)
    schedule_df.to_csv(output_path)
    print(f"\n[INFO] Schedule saved to {output_path}")
