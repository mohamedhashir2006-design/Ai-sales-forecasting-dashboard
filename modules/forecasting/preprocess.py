import pandas as pd

def load_and_prepare_data(path):
    # Load CSV
    df = pd.read_csv(path, parse_dates=["date"])

    # Sort by date
    df = df.sort_values("date")

    # Aggregate (IMPORTANT)
    df = df.groupby("date")["revenue_usd"].sum().reset_index()

    # Rename for Prophet
    df.columns = ["ds", "y"]

    return df