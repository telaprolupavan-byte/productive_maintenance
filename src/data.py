"""Load the raw Azure PdM tables."""
from pathlib import Path
import pandas as pd

FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors":    "PdM_errors.csv",
    "maint":     "PdM_maint.csv",
    "failures":  "PdM_failures.csv",
    "machines":  "PdM_machines.csv",
}

def load_tables(data_dir):
    """Load all 5 tables and convert datetime columns to real dates."""
    tables = {}
    for name, file in FILES.items():
        df = pd.read_csv(Path(data_dir) / file)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        tables[name] = df
    return tables
