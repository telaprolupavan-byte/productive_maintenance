"""Feature engineering: build the machine-day dataset.

One row = one machine on one day, stamped at the 06:00 prediction moment (T).
Every feature uses data logged at or before T; the label looks 24h ahead.
"""
import pandas as pd

from src.data import load_tables

PREDICTION_HOUR = 6                      # the model runs every day at 06:00
HORIZON = pd.Timedelta(hours=24)         # label window: (T, T + 24h]
LOOKBACK = "24h"                         # feature window: (T - 24h, T]

SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERRORS = ["error1", "error2", "error3", "error4", "error5"]
COMPS = ["comp1", "comp2", "comp3", "comp4"]

FIRST_DAY = "2015-01-01"
LAST_DAY = "2015-12-31"
FIRST_FULL_DAY = "2015-01-02"            # telemetry starts 2015-01-01 06:00, so day 1 has < 24h of history


def make_skeleton(machine_ids, first_day=FIRST_DAY, last_day=LAST_DAY):
    """One row per machine per day, stamped at 06:00 (prediction_time = T)."""
    days = pd.date_range(first_day, last_day, freq="D") + pd.Timedelta(hours=PREDICTION_HOUR)
    index = pd.MultiIndex.from_product([machine_ids, days], names=["machineID", "prediction_time"])
    return index.to_frame(index=False)


def add_label(df, failures):
    """Add failure_next_24h: 1 if the machine has a failure in (T, T + 24h].

    The window is open at the start, so a failure logged exactly at 06:00 on
    day D belongs to the row of day D-1, not day D.
    """
    # Failures of different components at the same time count once (yes/no target)
    events = failures[["machineID", "datetime"]].drop_duplicates()

    pairs = df.merge(events, on="machineID", how="left")
    in_window = (
        (pairs["datetime"] > pairs["prediction_time"])
        & (pairs["datetime"] <= pairs["prediction_time"] + HORIZON)
    )
    positive = pairs.loc[in_window, ["machineID", "prediction_time"]].drop_duplicates()
    positive["failure_next_24h"] = 1

    out = df.merge(positive, on=["machineID", "prediction_time"], how="left")
    out["failure_next_24h"] = out["failure_next_24h"].fillna(0).astype(int)
    return out


def add_telemetry_features(df, telemetry):
    """24h rolling mean and std of each sensor, as of 06:00 (never past T)."""
    tel = telemetry.sort_values(["machineID", "datetime"])

    rolled = (
        tel.set_index("datetime")
           .groupby("machineID")[SENSORS]
           .rolling(LOOKBACK)                  # window = (t - 24h, t]
           .agg(["mean", "std"])
    )
    rolled.columns = [f"{sensor}_{stat}_24h" for sensor, stat in rolled.columns]
    rolled = rolled.reset_index()

    return df.merge(
        rolled,
        left_on=["machineID", "prediction_time"],
        right_on=["machineID", "datetime"],
        how="left",
    ).drop(columns="datetime")


def add_error_features(df, errors, telemetry):
    """Count of each error type in the last 24h, plus any_error_24h."""
    counts = (
        errors.groupby(["machineID", "datetime", "errorID"]).size()
              .unstack(fill_value=0)
              .reindex(columns=ERRORS, fill_value=0)
    )

    # Put the counts on the hourly telemetry grid (0 where nothing happened),
    # so the rolling window is always defined at 06:00
    grid = pd.MultiIndex.from_frame(telemetry[["machineID", "datetime"]].sort_values(["machineID", "datetime"]))
    counts_hourly = counts.reindex(grid, fill_value=0)

    rolled = (
        counts_hourly.reset_index()
                     .set_index("datetime")
                     .groupby("machineID")[ERRORS]
                     .rolling(LOOKBACK).sum()
    )
    count_cols = [f"{e}_count_24h" for e in ERRORS]
    rolled.columns = count_cols
    rolled = rolled.reset_index()

    out = df.drop(columns=count_cols + ["any_error_24h"], errors="ignore")   # safe to re-run
    out = out.merge(
        rolled,
        left_on=["machineID", "prediction_time"],
        right_on=["machineID", "datetime"],
        how="left",
    ).drop(columns="datetime")
    out["any_error_24h"] = (out[count_cols].sum(axis=1) > 0).astype(int)
    return out


def add_maintenance_features(df, maint):
    """Days since each component was last replaced, using events up to T only."""
    out = df.drop(columns=[f"days_since_{c}" for c in COMPS], errors="ignore")   # safe to re-run
    out = out.sort_values("prediction_time")             # merge_asof needs sorted keys

    for comp in COMPS:
        last = (
            maint[maint["comp"] == comp][["machineID", "datetime"]]
            .drop_duplicates()
            .rename(columns={"datetime": "last_replaced"})
            .sort_values("last_replaced")
        )
        out = pd.merge_asof(
            out, last,
            left_on="prediction_time", right_on="last_replaced",
            by="machineID", direction="backward",         # latest replacement at or before T
        )
        out[f"days_since_{comp}"] = (out["prediction_time"] - out["last_replaced"]).dt.total_seconds() / 86400
        out = out.drop(columns="last_replaced")

    return out.sort_values(["machineID", "prediction_time"]).reset_index(drop=True)


def add_machine_features(df, machines):
    """Machine age, and the machine model as one-hot columns."""
    out = df.drop(columns=["age", "model"] + [c for c in df.columns if c.startswith("model_")],
                  errors="ignore")                        # safe to re-run
    out = out.merge(machines, on="machineID", how="left")
    return pd.get_dummies(out, columns=["model"], dtype=int)


def drop_incomplete_history(df, first_full_day=FIRST_FULL_DAY):
    """Drop rows with less than 24h of telemetry history (the first day)."""
    return df[df["prediction_time"] >= first_full_day].reset_index(drop=True)


def build_dataset(data_dir):
    """Raw tables -> machine-day dataset (features + failure_next_24h)."""
    t = load_tables(data_dir)
    df = make_skeleton(t["machines"]["machineID"])
    df = add_label(df, t["failures"])
    df = add_telemetry_features(df, t["telemetry"])
    df = add_error_features(df, t["errors"], t["telemetry"])
    df = add_maintenance_features(df, t["maint"])
    df = add_machine_features(df, t["machines"])
    return drop_incomplete_history(df)
