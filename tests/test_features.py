import pandas as pd

from src.features import (
    make_skeleton, add_label, add_telemetry_features, add_error_features,
    add_maintenance_features, drop_incomplete_history, ERRORS, COMPS,
)

T = pd.Timestamp("2015-03-10 06:00")        # a prediction moment


def one_row(machine=1, when=T):
    """A one-row skeleton for a single machine at prediction time `when`."""
    return pd.DataFrame({"machineID": [machine], "prediction_time": [when]})


def failures_at(*times, machine=1):
    return pd.DataFrame({"datetime": pd.to_datetime(list(times)),
                         "machineID": machine, "failure": "comp1"})


def telemetry(values_by_time, machine=1):
    """Hourly telemetry; every sensor gets the value given for that hour."""
    rows = [{"datetime": t, "machineID": machine, "volt": v, "rotate": v,
             "pressure": v, "vibration": v} for t, v in values_by_time.items()]
    return pd.DataFrame(rows)


# ---- skeleton and label -------------------------------------------------

def test_skeleton_has_one_row_per_machine_per_day_at_0600():
    sk = make_skeleton([1, 2], "2015-03-01", "2015-03-04")
    assert len(sk) == 2 * 4
    assert (sk["prediction_time"].dt.hour == 6).all()


def label_of(failure_times, when):
    df = add_label(one_row(when=pd.Timestamp(when)), failures_at(*failure_times))
    return int(df["failure_next_24h"].iloc[0])


def test_failure_at_exactly_0600_belongs_to_previous_day():
    assert label_of(["2015-03-10 06:00"], "2015-03-09 06:00") == 1
    assert label_of(["2015-03-10 06:00"], "2015-03-10 06:00") == 0


def test_failure_just_after_0600_belongs_to_same_day():
    assert label_of(["2015-03-10 07:00"], "2015-03-10 06:00") == 1


def test_failure_more_than_24h_ahead_is_not_counted():
    assert label_of(["2015-03-11 06:01"], "2015-03-10 06:00") == 0


def test_simultaneous_failures_make_one_positive_row():
    fl = pd.DataFrame({"datetime": pd.to_datetime(["2015-03-10 09:00"] * 2),
                       "machineID": 1, "failure": ["comp1", "comp2"]})
    df = add_label(one_row(), fl)
    assert len(df) == 1 and df["failure_next_24h"].iloc[0] == 1


def test_other_machines_failures_are_ignored():
    df = add_label(one_row(machine=2), failures_at("2015-03-10 09:00", machine=1))
    assert df["failure_next_24h"].iloc[0] == 0


# ---- no future data in the features -------------------------------------

def test_telemetry_features_ignore_readings_after_T():
    hours = pd.date_range(T - pd.Timedelta(hours=30), T + pd.Timedelta(hours=5), freq="h")
    normal = {t: 10.0 for t in hours}
    spiked = {t: (10.0 if t <= T else 9999.0) for t in hours}     # huge values after T
    a = add_telemetry_features(one_row(), telemetry(normal))
    b = add_telemetry_features(one_row(), telemetry(spiked))
    assert a["volt_mean_24h"].iloc[0] == b["volt_mean_24h"].iloc[0] == 10.0


def test_telemetry_window_is_the_24_hours_up_to_and_including_T():
    hours = pd.date_range(T - pd.Timedelta(hours=30), T, freq="h")
    values = {t: (1000.0 if t <= T - pd.Timedelta(hours=24) else 10.0) for t in hours}
    out = add_telemetry_features(one_row(), telemetry(values))
    assert out["volt_mean_24h"].iloc[0] == 10.0       # the reading at T-24h is outside the window


def err_features(error_times):
    errs = pd.DataFrame({"datetime": pd.to_datetime(error_times),
                         "machineID": 1, "errorID": "error1"})
    tel = telemetry({t: 1.0 for t in pd.date_range(T - pd.Timedelta(hours=48),
                                                    T + pd.Timedelta(hours=48), freq="h")})
    return add_error_features(one_row(), errs, tel).iloc[0]


def test_error_at_T_counts_but_error_after_T_does_not():
    row = err_features(["2015-03-10 06:00", "2015-03-10 07:00"])
    assert row["error1_count_24h"] == 1 and row["any_error_24h"] == 1


def test_error_exactly_24h_before_T_is_outside_the_window():
    row = err_features(["2015-03-09 06:00"])
    assert row["error1_count_24h"] == 0 and row["any_error_24h"] == 0


# ---- maintenance ---------------------------------------------------------

def maint_for(*events):
    """events: (time, comp) pairs for machine 1; every comp also gets an old replacement."""
    rows = [{"datetime": pd.Timestamp("2014-06-01 06:00"), "machineID": 1, "comp": c} for c in COMPS]
    rows += [{"datetime": pd.Timestamp(t), "machineID": 1, "comp": c} for t, c in events]
    return pd.DataFrame(rows)


def test_days_since_uses_latest_replacement_up_to_T():
    out = add_maintenance_features(one_row(), maint_for(("2015-03-05 06:00", "comp1")))
    assert out["days_since_comp1"].iloc[0] == 5.0


def test_replacement_after_T_is_not_used():
    out = add_maintenance_features(one_row(), maint_for(("2015-03-10 07:00", "comp1")))
    assert out["days_since_comp1"].iloc[0] > 200      # still the 2014 replacement


def test_replacement_exactly_at_T_counts():
    out = add_maintenance_features(one_row(), maint_for(("2015-03-10 06:00", "comp1")))
    assert out["days_since_comp1"].iloc[0] == 0.0


# ---- first day -----------------------------------------------------------

def test_first_day_is_dropped():
    sk = make_skeleton([1], "2015-01-01", "2015-01-03")
    assert list(drop_incomplete_history(sk)["prediction_time"].dt.day) == [2, 3]