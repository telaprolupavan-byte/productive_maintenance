# Project Brief: Predictive Maintenance

## 1 · Business Problem

The goal is to reduce unexpected machine failures by identifying machines that are
likely to fail soon enough for the maintenance team to take preventive action.

The dataset contains 100 machines, with a median machine age of 12 years, and 761
recorded component failure events during 2015. An unexpected failure interrupts
machine operation and requires an unplanned repair or component replacement, which
costs more than a planned replacement done before the failure.

The business question is:

> Can we use historical machine telemetry, error events, maintenance history, and
> machine information to identify machines at higher risk of failure before the
> failure occurs?

## 2 · User and Action

The primary user is the maintenance team. The model's prediction helps the team
decide which machines need attention before a failure occurs:

1. The system generates a failure-risk prediction for each machine.
2. The maintenance team reviews machines with elevated risk.
3. The team investigates the machine and its recent behaviour.
4. If appropriate, they inspect or replace a component before an unexpected failure.

The model supports maintenance prioritisation. It does not automatically decide
that a component must be replaced.

## 3 · Prediction Moment

The model runs once a day at 06:00, aligned with the daily maintenance cycle.

The data supports this choice: most failure records and maintenance records are
logged at 06:00.

At each prediction point, the system asks:

> Based only on information available up to this point, is this machine likely to
> fail in the next 24 hours?

**Exact time windows** (prediction moment T = each day at 06:00):

- Features use data with timestamps up to and including T.
- The label is 1 if a failure is logged after T and up to T + 24 hours.
- A failure logged at exactly 06:00 on day D is therefore the label of the row at
  06:00 on day D−1, and a known past event for the row at 06:00 on day D.
- A row's features and its own label never overlap.

## 4 · Prediction Horizon

The model predicts:

> Will this machine experience a failure during the next 24 hours?

A 24-hour horizon gives the maintenance team time to investigate and schedule a
part replacement.

**Evidence from EDA:** each component has a warning sensor that shifts about
48 hours before a failure and stays shifted until it happens:

| Component | Warning sensor | Shift in the 24h before failure |
|---|---|---|
| comp1 | Voltage | +1.30 std (higher) |
| comp2 | Rotation | −1.48 std (lower) |
| comp3 | Pressure | +2.23 std (higher) |
| comp4 | Vibration | +1.86 std (higher) |

The shift is equally strong 24–48 hours before a failure and absent more than
48 hours before. With a 24-hour look-back window, a 24-hour horizon sees the full
warning signal for every failure. A horizon beyond 48 hours would see none of it,
so 24 hours is supported by the data.

## 5 · Target

The first target is **any failure: yes/no**.

- 1 → the machine has a failure within the next 24 hours
- 0 → the machine does not have a failure within the next 24 hours

This makes the first version a binary classification problem. Reasons:

- Several components can fail on the same machine at the same timestamp.
- Some components have relatively few failures (comp3: 131).

EDA shows each component has its own warning sensor, so predicting *which*
component will fail (multi-class) is a realistic extension for a later version.

The failures table provides the events used to build the target. The maintenance
table is not used as the target, because maintenance records include preventive
replacements as well as replacements after failures.

## 6 · Grain

**One row = one machine on one day** (at the 06:00 prediction moment).

With 100 machines and about 365 days of 2015 data, this gives roughly
100 × 365 ≈ 36,500 machine-day rows.

Example:

| date | machineID | features… | failure_next_24h |
|---|---|---|---|
| 2015-03-01 | 1 | telemetry / error features | 0 |
| 2015-03-01 | 2 | telemetry / error features | 1 |
| 2015-03-01 | 3 | telemetry / error features | 0 |

**Expected positive rate:** about 2% of machine-days (761 failures, fewer positive
rows after merging failures logged at the same time on the same machine).

This grain makes it straightforward to combine telemetry, errors, maintenance
history, machine attributes, and failure labels.

## 7 · Allowed Inputs

The model may only use information available at the prediction moment.

Potential inputs:

- Recent telemetry: voltage, rotation, pressure, vibration (rolling windows, for
  example 3 hours and 24 hours)
- Historical error events (for example, counts per error type in the last 24 hours)
- Historical maintenance events (for example, days since each component was last
  replaced)
- Machine age and machine model
- Historical failure-related features, built only from past failures

**Key rule:** no information from after the prediction moment can be used. For
example, if the prediction is made on March 10 at 06:00, nothing logged after that
moment may be used.

Maintenance data from 2014 can be used, because it occurred before the 2015
prediction period, but maintenance features must include only events up to the
prediction moment. This matters most for avoiding temporal leakage.

## 8 · Metric

**Cost of errors:** a missed failure means unplanned downtime and an emergency
repair. A false alarm means one unnecessary inspection. A missed failure costs
more, so the model should favour catching failures, but false alarms must stay
limited, or the team will stop trusting the alerts.

**Primary metric:** recall at a fixed precision floor: the share of real failures
caught while, for example, at least half of all alerts are real failures. The
exact floor will be set with the maintenance team's inspection capacity in mind.

**Secondary metric:** PR-AUC, to compare models regardless of threshold.

**Not used:** accuracy. With about 2% of machine-days containing a failure,
predicting "no failure" every time would be about 98% accurate and useless.

## 9 · Baselines

Before evaluating ML models, two simple non-ML rules set the bar:

1. **Error rule:** alert if the machine had any error in the previous 24 hours.
2. **Sensor rule (from EDA):** alert if any warning sensor's 24-hour average is
   more than 1 standard deviation from normal in its warning direction.

The baselines answer one question:

> Does the ML model add useful predictive value beyond a simple operational rule?

If the ML model cannot clearly beat the sensor rule, that is important information
about the value of the approach.

## 10 · Train / Validation / Test Split

The data is split by time, not randomly:

| Part | Months (2015) | Use |
|---|---|---|
| Train | January–August | Fit models |
| Validation | September–October | Compare models, tune settings, choose threshold |
| Test | November–December | Final evaluation, used once |

A random split would put later observations into training while earlier ones are
evaluated, letting the model learn from the future. The production scenario is:

> Past data → train model → predict future data

so evaluation follows the same direction in time.

## 11 · Architecture

The first version is a daily batch prediction pipeline:

    Raw tables (telemetry, errors, maintenance, failures, machines)
        ↓
    Data loading and validation        (src/data.py)
        ↓
    Feature engineering                (src/features.py)
        ↓
    Machine-day dataset                (data/processed/)
        ↓
    Trained model                      (src/train.py)
        ↓
    Daily failure-risk predictions
        ↓
    Maintenance team

A daily batch job fits the problem, because predictions are made once per machine
per day.

Later in the project, an API-based inference service (FastAPI in Docker) will
expose the trained model for on-demand predictions, showing the step from model
development to model serving.

## 12 · Risks and Limitations

1. **Temporal data leakage.** The biggest modelling risk is using information that
   would not have been available at prediction time. All features are built from
   historical data only, using the exact time windows in section 3.
2. **Imbalanced target.** About 2% of machine-days are positive, so the evaluation
   uses precision-bounded recall and PR-AUC, and the threshold must be tuned.
3. **Few failures per component.** comp1: 192 · comp2: 259 · comp3: 131 ·
   comp4: 179. A component-specific model may not have enough examples, which is
   another reason to start with the any-failure target.
4. **Limited time period.** One year of data may not capture every seasonal or
   operational pattern.
5. **Maintenance is not the same as failure.** Maintenance includes preventive
   replacements, so it cannot be used as the failure label.
6. **Logged time vs actual failure time.** Timestamps show when an event was
   recorded, not necessarily the exact moment the component failed.
7. **Simulated data.** The dataset was generated for a Microsoft tutorial, so its
   patterns are cleaner than a real factory's. Real sensor data would show weaker,
   noisier warning signs.
8. **Incomplete early history.** Telemetry starts on 2015-01-01, so rows in the
   first days of January have less than a full 24 hours of history.
9. **Simultaneous failures.** Several components can fail on the same machine at the
   same time. In the binary target they count as one positive row.
10. **Decision support only.** A high-risk prediction does not mean a component will
    certainly fail. It helps the team prioritise inspection, not automatically
    trigger a replacement.

## Decision Summary

| Decision | Initial choice |
|---|---|
| Problem | Predict machine failure before it happens |
| User | Maintenance team |
| Prediction moment | Daily at 06:00 |
| Horizon | Next 24 hours |
| Horizon evidence | Warning sensors shift about 48h before failure |
| Target | Any failure: yes/no |
| Grain | Machine-day (~36,500 rows, ~2% positive) |
| Inputs | Historical telemetry, errors, maintenance, machine information |
| Primary metric | Recall at a precision floor |
| Secondary metric | PR-AUC |
| Baselines | Error in previous 24h; warning-sensor rule |
| Split | Time-based: Jan–Aug / Sep–Oct / Nov–Dec |
| Architecture | Batch ML pipeline → daily predictions → maintenance team |
| Later serving | API inference (FastAPI, Docker) |
| Main risk | Temporal leakage |