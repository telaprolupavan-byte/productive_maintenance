# Project Brief

## 1. Business problem
*Who has the problem, and what does it cost them today?*


## 2. User and action
*Who receives the prediction, and what do they do with it?*


## 3. Prediction moment
*When does the model run?*


## 4. Horizon
*"Will the machine fail within the next ___ hours?" Why that number?*


## 5. Target (y)
*Any failure or which component? Binary or multi-class? Why?*
1 · Business Problem

The goal is to reduce unexpected machine failures by identifying machines that are likely to fail soon enough for the maintenance team to take preventive action.

The dataset contains 100 machines, with a median machine age of 12 years, and 761 recorded component failure events during 2015. An unexpected failure can interrupt machine operation and require an unplanned repair or component replacement.

The business problem is therefore:

Can we use historical machine telemetry, error events, maintenance history, and machine information to identify machines at higher risk of failure before the failure occurs?

2 · User and Action

The primary user is the maintenance team.

The model's prediction should help the team decide which machines need attention before a failure occurs.

For example:

The system generates a failure-risk prediction for a machine.
The maintenance team reviews machines with elevated risk.
The team investigates the machine and its recent behavior.
If appropriate, they can inspect or replace a component before an unexpected failure occurs.

The model is therefore intended to support maintenance prioritization, rather than automatically deciding that a component must be replaced.

3 · Prediction Moment

The initial design will use a daily prediction schedule, with the prediction generated around the daily maintenance cycle.

The dataset shows that most failure records are logged at 06:00, and maintenance records are also consistently logged at 06:00. This makes a daily prediction point practical for the first version of the project.

At each prediction point, the system asks:

Based only on information available up to this point, is this machine likely to experience a failure soon?

Anything that happens after the prediction moment cannot be used as a feature.

4 · Prediction Horizon

For the first version, I would use a 24-hour prediction horizon.

The model will predict:

Will this machine experience a failure during the next 24 hours?

A 24-hour horizon gives the maintenance team some time to investigate a machine and potentially take preventive action.

A shorter horizon could provide a more immediate prediction but less time to act. A much longer horizon would provide more planning time but could introduce more uncertainty into the prediction.

The horizon can be changed later if EDA and model performance show that another window is more appropriate.

5 · Target

The initial target will be any failure: yes/no.

The reason is that the failure data contains multiple component types, and the data shows that multiple components can be recorded for the same machine at the same timestamp.

Therefore, the first model will answer:

1 → machine has a failure within the next 24 hours
0 → machine does not have a failure within the next 24 hours

The component-specific failure type can be investigated later as a separate modeling problem.

This makes the first version a binary classification problem.

The failures table provides the failure events used to construct the target. The maint table should not be used as the target because maintenance records represent replacement/maintenance actions and can include preventive maintenance.

6 · Grain

The modeling dataset will use:

One row = one machine on one day

With 100 machines and approximately 365 days of 2015 data, this gives roughly:

100 × 365 ≈ 36,500 machine-day observations

Each row represents the information available for one machine at one prediction point.

For example:

date	machineID	features...	failure_next_24h
2015-03-01	1	telemetry/error features	0
2015-03-01	2	telemetry/error features	1
2015-03-01	3	telemetry/error features	0

This daily grain also makes it easier to combine telemetry, errors, maintenance history, machine attributes, and failure labels.

7 · Allowed Inputs

The model can use information that was available before the prediction moment.

Potential inputs include:

Recent telemetry measurements
voltage
rotation
pressure
vibration
Historical error events
Historical maintenance/replacement events
Machine age
Machine model
Historical failure-related features, where appropriate

The key rule is:

No information from after the prediction moment can be used to create features.

For example, if the prediction is made on March 10, information from March 11 cannot be used.

The maintenance data beginning in 2014 can be used because it occurred before the 2015 prediction period. However, maintenance information must be constructed carefully so that only historical maintenance events are included.

This is especially important for avoiding temporal data leakage.

8 · Metric

The primary evaluation metric will be recall, with PR-AUC used as an additional metric.

Failures are relatively rare compared with normal machine-days. Because of this, accuracy could be misleading.

For example, if almost every machine-day does not contain a failure, a model could achieve high accuracy simply by predicting:

No failure

for almost everything.

That would not be useful to the maintenance team.

Recall measures how many of the actual failures the model successfully identifies.

A missed failure is particularly important because the purpose of the system is to give the maintenance team an opportunity to act before the failure.

PR-AUC will provide another view of performance on the imbalanced positive class.

9 · Baseline

Before evaluating ML models, I will create a simple non-ML baseline.

One initial rule will be:

Alert if the machine experienced an error during the previous 24 hours.

This provides a simple benchmark against which the ML model can be compared.

The purpose of the baseline is not to maximize performance. It answers:

Does the ML model provide useful predictive value beyond a simple operational rule?

If the ML model cannot improve meaningfully over the baseline, that is important information about the value of the approach.

10 · Train/Test Split

The data will be split chronologically rather than randomly.

The dataset covers approximately one year of operational data in 2015, so the initial approach will train on earlier periods and evaluate on later periods.

For example:

Earlier months → Training
Later months   → Validation/Test

The exact month boundaries will be finalized after the EDA and feature-building stages.

A random train/test split would allow observations from later periods to enter the training set while earlier observations are being evaluated. That would not represent the way the model would actually be used in production.

The production scenario is:

Past data → train model → predict future data

Therefore, the evaluation should follow the same temporal direction.

11 · Architecture

The initial architecture will be a batch prediction pipeline.

Conceptually:

Raw Tables
    ↓
Data Loading & Validation
    ↓
Feature Engineering
    ↓
Daily Machine-Level Dataset
    ↓
ML Model
    ↓
Failure Risk Prediction
    ↓
Maintenance Team

The raw data consists of:

Telemetry
Errors
Maintenance
Failures
Machines

These tables will be cleaned/validated and transformed into machine-day features.

The trained model will then generate a failure-risk prediction for each machine.

For the first version, a daily batch job is appropriate because the prediction problem is based on daily machine-level observations.

Later in the project, an API-based inference service can expose the trained model for on-demand predictions. That will allow the project to demonstrate the transition from model development to model serving.

12 · Risks and Limitations
1. Temporal data leakage

The biggest modeling risk is accidentally using information that would not have been available when the prediction was made.

Features must therefore be generated using only historical data.

2. Imbalanced target

There are 761 recorded failure events, compared with a much larger number of normal observations.

This means the positive class is relatively rare and requires appropriate evaluation metrics and potentially threshold tuning.

3. Few failures per component

The failure events are distributed across four components:

comp1 → 192
comp2 → 259
comp3 → 131
comp4 → 179

Some components therefore have relatively fewer failure examples. A component-specific model may not have enough observations to learn reliable patterns.

This is another reason to start with the broader any-failure target.

4. Limited time period

The primary operational dataset covers approximately one year of 2015 data.

A single year may not capture every seasonal or operational pattern that could occur over a longer period.

5. Maintenance is not the same as failure

Maintenance records cannot simply be treated as failure labels because maintenance can represent preventive replacement as well as replacement associated with failures.

6. Logged time versus actual failure time

The timestamp represents when the event was recorded/logged in the dataset. It should not automatically be interpreted as the exact physical moment when the component actually failed.

7. Model predictions are decision support

A high-risk prediction does not necessarily mean that a component will fail.

The prediction should help the maintenance team prioritize inspection and intervention rather than automatically trigger a replacement.

Project decision summary
Decision	Initial choice
Problem	Predict machine failure before it happens
User	Maintenance team
Prediction	Daily
Horizon	Next 24 hours
Target	Any failure: yes/no
Grain	Machine-day
Inputs	Historical telemetry, errors, maintenance, machine information
Primary metric	Recall
Secondary metric	PR-AUC
Baseline	Error in previous 24 hours
Split	Time-based
Architecture	Batch ML pipeline → predictions → maintenance team
Later serving	API inference
Main risk	Temporal leakage

## 6. Grain
*One row = one ___ at one ___. Roughly how many rows?*


## 7. Allowed inputs
*Which tables and time windows can features use? What is NOT allowed?*


## 8. Success metric
*Primary metric and why? Cost of a missed failure vs a false alarm?*


## 9. Baseline to beat
*The simplest rule anyone could use without ML.*


## 10. Split strategy
*How will you split train/val/test? Why not a random split?*


## 11. Architecture
*raw tables → ... → how the prediction reaches the user*


## 12. Risks and open questions
*What could go wrong, and what don't you know yet?*

