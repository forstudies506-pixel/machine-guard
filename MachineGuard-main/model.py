import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import config


def features_from_row(latest):
    """
    Build a one-row feature DataFrame from a single reading (a pandas
    Series, e.g. test_df.iloc[i]), with the exact column names and order
    the model was trained on - so predictions line up and scikit-learn
    does not warn.

    Rows that came through data.preprocess() already carry the engineered
    physics columns. If a caller passes a bare 5-sensor row, we derive them
    here so scoring still works.
    """
    latest = _with_derived(latest)
    return pd.DataFrame(
        [[latest[column] for column in config.MODEL_FEATURE_COLUMNS]],
        columns=config.MODEL_FEATURE_COLUMNS,
    )


def _with_derived(latest):
    """Return the reading with the 4 engineered features present, computing
    any that are missing from the 5 raw sensors."""
    if all(col in latest for col in config.DERIVED_FEATURE_COLUMNS):
        return latest
    latest = latest.copy()
    torque = latest["Torque [Nm]"]
    rpm = latest["Rotational speed [rpm]"]
    wear = latest["Tool wear [min]"]
    latest["Mechanical power [W]"] = torque * rpm * 2.0 * math.pi / 60.0
    latest["Temp difference [K]"] = (
        latest["Process temperature [K]"] - latest["Air temperature [K]"]
    )
    latest["Overstrain [min Nm]"] = wear * torque
    latest["Wear-speed [min rpm]"] = wear * rpm
    return latest


def _score(y_true, y_pred):
    """
    Bundle the four classification metrics + the confusion matrix for one
    set of predictions.

    Because only ~3.4% of rows are failures, ACCURACY ALONE IS MISLEADING -
    a model that never predicts failure still scores ~97%. So we also keep:
      precision  of the rows we CALLED "failure", how many really were
      recall     of the rows that really WERE failures, how many we caught
      f1         harmonic mean of precision and recall
      confusion  [[TN, FP], [FN, TP]]
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
        "n_failures": int(sum(y_true)),
    }


def train_model(df):
    """
    Train the failure-prediction model (a Random Forest classifier) on the
    AI4I 2020 dataset using a 60 / 20 / 20 train / validation / test split.

    Returns (model, metrics, test_df):
      model    the fitted Random Forest
      metrics  {
                 "validation": _score(...) on the 20% validation set,
                 "test":       _score(...) on the 20% test set,
                 "split":      {"train": n, "validation": n, "test": n},
               }
      test_df  the 20% of rows the model NEVER saw and that were NEVER used
               while building it. The dashboard inspects only these rows, so
               the "failure risk" it shows is a true out-of-sample number.

    Why a validation set as well as a test set: the validation set is what
    you look at WHILE developing (comparing settings, picking a threshold).
    The test set stays sealed until the very end, so its score is an honest
    estimate of performance on unseen data. Reporting both shows the model
    is not over-fitted - the two scores should be close.
    """

    # --- Step 1: carve off the FINAL test set (20% of everything) --------
    # Splitting the whole table (not just X/y) lets us hand the untouched
    # test rows back to the dashboard. stratify keeps the ~3.4% failure
    # rate in every part.
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df[config.TARGET_COLUMN],
    )

    # --- Step 2: split the remaining 80% into train (60%) + val (20%) ----
    # val is VAL_SIZE / (1 - TEST_SIZE) of what is left = 0.2 / 0.8 = 0.25.
    val_fraction = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=config.RANDOM_SEED,
        stratify=train_val_df[config.TARGET_COLUMN],
    )

    X_train = train_df[config.MODEL_FEATURE_COLUMNS]
    y_train = train_df[config.TARGET_COLUMN]
    X_val = val_df[config.MODEL_FEATURE_COLUMNS]
    y_val = val_df[config.TARGET_COLUMN]
    X_test = test_df[config.MODEL_FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COLUMN]

    # class_weight="balanced" tells the forest the rare "failure" rows
    # matter as much as the common ones. Without it the model quietly
    # under-predicts failure to chase accuracy, and misses about a third
    # of real failures. With it we catch more, at the cost of a few more
    # false alarms - the right trade for maintenance.
    #
    # min_samples_leaf=2 stops the forest carving leaves around single rows,
    # which trims a little over-fitting (validation and test scores end up
    # closer together). The bigger lever, though, is the engineered physics
    # features added in data.py - they are what take test F1 from ~0.64 to
    # ~0.81.
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    metrics = {
        "validation": _score(y_val, model.predict(X_val)),
        "test": _score(y_test, model.predict(X_test)),
        "split": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }

    # reset_index so the dashboard can use simple 0..N-1 positions.
    return model, metrics, test_df.reset_index(drop=True)


def predict_machine_risk(model, latest):
    """
    Given the latest sensor reading, return the model's estimated
    probability of failure as a number between 0.0 and 1.0.
    """

    # Build a one-row DataFrame with the SAME columns (5 raw sensors + the
    # 4 engineered physics features) and order the model trained on, so the
    # features line up and scikit-learn does not warn.
    machine_input = features_from_row(latest)

    # predict_proba returns one probability per class the model knows about.
    # We want the probability of class "1" (failure). If the model somehow
    # only ever saw the "normal" class, there is no failure probability to
    # report, so we fall back to 0.0.
    known_classes = list(model.classes_)
    if 1 not in known_classes:
        return 0.0

    failure_column = known_classes.index(1)
    probability = model.predict_proba(machine_input)[0][failure_column]

    return float(probability)


def get_feature_importance(model):
    """
    How much each sensor contributed to the model's ability to tell
    'failure' from 'normal', measured across ALL of its training data.

    Random Forests calculate this for free while training: every time a
    sensor is used to split the data and that split cleanly separates
    failures from normals, that sensor earns 'importance'. The five
    scores are then scaled to add up to 1.0 (i.e. 100%).

    Returns a pandas Series (sensor name -> importance), largest first.

    NOTE: this is a GLOBAL summary of the whole model. It is not, on its
    own, an explanation of why one specific reading was scored risky - for
    that, see explain.py (SHAP).
    """
    importance = pd.Series(
        model.feature_importances_,
        index=config.MODEL_FEATURE_COLUMNS
    )

    return importance.sort_values(ascending=False)


def estimate_risk_contributions(model, latest):
    """
    A per-reading estimate of which sensors are pushing THIS reading's
    risk up right now.

    It is a simple heuristic, NOT a rigorous model explanation:

        contribution(sensor) = global_importance(sensor)
                               x  how far the sensor sits from its healthy
                                  mean, in standard deviations (either
                                  direction)

    ...then scaled so the numbers add up to 100%. A sensor sitting near its
    normal value contributes 0. For a true per-prediction explanation you
    would use SHAP - deferred to a later phase.

    Returns a dict:
      contributions -> pandas Series (sensor -> percent), largest first
      any_elevated  -> True if at least one sensor is unusually far from normal
    """
    global_importance = pd.Series(
        model.feature_importances_,
        index=config.MODEL_FEATURE_COLUMNS
    )

    weighted = {}
    for sensor in config.SENSOR_COLUMNS:
        baseline = config.SENSOR_BASELINES[sensor]
        noise = config.SENSOR_NOISE[sensor]

        # How far the reading sits from its healthy mean, in standard
        # deviations, in EITHER direction. On this dataset a sensor can be
        # dangerous when it is unusually high OR unusually low (a very low
        # and a very high rotational speed both cause a power failure), so
        # we measure absolute distance, not one fixed direction.
        sigmas_from_normal = abs(latest[sensor] - baseline) / noise

        # Ignore ordinary wobble within the noise band.
        if sigmas_from_normal < config.CONTRIBUTION_MIN_SIGMAS:
            sigmas_from_normal = 0.0

        weighted[sensor] = global_importance[sensor] * sigmas_from_normal

    weighted = pd.Series(weighted)
    total = weighted.sum()

    if total == 0:
        zeros = pd.Series(0.0, index=config.SENSOR_COLUMNS)
        return {"contributions": zeros, "any_elevated": False}

    contributions = (weighted / total * 100).sort_values(ascending=False)
    return {"contributions": contributions, "any_elevated": True}
