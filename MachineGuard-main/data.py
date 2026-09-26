# data.py
# ---------------------------------------------------------------------------
# Load the AI4I 2020 CSV and turn it into the table the models train on.
#
# Preprocessing is deliberately EXPLICIT and small - every step is listed
# here and recorded in the "report" that preprocess() returns, so the
# dashboard can show exactly what was done to the raw data.
# ---------------------------------------------------------------------------

import math

import pandas as pd

import config


def load_raw():
    """
    Read the CSV exactly as it sits on disk.

    encoding="utf-8-sig" strips the byte-order mark that some exports put at
    the very start of the file, which would otherwise corrupt the first
    column name ("UDI" -> a garbled string).
    """
    return pd.read_csv(config.DATASET_PATH, encoding="utf-8-sig")


def _add_physics_features(df):
    """
    Add the four engineered features in config.DERIVED_FEATURE_COLUMNS.

    Each is a plain arithmetic combination of the 5 raw sensors - the very
    quantities the AI4I failure criteria are written in terms of (see
    config.py / diagnosis.py). Nothing here touches a target column, so
    there is no label leakage. Returns a NEW frame; the input is untouched.
    """
    df = df.copy()
    torque = df["Torque [Nm]"]
    rpm = df["Rotational speed [rpm]"]
    wear = df["Tool wear [min]"]

    df["Mechanical power [W]"] = torque * rpm * 2.0 * math.pi / 60.0
    df["Temp difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    df["Overstrain [min Nm]"] = wear * torque
    df["Wear-speed [min rpm]"] = wear * rpm
    return df


def add_physics_features(df):
    """Public wrapper - build the engineered features for a frame of raw
    sensor rows (e.g. when scoring readings that did not come through
    preprocess())."""
    return _add_physics_features(df)


def preprocess(raw):
    """
    Turn the raw AI4I DataFrame into the clean training table.

    Steps, in order:
      1. Drop identifier columns (UDI, Product ID) - a row number and a
         serial string carry no failure signal, and keeping them would let
         the model memorise individual rows.
      2. Rename the binary target  "Target" -> "Failure"  (clearer name;
         "Failure Type" is left untouched for the diagnosis engine).
      3. Data-quality checks - count missing values and duplicate rows
         (AI4I has none of either; we check rather than assume).
      4. Add engineered physics features (mechanical power, the temp gap,
         wear x torque, wear x speed). These are exact functions of the 5
         raw sensors - the same quantities the documented failure modes are
         defined on - so they leak nothing; they just save the model from
         re-deriving them. This is where most of the model's accuracy comes
         from.
      5. NO feature scaling. A Random Forest splits on thresholds
         ("torque > 45?"), so multiplying a column by a constant changes
         nothing. StandardScaler / MinMax would add complexity for zero gain.
      6. Class-imbalance is NOT fixed here (no over/under-sampling). It is
         handled inside the model with class_weight="balanced".

    Returns (clean_df, report) where report is a dict describing every step,
    for display in the dashboard.
    """
    report = {"raw_shape": tuple(raw.shape)}

    # 1. drop identifiers
    clean = raw.drop(columns=config.ID_COLUMNS)
    report["dropped_columns"] = list(config.ID_COLUMNS)

    # 2. rename target
    clean = clean.rename(
        columns={config.RAW_TARGET_COLUMN: config.TARGET_COLUMN}
    )
    report["renamed_column"] = {
        config.RAW_TARGET_COLUMN: config.TARGET_COLUMN
    }

    # 3. data-quality checks (on the raw frame)
    report["missing_values"] = int(raw.isna().sum().sum())
    report["duplicate_rows"] = int(raw.duplicated().sum())

    # 4. engineered physics features (deterministic - no leakage)
    clean = _add_physics_features(clean)
    report["engineered_features"] = {
        "Mechanical power [W]": "Torque [Nm] x Rotational speed [rpm] x 2*pi/60",
        "Temp difference [K]":  "Process temperature [K] - Air temperature [K]",
        "Overstrain [min Nm]":  "Tool wear [min] x Torque [Nm]",
        "Wear-speed [min rpm]": "Tool wear [min] x Rotational speed [rpm]",
    }

    # 4 / 5. record the choices we did NOT make
    report["feature_scaling"] = (
        "none - Random Forest is threshold-based, so feature scale is irrelevant"
    )
    report["imbalance_handling"] = (
        "class_weight='balanced' inside the model - no resampling of rows"
    )

    # what actually goes into the model, and what is left out
    report["features_used"] = list(config.MODEL_FEATURE_COLUMNS)
    report["feature_excluded"] = {
        config.TYPE_COLUMN: (
            "categorical product grade; tested as a one-hot feature, "
            "the model rated it <1% important and recall dropped, so it is "
            "used by the diagnosis rules only"
        )
    }
    report["target"] = config.TARGET_COLUMN
    report["second_target_not_a_feature"] = config.FAILURE_TYPE_COLUMN

    # class balance of the clean set
    failures = int(clean[config.TARGET_COLUMN].sum())
    total = int(len(clean))
    report["class_balance"] = {
        "rows": total,
        "failures": failures,
        "non_failures": total - failures,
        "failure_rate": failures / total,
    }
    report["clean_shape"] = tuple(clean.shape)

    return clean, report


def load_dataset_with_report():
    """Raw load + preprocess. Returns (clean_df, report)."""
    return preprocess(load_raw())


def load_dataset():
    """Raw load + preprocess. Returns just the clean DataFrame."""
    clean, _report = load_dataset_with_report()
    return clean
