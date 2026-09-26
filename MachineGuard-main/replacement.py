# replacement.py
# ---------------------------------------------------------------------------
# The "repair vs replace" helper for the OPTIMIZE step.
#
# cost.py answers "maintain now vs run to failure". This adds the other
# money question a plant asks: is this machine worth fixing at all, or has
# it reached the point where a new one is cheaper?
#
# The rule of thumb (standard fleet management): if a single repair costs
# more than a set share of a NEW equivalent machine, replace it.
#
# We need one number the rule can't get from AI4I: the price of a new
# equivalent machine. This module LEARNS that from a second dataset -
# data.csv, ~1,700 real used-equipment listings (price + age + usage +
# opaque category / manufacturer / region codes).
#
# HONESTY:
#   * data.csv is heavy-equipment resale data in USD, NOT milling machines.
#     It shows the METHOD works; the rupee figure is a demo stand-in, same
#     footing as the other cost assumptions in config.py.
#   * category / manufacturer / region are unlabelled integer codes. We
#     pick one representative profile (config.REPLACEMENT_PROFILE) to stand
#     in for "our machine".
#   * If the model trains poorly, callers fall back to
#     config.DEFAULT_REPLACEMENT_COST and nothing breaks.
# ---------------------------------------------------------------------------

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import config


# The columns data.csv actually carries.
_ID_COLUMN = "uuid"
_TARGET = "price_usd"
_NUMERIC = ["meter_hours", "age", "is_new"]
_CATEGORICAL = ["category", "manufacturer", "region"]


def load_equipment_prices():
    """Read data.csv and clean it into model-ready shape.

    Returns (frame, report) where report records what cleaning happened,
    so the dashboard can show it like data.py's preprocessing report.
    """
    raw = pd.read_csv(config.EQUIPMENT_PRICE_PATH)
    n_raw = len(raw)

    df = raw.drop(columns=[_ID_COLUMN])

    # meter_hours has a few absurd outliers (max ~600,000 vs a 99th
    # percentile near 16,000) - almost certainly data-entry errors. Clip
    # them to the 99th percentile so a handful of rows can't dominate.
    hours_cap = float(df["meter_hours"].quantile(0.99))
    n_clipped = int((df["meter_hours"] > hours_cap).sum())
    df["meter_hours"] = df["meter_hours"].clip(upper=hours_cap)

    # age is more meaningful to a model than a raw calendar year.
    df["age"] = config.REPLACEMENT_REF_YEAR - df["year"]
    df["age"] = df["age"].clip(lower=0)

    report = {
        "rows_raw": n_raw,
        "rows_clean": len(df),
        "target": _TARGET,
        "currency_raw": "USD",
        "meter_hours_cap": round(hours_cap),
        "meter_hours_clipped_rows": n_clipped,
        "features_numeric": list(_NUMERIC),
        "features_categorical": list(_CATEGORICAL),
        "price_median_usd": round(float(df[_TARGET].median())),
    }
    return df, report


def _design_matrix(df, template_columns=None):
    """Build the one-hot feature matrix.

    template_columns: if given, reindex to exactly those columns (so a
    single prediction row lines up with what the model was trained on).
    """
    x = df[_NUMERIC].copy()
    dummies = pd.get_dummies(
        df[_CATEGORICAL].astype("category"), columns=_CATEGORICAL, prefix=_CATEGORICAL
    )
    x = pd.concat([x, dummies], axis=1)
    if template_columns is not None:
        x = x.reindex(columns=template_columns, fill_value=0)
    return x


def _metrics(y_true, y_pred):
    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": round(float(np.mean(abs_err))),
        "rmse": round(float(math.sqrt(np.mean(err ** 2)))),
        "mape": round(float(np.mean(abs_err / np.clip(y_true, 1, None)) * 100), 1),
        "r2": round(1.0 - ss_res / ss_tot, 3) if ss_tot else 0.0,
        "n": int(len(y_true)),
    }


def train_replacement_model(df=None):
    """Fit the price model on data.csv.

    Returns (model, meta) where meta holds:
      feature_columns  the exact one-hot column order the model expects
      metrics          held-out error (mae/rmse/mape/r2 in USD)
      median_price_usd fallback reference
      report           the cleaning report from load_equipment_prices()
    We regress on log(price) because prices span 10k-480k and are
    right-skewed; predictions are exp()'d back.
    """
    if df is None:
        df, report = load_equipment_prices()
    else:
        _, report = load_equipment_prices()

    x = _design_matrix(df)
    y = np.log(df[_TARGET].to_numpy(dtype=float))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=config.RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    # Score in real dollars, not log space - that is what a reader cares about.
    pred_test = np.exp(model.predict(x_test))
    true_test = np.exp(y_test)

    meta = {
        "feature_columns": list(x.columns),
        "metrics": _metrics(true_test, pred_test),
        "median_price_usd": round(float(df[_TARGET].median())),
        "report": report,
    }
    return model, meta


def estimate_replacement_cost(model, feature_columns, profile=None):
    """Predict the price of a NEW equivalent machine, in config.CURRENCY.

    profile: {"category": int, "manufacturer": int, "region": int}.
    Defaults to config.REPLACEMENT_PROFILE - the representative stand-in
    for "our milling machine". We hold the asset identity fixed and set
    the condition knobs to brand-new (age 0, 0 hours, is_new 1).
    """
    if profile is None:
        profile = config.REPLACEMENT_PROFILE

    row = pd.DataFrame([{
        "meter_hours": 0.0,
        "age": 0,
        "is_new": 1,
        "category": profile["category"],
        "manufacturer": profile["manufacturer"],
        "region": profile["region"],
    }])
    x = _design_matrix(row, template_columns=feature_columns)
    price_usd = float(np.exp(model.predict(x)[0]))
    return round(price_usd * config.USD_TO_RS)
