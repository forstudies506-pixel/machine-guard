# explain.py
# ---------------------------------------------------------------------------
# Per-reading explanation with SHAP.
#
# get_feature_importance() (in model.py) is GLOBAL - "which features matter
# across all the training data". It cannot say why THIS one reading scored
# risky. SHAP can: it splits a single prediction into a contribution from
# each feature, measured in probability, that add up to the gap between the
# model's average output and this reading's output.
#
#   contribution > 0  ->  this feature pushed the reading TOWARD "failure"
#   contribution < 0  ->  this feature pushed it toward "healthy"
#
# For a Random Forest we use shap.TreeExplainer, which is exact and fast
# (no sampling). If the shap package is not installed, SHAP_AVAILABLE is
# False and the dashboard falls back to its simpler heuristic.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

import config
from model import features_from_row

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:                      # not installed, or failed to import
    SHAP_AVAILABLE = False

_explainer = None


def _get_explainer(model):
    """Build the TreeExplainer once and reuse it (it is a little expensive)."""
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_reading(model, latest):
    """
    Explain one reading.

    Returns a dict, or None if SHAP is unavailable:
      contributions  pandas Series (feature -> signed probability contribution
                     toward "failure"), ordered by absolute size, largest first
      base_value     the model's average failure probability (the starting
                     point every explanation builds from)
    """
    if not SHAP_AVAILABLE:
        return None

    x = features_from_row(latest)
    explainer = _get_explainer(model)
    explanation = explainer(x)

    # shap's output shape varies with version / classifier:
    #   (1, n_features, n_classes)  -> take the "failure" class (index 1)
    #   (1, n_features)             -> already the failure-side values
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        contrib = values[0, :, 1]
        base_value = float(np.ravel(base)[-1])
    else:
        contrib = values[0]
        base_value = float(np.ravel(base)[0])

    series = pd.Series(contrib, index=config.MODEL_FEATURE_COLUMNS)
    series = series.reindex(series.abs().sort_values(ascending=False).index)

    return {"contributions": series, "base_value": base_value}
