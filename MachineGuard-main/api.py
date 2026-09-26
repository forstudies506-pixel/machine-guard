# api.py
# ---------------------------------------------------------------------------
# Thin HTTP layer so the 3D front end (prototype/index.html) can show REAL
# model output instead of mock data. It reuses the exact same modules as the
# Streamlit app - nothing is re-implemented here except the status badge,
# which lives in app.py.
#
# app.py (Streamlit) stays as the standalone failsafe. This file is additive.
#
#   pip install -r requirements.txt      # adds flask + flask-cors
#   python api.py                        # serves on http://localhost:5000
#
# Endpoints:
#   GET /                     -> the 3D front end
#   GET /api/meta             -> dataset stats, metrics, importances, per-type,
#                                failure-mode mix, preprocessing report,
#                                sensor histograms, per-mode economics
#   GET /api/reading/<i>      -> the full pipeline run on held-out test row i
#   GET /api/random-failure   -> {"index": i} for a random held-out failure
#   GET /api/find?pid=M14860  -> {"index": i} for that product ID (or 404)
# ---------------------------------------------------------------------------

import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split

import config
from data import load_dataset_with_report
from model import train_model, predict_machine_risk, get_feature_importance
from anomaly import train_anomaly_model, detect_anomaly, interpret_anomaly
from diagnosis import diagnose
from decision import decide
from remedies import repair_bill
from cost import estimate_costs, repair_vs_replace
from explain import explain_reading, SHAP_AVAILABLE
from replacement import train_replacement_model, estimate_replacement_cost

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "prototype"), static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# Train once on startup (a few seconds), keep everything in memory.
# ---------------------------------------------------------------------------
print("MachineGuard API: loading data and training models ...")
DF, REPORT = load_dataset_with_report()
MODEL, METRICS, TEST_DF = train_model(DF)
ANOM = train_anomaly_model(DF)
FI = get_feature_importance(MODEL)

# Second model: equipment price, for the "repair vs replace" call. Trains
# on data.csv. If anything about that file is wrong we fall back to the
# config constant rather than take the API down.
try:
    REPL_MODEL, REPL_META = train_replacement_model()
    REPLACEMENT_COST = estimate_replacement_cost(
        REPL_MODEL, REPL_META["feature_columns"]
    )
    REPL_OK = True
except Exception as exc:                      # pragma: no cover - defensive
    print(f"  replacement model unavailable ({exc}); using config default")
    REPL_META = {"metrics": {}, "report": {}, "median_price_usd": None}
    REPLACEMENT_COST = config.DEFAULT_REPLACEMENT_COST
    REPL_OK = False

# Re-run the SAME split (same seed) just to recover the original row indices,
# so we can map a held-out test row back to its Product ID in the raw file.
_RAW = pd.read_csv(config.DATASET_PATH, encoding="utf-8-sig")
_train_val, _test = train_test_split(
    DF, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED,
    stratify=DF[config.TARGET_COLUMN],
)
TEST_ORIG_INDEX = list(_test.index)                       # positions in DF / _RAW
PRODUCT_IDS = [str(_RAW.iloc[k]["Product ID"]) for k in TEST_ORIG_INDEX]
PID_TO_ROW = {pid.upper(): i for i, pid in enumerate(PRODUCT_IDS)}

# Every Product ID in the WHOLE file -> its raw row number. Lets the search
# box resolve any real ID (most are training rows) to the NEAREST held-out
# reading, instead of just 404-ing.
_ALL_PID_TO_RAW = {
    str(pid).upper(): k for k, pid in enumerate(_RAW["Product ID"].tolist())
}
_TEST_ORIG_SORTED = sorted(range(len(TEST_ORIG_INDEX)), key=lambda i: TEST_ORIG_INDEX[i])


def _nearest_heldout(raw_index):
    """Held-out test-row position whose original row number is closest to
    raw_index."""
    best_i, best_d = 0, None
    for i in _TEST_ORIG_SORTED:
        d = abs(TEST_ORIG_INDEX[i] - raw_index)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
        elif TEST_ORIG_INDEX[i] > raw_index:
            break
    return best_i

SENSORS = list(config.SENSOR_COLUMNS)
# short labels for every model feature (5 raw sensors + 4 engineered)
SHORT = {c: config.SENSOR_DISPLAY[c][0] for c in config.SENSOR_DISPLAY}
# stable short codes the front end keys on
CODE = {
    "Air temperature [K]": "AIR",
    "Process temperature [K]": "PROC",
    "Rotational speed [rpm]": "RPM",
    "Torque [Nm]": "TORQUE",
    "Tool wear [min]": "WEAR",
    "Mechanical power [W]": "POWER",
    "Temp difference [K]": "TEMPGAP",
    "Overstrain [min Nm]": "OVERSTRAIN",
    "Wear-speed [min rpm]": "WEARSPEED",
}

# population stats for percentiles + histograms (whole dataset)
_RANGES = {
    c: {
        "min": float(DF[c].min()), "max": float(DF[c].max()),
        "mean": float(DF[c].mean()), "p50": float(DF[c].median()),
        "sd": float(DF[c].std()),
    }
    for c in SENSORS
}
_HIST = {}
for c in SENSORS:
    counts, edges = np.histogram(DF[c], bins=22)
    _HIST[c] = {"counts": [int(x) for x in counts], "edges": [float(x) for x in edges]}

_FAILED = DF[DF[config.TARGET_COLUMN] == 1]
_MODE_COUNTS = (
    _FAILED[_FAILED[config.FAILURE_TYPE_COLUMN] != "No Failure"]
    [config.FAILURE_TYPE_COLUMN].value_counts()
)

# map a documented-mode headline -> the fault key the front end uses
HEADLINE_KEY = {
    "Possible Heat Dissipation Failure": "hdf",
    "Possible Power Failure (underpower)": "pwf",
    "Possible Power Failure (overpower)": "pwf",
    "Possible Overstrain Failure": "osf",
    "Possible Tool Wear Failure": "twf",
    "No Specific Fault Identified": None,
}
KEY_PART = {"hdf": "cooling", "pwf": "motor", "osf": "table", "twf": "tool", "spindle": "gears"}


def _status(row, diag):
    gap = row["Process temperature [K]"] - row["Air temperature [K]"]
    wear = row["Tool wear [min]"]
    if diag["headline"] != "No Specific Fault Identified":
        return "Critical"
    if gap < config.WARNING_TEMP_DIFF_K or wear >= config.WARNING_TOOLWEAR_MIN:
        return "Warning"
    return "Healthy"


def _hot(col, value):
    """0 = normal, 1 = drifting, 2 = extreme (vs the healthy baseline)."""
    base = config.SENSOR_BASELINES[col]
    noise = config.SENSOR_NOISE[col] or 1.0
    sig = abs(value - base) / noise
    return 2 if sig >= 3.0 else 1 if sig >= 1.6 else 0


_CACHE = {}


def _reading(i):
    if i in _CACHE:
        return _CACHE[i]
    row = TEST_DF.iloc[i]
    risk = predict_machine_risk(MODEL, row)
    pred, score = detect_anomaly(ANOM, row)
    anom = interpret_anomaly(pred, score)
    diag = diagnose(row)
    status = _status(row, diag)
    rec = decide(risk, anom, diag, status)
    bill = repair_bill(diag["headline"])
    _repair_cost = bill["expected"] if bill else config.DEFAULT_REPAIR_COST
    costs = estimate_costs(risk, bill["expected"] if bill else None)
    rvr = repair_vs_replace(_repair_cost, REPLACEMENT_COST)
    shap = explain_reading(MODEL, row)

    key = HEADLINE_KEY.get(diag["headline"])
    band = "high" if risk >= config.RISK_HIGH else "medium" if risk >= config.RISK_MEDIUM else "low"

    sensors = {}
    for c in SENSORS:
        unit = config.SENSOR_DISPLAY[c][1]
        v = float(row[c])
        sensors[CODE[c]] = {
            "code": CODE[c], "label": SHORT[c],
            "value": round(v, 1), "unit": unit, "text": f"{v:.1f} {unit}",
            "hot": _hot(c, v),
            "pctl": float((DF[c] < v).mean()),
        }

    shap_list = None
    if shap:
        s = shap["contributions"]
        shap_list = [
            {"code": CODE.get(k, k), "feature": SHORT.get(k, k), "value": float(val)}
            for k, val in s.items()
        ]

    payload = {
        "index": i,
        "product_id": PRODUCT_IDS[i],
        "type": str(row[config.TYPE_COLUMN]),
        "fault_key": key,
        "part": KEY_PART.get(key),
        "status": status,
        "failure_risk": float(risk),
        "risk_band": band,
        "anomaly": {
            "is_anomaly": bool(anom["is_anomaly"]),
            "label": anom["label"],
            "explanation": anom["explanation"],
            "score": float(anom["score"]),
        },
        "diagnosis": {
            "headline": diag["headline"],
            "findings": [
                {"fault": f["fault"], "reasons": list(f["reasons"])}
                for f in diag["findings"]
            ],
            "symptoms": list(diag["symptoms"]),
        },
        "recommendation": {
            "action": rec["action"], "urgency": rec["urgency"],
            "reasons": list(rec["reasons"]),
        },
        "repair_bill": None if not bill else {
            "rows": [
                {"cause": r["cause"], "likelihood": r["likelihood"],
                 "fix": r["fix"], "parts": r["parts"], "cost": r["cost"]}
                for r in bill["rows"]
            ],
            "total": bill["total"], "expected": bill["expected"],
        },
        "costs": costs,
        "repair_vs_replace": rvr,
        "shap": shap_list,
        "ground_truth": {
            "failed": int(row[config.TARGET_COLUMN]) == 1,
            "failure_type": str(row[config.FAILURE_TYPE_COLUMN]),
        },
        "sensors": sensors,
        "currency": config.CURRENCY,
    }
    _CACHE[i] = payload
    return payload


@app.get("/api/meta")
def meta():
    total = len(DF)
    failures = int(DF[config.TARGET_COLUMN].sum())
    per_type = []
    for t in config.TYPE_VALUES:
        sub = DF[DF[config.TYPE_COLUMN] == t]
        sub_f = sub[sub[config.TARGET_COLUMN] == 1]
        named = sub_f[sub_f[config.FAILURE_TYPE_COLUMN] != "No Failure"]
        per_type.append({
            "type": t,
            "rows": int(len(sub)),
            "fleet_share": round(len(sub) / total * 100, 1),
            "failures": int(len(sub_f)),
            "failure_rate": round(len(sub_f) / len(sub) * 100, 2),
            "top_mode": (named[config.FAILURE_TYPE_COLUMN].value_counts().index[0]
                         if len(named) else "-"),
        })

    econ_by_mode = []
    for headline, key in [
        ("Possible Heat Dissipation Failure", "hdf"),
        ("Possible Power Failure (underpower)", "pwf"),
        ("Possible Overstrain Failure", "osf"),
        ("Possible Tool Wear Failure", "twf"),
    ]:
        bill = repair_bill(headline)
        # a representative risk for the mode = mean model risk over test rows
        # the diagnosis engine tags with it
        c = estimate_costs(0.75, bill["expected"] if bill else None)
        econ_by_mode.append({
            "mode": headline.replace("Possible ", ""),
            "maintain": c["maintain_now"],
            "run": c["run_to_failure_expected"],
            "saved": c["avoided"],
        })

    rep = dict(REPORT)
    for k, v in list(rep.items()):
        if isinstance(v, tuple):
            rep[k] = list(v)

    return jsonify({
        "n_test": METRICS["split"]["test"],
        "n_test_failures": int(TEST_DF[config.TARGET_COLUMN].sum()),
        "metrics": METRICS,
        "feature_importance": [
            {"code": CODE.get(k, k), "label": SHORT.get(k, k), "value": float(v)}
            for k, v in FI.items()
        ],
        "failure_mode_counts": {k: int(v) for k, v in _MODE_COUNTS.items()},
        "per_type": per_type,
        "preprocessing": rep,
        "class_balance": {
            "rows": total, "failures": failures,
            "non_failures": total - failures,
            "failure_rate": round(failures / total * 100, 2),
        },
        "sensor_ranges": {CODE[c]: dict(_RANGES[c], label=SHORT[c], unit=config.SENSOR_DISPLAY[c][1]) for c in SENSORS},
        "sensor_hist": {CODE[c]: dict(_HIST[c], label=SHORT[c]) for c in SENSORS},
        "economics_by_mode": econ_by_mode,
        "risk_bands": {"medium": config.RISK_MEDIUM, "high": config.RISK_HIGH},
        "shap_available": bool(SHAP_AVAILABLE),
        "product_ids": PRODUCT_IDS,
        "sensor_order": [CODE[c] for c in SENSORS],
        "currency": config.CURRENCY,
        "replacement": {
            "trained": REPL_OK,
            "cost": REPLACEMENT_COST,
            "fraction": config.REPLACE_COST_FRACTION,
            "metrics": REPL_META.get("metrics", {}),
            "median_price_usd": REPL_META.get("median_price_usd"),
            "usd_to_rs": config.USD_TO_RS,
            "report": REPL_META.get("report", {}),
        },
    })


@app.get("/api/reading/<int:i>")
def reading(i):
    if i < 0 or i >= len(TEST_DF):
        return jsonify({"error": "index out of range", "max": len(TEST_DF) - 1}), 404
    return jsonify(_reading(i))


@app.get("/api/random-failure")
def random_failure():
    fails = list(np.where(TEST_DF[config.TARGET_COLUMN].values == 1)[0])
    if not fails:
        return jsonify({"index": 0})
    return jsonify({"index": int(np.random.choice(fails))})


@app.get("/api/find")
def find():
    pid = (request.args.get("pid") or "").strip().upper()

    # 1. exact hit in the held-out set
    if pid in PID_TO_ROW:
        return jsonify({"index": PID_TO_ROW[pid], "product_id": pid})

    # 2. a bare number = "test row N"
    if pid.isdigit() and 0 <= int(pid) < len(TEST_DF):
        return jsonify({"index": int(pid), "product_id": PRODUCT_IDS[int(pid)]})

    # 3. a real Product ID that is a TRAINING row (most of them are). The
    #    dashboard only inspects held-out rows, so jump to the nearest one
    #    in the dataset and say so.
    if pid in _ALL_PID_TO_RAW:
        j = _nearest_heldout(_ALL_PID_TO_RAW[pid])
        return jsonify({
            "index": j,
            "product_id": PRODUCT_IDS[j],
            "requested": pid,
            "note": (f"{pid} is a training row - showing the nearest held-out "
                     f"reading, {PRODUCT_IDS[j]}"),
        })

    return jsonify({"error": "no such Product ID in the dataset", "query": pid}), 404


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    print(f"MachineGuard API ready - {len(TEST_DF)} held-out readings, "
          f"SHAP {'on' if SHAP_AVAILABLE else 'off'}.  http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
