# MachineGuard

AI-assisted **predictive + prescriptive maintenance** for industrial machines.

**DETECT → DIAGNOSE → PREDICT → DECIDE → OPTIMIZE.** The point is not just
"predict failure" — it is to tell the operator *what* looks wrong, *why* the
model thinks so, *what happens* if the machine keeps running, and *which action*
makes financial sense.

Trained and tested on the **AI4I 2020 Predictive Maintenance dataset** (UCI /
Kaggle): 10,000 real milling-machine process readings, ~3.4 % of them failures.

---

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Put the dataset file **`predictive_maintenance.csv`** in the project root
(download it from Kaggle — it is deliberately not committed).

`shap` is in `requirements.txt` for the per-reading explanations. If it fails to
install on your platform, the app still runs — it falls back to a simpler
heuristic and prints nothing scary.

## 2. Run it

**The dashboard (failsafe, always works):**
```bash
streamlit run app.py
```
Opens on <http://localhost:8501>. The slider inspects any held-out reading;
**"Jump to a random failure"** for the demo. This is the reference UI — every
number here is straight from the models.

**The 3D front end wired to the real model:**
```bash
python api.py
```
Opens the interactive line at <http://localhost:5000>. `api.py` is a thin Flask
layer that trains the models once on startup and serves them as JSON
(`/api/meta`, `/api/reading/<i>`, `/api/random-failure`, `/api/find?pid=...`);
the front end (`prototype/index.html`) fetches from it. Type a real Product ID
(e.g. `M14860`), scrub the window, open **Full report** on a failing station for
the diagnosis, repair bill, cost comparison and SHAP breakdown — the same panels
as the Streamlit app.

If `api.py` is not running (e.g. the GitHub Pages copy in `docs/`), the front end
falls back to deterministic mock readings and still works as a concept piece.

## 3. Run the checks

There is no formal test suite yet. Quick confidence check:

```bash
python -c "import config, data, model, anomaly, diagnosis, decision, remedies, cost, explain; print('imports OK')"
```

---

## 4. Project map

| File | Responsibility |
|---|---|
| `config.py` | Every constant: dataset paths, feature list, the dataset's documented failure-physics thresholds, risk bands, cost assumptions. Import constants from here — never hard-code. |
| `data.py` | `load_dataset()` — read the CSV, drop ID columns, rename the binary target to `Failure`. |
| `model.py` | The failure-risk model (Random Forest) + held-out test split + global feature importance + a per-reading heuristic. |
| `anomaly.py` | Isolation Forest trained only on healthy rows — "does this reading look unlike normal operation?" |
| `diagnosis.py` | **Rule-based**, not ML. Checks the reading against the dataset's own documented failure criteria and names a likely mode. Never reads the true label. |
| `decision.py` | Combines every signal into one recommended action + urgency + reasons. Transparent rule ladder. |
| `remedies.py` | Knowledge base: each failure mode → ranked likely root causes, the fix for each, and an itemised repair-cost "bill". |
| `cost.py` | The OPTIMIZE step: `estimate_costs` (run-to-failure vs. maintain-now) **and** `repair_vs_replace` (repair cost vs. a share of a new machine). |
| `replacement.py` | A second model: equipment **price**, trained on `data.csv` (~1,700 used-equipment listings). Supplies the "new machine" figure `repair_vs_replace` needs. Heavy-equipment resale data, not milling machines — a demo stand-in; falls back to a config constant if unavailable. |
| `explain.py` | SHAP per-reading explanations (optional dependency). |
| `app.py` | The Streamlit dashboard. Wires all of the above together for one selected reading. |

**The front end is being rebuilt in plain HTML/CSS.** Everything except `app.py`
is plain Python with no Streamlit dependency — call those modules directly. A
clean service/API split is planned but not done yet, so for now the integration
contract is the function list in section 6.

---

## 5. The data & the pipeline

### Columns

| Column | Role |
|---|---|
| `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` | The 5 raw sensor features (`config.SENSOR_COLUMNS`). |
| `Mechanical power [W]`, `Temp difference [K]`, `Overstrain [min Nm]`, `Wear-speed [min rpm]` | **Engineered** in `data.preprocess()` — exact functions of the 5 sensors (the quantities the failure modes are defined on). `config.DERIVED_FEATURE_COLUMNS`. Together with the raw 5 they are `MODEL_FEATURE_COLUMNS`. |
| `Type` | Product quality variant `L` / `M` / `H`. Used by diagnosis and the per-type summary. **Not** a model feature (tested, it hurt recall). |
| `Failure` | Binary target — 1 = the machine failed on this reading. (Renamed from `Target` on load.) |
| `Failure Type` | The failure mode, or `No Failure`. **A second target.** Never used as a feature or shown to any model — only for after-the-fact scoring in the dashboard. |

> **Leakage rule:** `Failure` and `Failure Type` are both targets. Neither may
> ever enter a feature vector.

### What runs for one reading

```
reading ─┬─► anomaly.detect_anomaly / interpret_anomaly ──► anomaly_result
         ├─► model.predict_machine_risk ───────────────────► failure_risk (0–1)
         ├─► diagnosis.diagnose ───────────────────────────► diagnosis {headline, findings, symptoms}
         │        │
         │        └─ + rule-based status badge (Healthy / Warning / Critical)
         │
         ├─► decision.decide(failure_risk, anomaly_result, diagnosis, status)
         │                                                  ─► recommendation {action, urgency, reasons}
         ├─► remedies.repair_bill(diagnosis["headline"]) ──► bill {rows, total, expected}
         ├─► cost.estimate_costs(failure_risk, bill["expected"]) ─► maintain-now vs run-to-failure
         ├─► cost.repair_vs_replace(bill["expected"], replacement_cost) ─► "Repair" / "Replace"
         │        └─ replacement_cost from replacement.estimate_replacement_cost(...)
         └─► explain.explain_reading(model, reading) ──────► SHAP contributions (or None)
```

`status` (computed in `app.py`, not a module yet):
`Critical` if diagnosis found a named fault, else `Warning` if the heat gap is
tight or tool wear is high (`config.WARNING_*`), else `Healthy`.

---

## 6. Module API — integration contract

All "reading" arguments are one row of the dataset as a **pandas Series** (e.g.
`test_df.iloc[i]`), i.e. something indexable by the column names above.

### `data.py`

```python
load_dataset() -> pandas.DataFrame
```
All 10,000 rows, ID columns dropped, `Target` renamed to `Failure`.

### `model.py`

```python
train_model(df) -> (model, metrics, test_df)
```
- `model` — a fitted `RandomForestClassifier`.
- `metrics` — `dict`: `accuracy`, `precision`, `recall`, `f1` (floats 0–1),
  `confusion` (2×2 list `[[TN, FP], [FN, TP]]`), `n_test`, `n_test_failures`.
- `test_df` — the ~20 % of rows the model never saw, index reset. **The
  dashboard only ever inspects these**, so any risk it shows is genuinely
  out-of-sample.

```python
predict_machine_risk(model, reading) -> float          # 0.0–1.0 probability of failure
get_feature_importance(model) -> pandas.Series          # sensor -> importance, sums to 1, sorted desc
estimate_risk_contributions(model, reading) -> dict     # {"contributions": Series(%), "any_elevated": bool}
```
`estimate_risk_contributions` is the crude fallback for `explain_reading`.

### `anomaly.py`

```python
train_anomaly_model(df) -> IsolationForest              # fitted on df[Failure == 0] only
detect_anomaly(anomaly_model, reading) -> (prediction, score)
                                                        # prediction:  1 normal, -1 anomaly
                                                        # score:       >= 0 normal, < 0 unusual
interpret_anomaly(prediction, score) -> dict            # {"is_anomaly": bool, "label": str,
                                                        #  "explanation": str, "score": float}
```

### `diagnosis.py`

```python
diagnose(reading) -> dict
```
- `headline` — one string: `"Possible Heat Dissipation Failure"`,
  `"Possible Power Failure (underpower)"`, `"Possible Power Failure (overpower)"`,
  `"Possible Overstrain Failure"`, `"Possible Tool Wear Failure"`, or
  `"No Specific Fault Identified"`.
- `findings` — list of `{fault, reasons: [str], strength: int}`, strongest first.
- `symptoms` — list of readable strings (derived quantities: temp gap, power, …).

### `decision.py`

```python
decide(failure_risk, anomaly_result, diagnosis, status) -> dict
```
- `action` — `"Continue Operation"` / `"Reduce Load"` / `"Schedule Maintenance"` /
  `"Immediate Inspection"`.
- `urgency` — `"low"` / `"medium"` / `"high"` (use for colour).
- `reasons` — list of short strings.

### `remedies.py`

```python
get_remedies(headline) -> list[dict]                    # [] if the mode has no known causes
repair_bill(headline) -> dict | None                    # None if the mode has no remedy list
```
Each remedy: `{cause, likelihood (int %, sum to 100), fix, parts, cost (int)}`.
`repair_bill` returns `{rows: [remedy...], total: int, expected: int}` where
`expected` = Σ(likelihood/100 × cost) — the probability-weighted figure to plan
around.

> `likelihood` values are **FMEA-style engineering priors**, and `cost` values
> are **configurable demo figures** — neither is computed from the dataset. Label
> them as such in the UI.

### `cost.py`

```python
estimate_costs(failure_risk, repair_cost=None) -> dict
```
`repair_cost` defaults to `config.DEFAULT_REPAIR_COST`; pass `bill["expected"]`.
Returns:
- `maintain_now` — repair + a planned stop.
- `run_to_failure_expected` — `failure_risk × (emergency repair + long stop)`.
- `avoided` — `run_to_failure_expected − maintain_now` (positive ⇒ act now).
- `recommendation` — `"Maintain now"` / `"Running on is cheaper for now"`.
- `breakdown` — every line item, for display.

> All amounts come from `config.py` and are **configurable demo assumptions, not
> quotes**.

```python
repair_vs_replace(repair_cost, replacement_cost=None) -> dict
```
`replacement_cost` defaults to `config.DEFAULT_REPLACEMENT_COST`; pass the value
from `replacement.estimate_replacement_cost(...)`. Returns `verdict`
(`"Repair"` / `"Replace"`), `repair_cost`, `replacement_cost`, `threshold`
(`config.REPLACE_COST_FRACTION x replacement_cost`), `ratio`, `fraction`.

### `replacement.py`

```python
train_replacement_model() -> (model, meta)          # meta: feature_columns, metrics, report
estimate_replacement_cost(model, feature_columns, profile=None) -> int   # in config.CURRENCY
```
Trains a `RandomForestRegressor` on `data.csv` (`log(price_usd)`), held-out
R² ~0.89. `profile` picks the category/manufacturer/region codes to price;
defaults to `config.REPLACEMENT_PROFILE`. Prediction is for a *new* equivalent
(age 0, 0 hours). Converted to rupees with `config.USD_TO_RS`. **Heavy-equipment
resale data in USD — a demo stand-in, not milling-machine prices.**

### `explain.py`

```python
SHAP_AVAILABLE: bool                                    # False if the shap package is missing
explain_reading(model, reading) -> dict | None          # None when SHAP_AVAILABLE is False
```
Returns `{contributions: Series, base_value: float}`. `contributions` is signed,
in probability points toward *failure* (positive) or *healthy* (negative), and
`base_value + sum(contributions) == predict_machine_risk(...)`. The base value is
~0.5 because the model uses `class_weight="balanced"`.

---

## 7. Honesty / limitations — keep these in the pitch

- **One public dataset, one machine.** Metrics describe *this* milling dataset,
  not "real-world accuracy" for any other equipment.
- **Diagnosis is rules, not AI.** It re-checks the dataset's own documented
  failure physics. Call it a rule engine, not a model.
- **Root-cause percentages and all rupee costs are assumptions**, not learned
  values — the dataset has no root-cause or cost information. A real deployment
  would learn them from the plant's maintenance logs.
- **The "new machine" price for repair-vs-replace is a learned estimate from a
  DIFFERENT dataset** (`data.csv` — heavy-equipment resale listings in USD). It
  shows the method works and gives a demo figure; it is not a milling-machine
  quote. The repair-vs-replace verdict is almost always "Repair" here because
  these faults are cheap relative to the whole machine — which is the correct
  real-world answer.
- **`Failure Type` is shown in the dashboard only to check our own work.** It is
  never given to any model or rule.
- **Test recall ~0.84, precision ~0.78, F1 ~0.81** (validation ~0.79 across the
  board — close to test, so not over-fitted). The model still misses ~1 real
  failure in 6. Accuracy (~98.7 %) is the least useful number at a 3.4 % failure
  rate — quote recall and precision.
- **Most of that performance is engineered features, not the algorithm.** The 4
  physics features (`data.py`) are exact functions of the 5 raw sensors — the
  quantities the failure modes are defined on. No leakage; they just hand the
  model the combination instead of making it rediscover it.

---

## 8. Current status

Working end to end: DETECT → DIAGNOSE → PREDICT → DECIDE → OPTIMIZE.
Open: polished single-screen layout, and splitting the ML logic behind a clean
API for the new HTML/CSS front end.
