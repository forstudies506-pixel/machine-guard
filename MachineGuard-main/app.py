import random

import pandas as pd
import streamlit as st

import config
from data import load_dataset_with_report
from model import (
    train_model,
    predict_machine_risk,
    get_feature_importance,
    estimate_risk_contributions,
)
from anomaly import train_anomaly_model, detect_anomaly, interpret_anomaly
from diagnosis import diagnose
from decision import decide
from remedies import repair_bill
from cost import estimate_costs, repair_vs_replace
from explain import explain_reading
from replacement import train_replacement_model, estimate_replacement_cost

st.set_page_config(
    page_title="MachineGuard",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached setup
#
# Streamlit re-runs this whole script on every interaction. The decorators
# make each function run ONCE and reuse the result:
#   @st.cache_data     -> plain data (our DataFrame)
#   @st.cache_resource -> heavy objects kept as-is (our trained models)
# ---------------------------------------------------------------------------

@st.cache_data
def get_dataset_and_report():
    return load_dataset_with_report()          # (clean_df, preprocessing report)


def get_dataset():
    df, _report = get_dataset_and_report()
    return df


@st.cache_resource
def get_failure_model():
    return train_model(get_dataset())          # returns (model, metrics, test_df)


@st.cache_resource
def get_anomaly_model():
    return train_anomaly_model(get_dataset())


@st.cache_resource
def get_replacement_model():
    """Price model for 'repair vs replace'. Falls back to the config
    constant if data.csv can't be used."""
    try:
        rmodel, rmeta = train_replacement_model()
        cost = estimate_replacement_cost(rmodel, rmeta["feature_columns"])
        return cost, rmeta, True
    except Exception:
        return config.DEFAULT_REPLACEMENT_COST, {"metrics": {}}, False


df, prep_report = get_dataset_and_report()      # all 10,000 rows + how they were cleaned
model, metrics, test_df = get_failure_model()  # test_df = rows the model never saw / used
anomaly_model = get_anomaly_model()
replacement_cost, replacement_meta, replacement_ok = get_replacement_model()

# short label + unit for each sensor, for tidy display
SHORT = {col: config.SENSOR_DISPLAY[col][0] for col in config.SENSOR_COLUMNS}


st.title("MachineGuard")
st.subheader("AI-powered Predictive Maintenance System")
st.write(
    "Trained on the **AI4I 2020 Predictive Maintenance dataset** (10,000 real "
    "milling-machine process readings, ~3.4% failures). The readings you can "
    f"inspect below are the **{len(test_df)} held-out rows** the model never "
    "saw during training - so every score shown is a true out-of-sample result."
)

st.divider()


# ---------------------------------------------------------------------------
# Pick which held-out reading to inspect
#
# The dataset has no time axis - independent readings - so we let the user
# choose one from the held-out test set. The button jumps straight to a real
# failure for the demo.
# ---------------------------------------------------------------------------

if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0

pick_col, jump_col = st.columns([3, 1])

with jump_col:
    if st.button("Jump to a random failure", use_container_width=True):
        failure_rows = list(test_df.index[test_df[config.TARGET_COLUMN] == 1])
        st.session_state.row_idx = int(random.choice(failure_rows))

with pick_col:
    picked = st.slider(
        "Held-out reading number",
        min_value=1,
        max_value=len(test_df),
        value=st.session_state.row_idx + 1,
    )

st.session_state.row_idx = picked - 1
row_idx = st.session_state.row_idx
latest = test_df.iloc[row_idx]


# ---------------------------------------------------------------------------
# Run every model / rule on the selected reading
# ---------------------------------------------------------------------------

anomaly_prediction, anomaly_score = detect_anomaly(anomaly_model, latest)
anomaly_result = interpret_anomaly(anomaly_prediction, anomaly_score)

failure_risk = predict_machine_risk(model, latest)

diagnosis = diagnose(latest)

# Rule-based status badge, built from the SAME documented physics as the
# diagnosis engine - NOT from the ML model.
temp_gap = latest["Process temperature [K]"] - latest["Air temperature [K]"]
tool_wear = latest["Tool wear [min]"]

if diagnosis["headline"] != "No Specific Fault Identified":
    status = "Critical"
elif temp_gap < config.WARNING_TEMP_DIFF_K or tool_wear >= config.WARNING_TOOLWEAR_MIN:
    status = "Warning"
else:
    status = "Healthy"

recommendation = decide(failure_risk, anomaly_result, diagnosis, status)


# ---------------------------------------------------------------------------
# Machine status + sensor readings
# ---------------------------------------------------------------------------

st.header(f"Reading #{row_idx + 1}  ·  product type {latest[config.TYPE_COLUMN]}")

if status == "Healthy":
    st.success("Machine Status: Healthy")
elif status == "Warning":
    st.warning("Machine Status: Warning")
else:
    st.error("Machine Status: Critical")

cols = st.columns(len(config.SENSOR_COLUMNS))
for col_box, sensor in zip(cols, config.SENSOR_COLUMNS):
    name, unit = config.SENSOR_DISPLAY[sensor]
    with col_box:
        st.metric(name, f"{latest[sensor]:.1f} {unit}")


st.divider()


# ---------------------------------------------------------------------------
# Data & preprocessing  (Review 1: "the preprocessing isn't clear")
# ---------------------------------------------------------------------------

st.header("Data & preprocessing")

r = prep_report
st.write(
    f"**Raw file:** {r['raw_shape'][0]:,} rows x {r['raw_shape'][1]} columns"
    f"  ->  **after cleaning:** {r['clean_shape'][0]:,} rows x "
    f"{r['clean_shape'][1]} columns"
)

_renamed_from = list(r["renamed_column"])[0]
_renamed_to = r["renamed_column"][_renamed_from]
prep_steps = pd.DataFrame(
    [
        ["1. Drop identifier columns",
         f"{', '.join(r['dropped_columns'])} - no failure signal; keeping them "
         "would let the model memorise individual rows"],
        ["2. Rename target column", f"'{_renamed_from}'  ->  '{_renamed_to}'"],
        ["3. Missing values", f"{r['missing_values']} found (checked, not assumed)"],
        ["3. Duplicate rows", f"{r['duplicate_rows']} found"],
        ["4. Feature scaling", r["feature_scaling"]],
        ["5. Class imbalance", r["imbalance_handling"]],
    ],
    columns=["Step", "What & why"],
)
st.table(prep_steps)

_excl = list(r["feature_excluded"])[0]
st.write(
    f"**Features into the model ({len(r['features_used'])}):** "
    + ", ".join(r["features_used"])
)
st.caption(f"Excluded - **{_excl}**: {r['feature_excluded'][_excl]}")
st.caption(
    f"Target: **{r['target']}** (0/1). The second target "
    f"**{r['second_target_not_a_feature']}** is never used as an input "
    "(that would be label leakage)."
)

cb = r["class_balance"]
st.write(
    f"**Class balance:** {cb['failures']} failures vs "
    f"{cb['non_failures']:,} non-failures = "
    f"**{cb['failure_rate'] * 100:.2f}%** failure rate - heavily imbalanced, "
    "which is why accuracy alone is misleading below."
)

sp = metrics["split"]
st.write(
    f"**Split - 60 / 20 / 20, stratified** (failure rate preserved in each): "
    f"train **{sp['train']:,}**  ·  validation **{sp['validation']:,}**  ·  "
    f"test **{sp['test']:,}**"
)


st.divider()


# ---------------------------------------------------------------------------
# Model evaluation  (Review 1: confusion matrix + validation eval)
# ---------------------------------------------------------------------------

st.header("Model evaluation")


def _pct(x):
    return f"{x * 100:.1f}%"


compare = pd.DataFrame({
    "Validation (20%)": {
        metric.capitalize(): _pct(metrics["validation"][metric])
        for metric in ("accuracy", "precision", "recall", "f1")
    },
    "Test (20%)": {
        metric.capitalize(): _pct(metrics["test"][metric])
        for metric in ("accuracy", "precision", "recall", "f1")
    },
})
st.table(compare)
st.caption(
    "Validation is what we watch while building; test is scored once at the "
    "end. The two columns are close, so the model is not over-fitted. "
    "ACCURACY IS MISLEADING here - a model that never predicts failure still "
    "scores ~97% - so recall (failures caught) and precision (alerts that "
    "were right) are what matter."
)


def _cm_frame(conf):
    (tn, fp), (fn, tp) = conf
    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual: OK", "Actual: FAIL"],
        columns=["Predicted: OK", "Predicted: FAIL"],
    )


def _cm_colours(frame):
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css.iloc[0, 0] = "background-color:#14532d;color:#ffffff"   # TN - correct
    css.iloc[1, 1] = "background-color:#14532d;color:#ffffff"   # TP - correct
    css.iloc[0, 1] = "background-color:#7f1d1d;color:#ffffff"   # FP - error
    css.iloc[1, 0] = "background-color:#7f1d1d;color:#ffffff"   # FN - error
    return css


st.write("**Confusion matrix** (green = correct, red = error)")
cm_val, cm_test = st.columns(2)
for box, title, key in [
    (cm_val, "Validation set", "validation"),
    (cm_test, "Test set", "test"),
]:
    s = metrics[key]
    (tn, fp), (fn, tp) = s["confusion"]
    with box:
        st.write(f"_{title} - {s['n']} readings, {s['n_failures']} real failures_")
        st.dataframe(
            _cm_frame(s["confusion"]).style.apply(_cm_colours, axis=None),
            use_container_width=True,
        )
        st.caption(
            f"Caught {tp} of {tp + fn} failures (recall {s['recall'] * 100:.0f}%), "
            f"{fp} false alarms, {fn} missed."
        )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

st.subheader("Anomaly Detection (Isolation Forest)")
st.caption(
    "Trained only on the non-failed readings - it is never shown a failure. "
    "It flags when a reading stops looking like normal operation."
)

a1, a2 = st.columns([2, 1])
with a1:
    if anomaly_result["is_anomaly"]:
        st.error(f"Anomaly Status: {anomaly_result['label']}")
    else:
        st.success(f"Anomaly Status: {anomaly_result['label']}")
    st.write(anomaly_result["explanation"])
with a2:
    st.metric("Anomaly Score", f"{anomaly_result['score']:.3f}")
    st.caption(
        "At or above 0 resembles the healthy baseline. Below 0 means the "
        "reading deviates from it - the lower, the more unusual."
    )


# ---------------------------------------------------------------------------
# Failure prediction
# ---------------------------------------------------------------------------

st.subheader("AI failure prediction")

st.metric("Failure Risk", f"{failure_risk * 100:.1f}%")
st.caption(
    "This reading was held out of training, so this is a genuine "
    "out-of-sample probability - not a memorised value."
)

if failure_risk >= config.RISK_HIGH:
    risk_level = "High"
elif failure_risk >= config.RISK_MEDIUM:
    risk_level = "Medium"
else:
    risk_level = "Low"

if risk_level == "High":
    st.error("AI Risk Level: High")
elif risk_level == "Medium":
    st.warning("AI Risk Level: Medium")
else:
    st.success("AI Risk Level: Low")


st.divider()


# ---------------------------------------------------------------------------
# Likely problem (rule-based, real documented physics)
# ---------------------------------------------------------------------------

st.header("Likely Problem")
st.caption(
    "Checks the reading against the AI4I dataset's OWN documented failure "
    "criteria (heat dissipation, power, overstrain, tool wear). POSSIBLE "
    "causes to guide inspection - the true failure label is never used here."
)

if diagnosis["headline"] == "No Specific Fault Identified":
    st.info("No specific fault identified")
    for reason in diagnosis["findings"][0]["reasons"]:
        st.write(f"- {reason}")
else:
    st.warning(f"Most likely: {diagnosis['headline']}")
    for position, finding in enumerate(diagnosis["findings"]):
        if position == 0:
            st.write("**Why the system thinks so:**")
        elif position == 1:
            st.write("**Other possible causes also consistent with the data:**")
        st.write(f"_{finding['fault']}_")
        for reason in finding["reasons"]:
            st.write(f"- {reason}")


# ---- Ground-truth check (transparency only) ------------------------------

st.subheader("Ground truth for this reading")

actual_failed = int(latest[config.TARGET_COLUMN]) == 1
actual_type = latest[config.FAILURE_TYPE_COLUMN]

g1, g2 = st.columns(2)
with g1:
    if actual_failed:
        st.error(f"Labelled a failure: {actual_type}")
    else:
        st.success("Labelled: No Failure")
with g2:
    st.write(f"Our diagnosis: **{diagnosis['headline']}**")
    if actual_failed and actual_type != "Random Failures":
        key = actual_type.split()[0].lower()
        if key in diagnosis["headline"].lower():
            st.write("Matches the true failure mode.")
        else:
            st.write("Does not match the true failure mode.")
    elif actual_type == "Random Failures":
        st.write("True mode is 'Random Failures' - no sensor cause exists to find.")

st.caption(
    "The dataset ships the true label. We show it here only to check our own "
    "work - it is never given to any model or rule."
)


st.divider()


# ---------------------------------------------------------------------------
# Risk contributors
# ---------------------------------------------------------------------------

st.header("Risk Contributors")
st.caption(
    "How much each sensor contributes to the failure model's decisions, "
    "across all of its training data (global feature importance)."
)

importance = get_feature_importance(model).rename(index=SHORT)
st.bar_chart(importance)
for sensor, value in importance.items():
    st.write(f"- **{sensor}**: {value * 100:.0f}%")

st.subheader("What's driving risk in this reading")

shap_result = explain_reading(model, latest)

if shap_result is not None:
    st.caption(
        "SHAP - each feature's contribution to THIS reading's failure "
        "probability. Positive bars pushed the reading toward failure, "
        "negative bars toward healthy. They build up from the model's "
        f"average output ({shap_result['base_value'] * 100:.0f}%) to this "
        f"reading's {failure_risk * 100:.0f}%."
    )
    shap_contributions = shap_result["contributions"].rename(index=SHORT)
    st.bar_chart(shap_contributions)
    for feature, value in shap_contributions.items():
        direction = "toward failure" if value >= 0 else "toward healthy"
        st.write(f"- **{feature}**: {value * 100:+.1f} pts ({direction})")
else:
    st.caption(
        "Heuristic (SHAP not installed): each sensor's global importance "
        "times how far it sits from its healthy mean (either direction), "
        "scaled to 100%."
    )
    risk_contributions = estimate_risk_contributions(model, latest)
    if failure_risk < config.RISK_MEDIUM:
        st.info(
            f"Failure risk is low ({failure_risk * 100:.0f}%). Nothing "
            "meaningful to break down."
        )
    elif risk_contributions["any_elevated"]:
        contributions = risk_contributions["contributions"].rename(index=SHORT)
        st.bar_chart(contributions)
        for sensor, value in contributions.items():
            st.write(f"- **{sensor}**: {value:.0f}%")
    else:
        st.info(
            "No single sensor stands out as the driver of the current risk."
        )


st.divider()


# ---------------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------------

st.header("Recommended Action")
st.caption(
    "Combines status, failure risk, anomaly result and likely problem into "
    "one suggested next step. Decision support - not a safety guarantee."
)

if recommendation["urgency"] == "high":
    st.error(f"Recommended: {recommendation['action']}")
elif recommendation["urgency"] == "medium":
    st.warning(f"Recommended: {recommendation['action']}")
else:
    st.success(f"Recommended: {recommendation['action']}")

st.write("**Why:**")
for reason in recommendation["reasons"]:
    st.write(f"- {reason}")


# ---- Failure-specific root causes, fixes & repair bill ----------------

bill = repair_bill(diagnosis["headline"])
CUR = config.CURRENCY

if bill:
    st.subheader(f"Likely root causes & repair bill - {diagnosis['headline']}")
    st.caption(
        "Causes for this failure mode, ranked, each with the fix and what "
        "that fix costs (parts + labour). The percentages are engineering-"
        "reference priors (FMEA-style) and the amounts are configurable demo "
        f"figures in {CUR} - NEITHER is computed from the dataset. A real "
        "deployment would learn both from the plant's own maintenance logs."
    )

    bill_rows = []
    for rank, r in enumerate(bill["rows"], start=1):
        bill_rows.append({
            "#": rank,
            "Likely cause": r["cause"],
            "Prob": f"{r['likelihood']}%",
            "Fix": r["fix"],
            "Parts / labour": r["parts"],
            f"Est. cost ({CUR})": f"{r['cost']:,}",
        })
    st.dataframe(
        pd.DataFrame(bill_rows), use_container_width=True, hide_index=True
    )

    st.write(
        f"**Probability-weighted repair estimate: {CUR} {bill['expected']:,}** "
        f"- the single figure to plan around (each cause pulls it toward its "
        f"own cost by how likely it is). If every listed check had to be done: "
        f"{CUR} {bill['total']:,}."
    )


st.divider()


# ---------------------------------------------------------------------------
# OPTIMIZE - maintain now vs run to failure
# ---------------------------------------------------------------------------

st.header("Cost: maintain now vs run to failure")
st.caption(
    "Weighs a planned stop now against the EXPECTED cost of a breakdown "
    f"(= failure risk x breakdown impact). All {CUR} figures are configurable "
    "demo assumptions from config.py, not quotes."
)

repair_cost = bill["expected"] if bill else None
costs = estimate_costs(failure_risk, repair_cost)
b = costs["breakdown"]

col_now, col_fail, col_save = st.columns(3)
col_now.metric("Maintain now", f"{CUR} {costs['maintain_now']:,}")
col_fail.metric(
    "Expected cost if run to failure",
    f"{CUR} {costs['run_to_failure_expected']:,}",
)
col_save.metric(
    "Potential saving from acting now", f"{CUR} {costs['avoided']:,}"
)

if costs["avoided"] > 0:
    st.success(
        f"{costs['recommendation']} - at {failure_risk * 100:.0f}% failure "
        "risk, acting now is the cheaper expected path."
    )
else:
    st.info(
        f"{costs['recommendation']} - at {failure_risk * 100:.0f}% failure "
        "risk the expected breakdown cost is still below a planned stop. "
        "Keep monitoring."
    )

st.write("**How it's calculated**")
st.write(
    f"- Maintain now = repair {CUR} {b['repair_cost']:,} + planned downtime "
    f"({config.PLANNED_DOWNTIME_HOURS} h x {CUR} "
    f"{config.DOWNTIME_COST_PER_HOUR:,}/h = {CUR} {b['planned_downtime']:,})"
)
st.write(
    f"- Breakdown impact = emergency repair {CUR} {b['emergency_repair']:,} "
    f"({config.EMERGENCY_REPAIR_MULTIPLIER}x the planned repair) + failure "
    f"downtime ({config.FAILURE_DOWNTIME_HOURS} h = {CUR} "
    f"{b['failure_downtime']:,}) = {CUR} {b['breakdown_impact']:,}"
)
st.write(
    f"- Expected cost if run to failure = {failure_risk * 100:.0f}% risk x "
    f"{CUR} {b['breakdown_impact']:,} = {CUR} "
    f"{costs['run_to_failure_expected']:,}"
)


st.divider()


# ---------------------------------------------------------------------------
# OPTIMIZE - repair vs replace
# ---------------------------------------------------------------------------

st.header("Repair vs replace")

rvr = repair_vs_replace(
    repair_cost if repair_cost is not None else config.DEFAULT_REPAIR_COST,
    replacement_cost,
)
rm = replacement_meta.get("metrics", {})

if replacement_ok:
    st.caption(
        f"Replacement cost is a **learned estimate** - a Random Forest price "
        f"model trained on {replacement_meta['report']['rows_clean']:,} real "
        f"equipment listings (held-out R² {rm.get('r2', '-')}, "
        f"~{rm.get('mape', '-')}% typical error). The listings are heavy "
        f"equipment priced in USD and converted at a flat rate, so read the "
        f"{CUR} figure as indicative for a milling machine, not a quote."
    )
else:
    st.caption(
        "Price model unavailable - using the config.py fallback replacement "
        "cost."
    )

rc1, rc2, rc3 = st.columns(3)
rc1.metric("Repair cost (expected)", f"{CUR} {rvr['repair_cost']:,}")
rc2.metric(
    f"New machine (≈{int(rvr['fraction'] * 100)}% = replace line)",
    f"{CUR} {rvr['replacement_cost']:,}",
)
rc3.metric("Repair as share of new", f"{rvr['ratio'] * 100:.0f}%")

if rvr["verdict"] == "Replace":
    st.error(
        f"**Replace** - the repair ({CUR} {rvr['repair_cost']:,}) is above "
        f"{int(rvr['fraction'] * 100)}% of a new machine "
        f"({CUR} {rvr['threshold']:,}). Fixing it is throwing good money after bad."
    )
else:
    st.success(
        f"**Repair** - the repair ({CUR} {rvr['repair_cost']:,}) is well under "
        f"the {CUR} {rvr['threshold']:,} replace line "
        f"({int(rvr['fraction'] * 100)}% of a new machine). Worth keeping."
    )


st.divider()


# ---------------------------------------------------------------------------
# Where this reading sits in the dataset
# ---------------------------------------------------------------------------

st.header("This reading vs the dataset")
st.caption("Percentile = how many of the 10,000 readings sit below this value.")

context_rows = []
for sensor in config.SENSOR_COLUMNS:
    name, unit = config.SENSOR_DISPLAY[sensor]
    value = float(latest[sensor])
    percentile = float((df[sensor] < value).mean() * 100)
    context_rows.append({
        "sensor": f"{name} ({unit})",
        "this reading": round(value, 1),
        "dataset mean": round(float(df[sensor].mean()), 1),
        "percentile": f"{percentile:.0f}%",
    })

st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)

st.subheader("Nearby held-out readings")
window = test_df.iloc[max(0, row_idx - 4): row_idx + 5]
st.dataframe(window, use_container_width=True)


st.divider()


# ---------------------------------------------------------------------------
# Failure modes by frequency (whole dataset)
# ---------------------------------------------------------------------------

st.header("Failure modes by frequency")
st.caption(
    "Of the failures in all 10,000 readings, how often each mode occurs. "
    "This is counted straight from the dataset's 'Failure Type' column - it "
    "tells you which problems to prepare for first."
)

failed = df[df[config.TARGET_COLUMN] == 1]

# A handful of failure rows (~9) carry "No Failure" in the mode column - a
# known labelling quirk of AI4I. Drop them from the mode breakdown.
named = failed[failed[config.FAILURE_TYPE_COLUMN] != "No Failure"]
mode_counts = named[config.FAILURE_TYPE_COLUMN].value_counts()
mode_share = (mode_counts / mode_counts.sum() * 100).round(1)

# horizontal bar chart, most common at the top
st.bar_chart(mode_share.sort_values(), horizontal=True)

mode_table = pd.DataFrame({
    "Failure mode": mode_counts.index,
    "Count": mode_counts.values,
    "Share of named failures": [f"{pct}%" for pct in mode_share.values],
})
st.dataframe(mode_table, use_container_width=True, hide_index=True)
st.caption(
    f"{int(mode_counts.sum())} failures with a named mode, out of "
    f"{len(failed)} total failure rows in {len(df):,} readings "
    f"(the other {len(failed) - int(mode_counts.sum())} are labelled "
    "'No Failure' in the mode column - an AI4I quirk)."
)


st.divider()


# ---------------------------------------------------------------------------
# Per product-type (L / M / H) summary
# ---------------------------------------------------------------------------

st.header("Summary by product type")
st.caption(
    "The AI4I machine runs three quality variants. L is the low-grade / "
    "cheap variant, H the high-grade one. Their failure behaviour differs - "
    "which is why the model now uses Type as a feature."
)

type_rows = []
for variant in config.TYPE_VALUES:
    sub = df[df[config.TYPE_COLUMN] == variant]
    sub_failed = sub[sub[config.TARGET_COLUMN] == 1]
    n_rows = len(sub)
    n_fail = len(sub_failed)
    sub_named = sub_failed[sub_failed[config.FAILURE_TYPE_COLUMN] != "No Failure"]
    if len(sub_named):
        top_mode = sub_named[config.FAILURE_TYPE_COLUMN].value_counts().index[0]
    else:
        top_mode = "-"
    type_rows.append({
        "Type": variant,
        "Readings": f"{n_rows:,}",
        "Share of fleet": f"{n_rows / len(df) * 100:.0f}%",
        "Failures": n_fail,
        "Failure rate": f"{n_fail / n_rows * 100:.2f}%",
        "Most common mode": top_mode,
    })

st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)

for row in type_rows:
    st.write(
        f"- **Type {row['Type']}** - {row['Readings']} readings "
        f"({row['Share of fleet']} of the fleet), {row['Failures']} failures "
        f"({row['Failure rate']}); most common mode: {row['Most common mode']}."
    )
