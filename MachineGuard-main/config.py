# config.py
# ---------------------------------------------------------------------------
# Central place for every constant in MachineGuard.
#
# MachineGuard now trains on the AI4I 2020 Predictive Maintenance dataset
# (UCI / Kaggle): 10,000 real process snapshots of a milling machine, each
# labelled by whether it failed and, if so, which failure mode.
#
# The failure criteria below are the dataset's DOCUMENTED physics - the
# exact rules its authors used - not assumptions we invented.
# Source: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
# ---------------------------------------------------------------------------


# ---- Dataset -----------------------------------------------------------

RANDOM_SEED = 42                              # fixes every split / model so runs repeat

DATASET_PATH = "predictive_maintenance.csv"   # sits next to the .py files


# The raw file has TWO target columns. We rename the binary one to "Failure"
# so the rest of the code keeps working, and keep the failure-mode column
# for the diagnosis engine. NEITHER target may ever be used as a feature -
# that would be label leakage.
RAW_TARGET_COLUMN = "Target"
TARGET_COLUMN = "Failure"
FAILURE_TYPE_COLUMN = "Failure Type"
TYPE_COLUMN = "Type"                          # product quality variant: L / M / H

# Pure identifiers with no predictive meaning - dropped on load.
ID_COLUMNS = ["UDI", "Product ID"]


# ---- Features the models learn from ----------------------------------
# One fixed order, imported everywhere, so columns can never get shuffled
# between training and prediction.

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Product quality variant (L / M / H). Used by the diagnosis engine (the
# Overstrain limit depends on it) and by the "Summary by product type"
# panel. It is NOT fed to the failure model: when tried as a one-hot
# feature the model rated it under 1% importance and recall dropped, so it
# stays out. The 5 sensors already carry the signal.
TYPE_VALUES = ["L", "M", "H"]

# Engineered features - the SAME physical quantities the documented failure
# modes are defined on (power for PWF, the temp gap for HDF, wear x torque
# for OSF). They are deterministic functions of the 5 raw sensors, so they
# add NO new information leakage - they just hand the model the combination
# it would otherwise have to discover. Adding them lifted test F1 from
# ~0.64 to ~0.81 (precision ~0.78, recall ~0.84).
DERIVED_FEATURE_COLUMNS = [
    "Mechanical power [W]",     # torque x angular velocity
    "Temp difference [K]",      # process temp - air temp
    "Overstrain [min Nm]",      # tool wear x torque
    "Wear-speed [min rpm]",     # tool wear x rotational speed
]

# The feature list the FAILURE model trains on: the 5 raw sensors PLUS the
# engineered physics features above. One place for the model / SHAP /
# importance code to look.
MODEL_FEATURE_COLUMNS = SENSOR_COLUMNS + DERIVED_FEATURE_COLUMNS

# Short label + unit for the dashboard (the raw column names are long).
SENSOR_DISPLAY = {
    "Air temperature [K]":     ("Air temp", "K"),
    "Process temperature [K]": ("Process temp", "K"),
    "Rotational speed [rpm]":  ("Rotational speed", "rpm"),
    "Torque [Nm]":             ("Torque", "Nm"),
    "Tool wear [min]":         ("Tool wear", "min"),
    "Mechanical power [W]":    ("Mech. power", "W"),
    "Temp difference [K]":     ("Temp gap", "K"),
    "Overstrain [min Nm]":     ("Overstrain", "min Nm"),
    "Wear-speed [min rpm]":    ("Wear x speed", "min rpm"),
}


# ---- Typical (healthy) values --------------------------------------
# Approximate mean and standard deviation of each sensor across the
# NON-failed rows, from the dataset's summary statistics. Used by the
# "what's driving this reading's risk" heuristic to judge how unusual a
# value is.

SENSOR_BASELINES = {
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1540.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 108.0,
}

SENSOR_NOISE = {                              # standard deviation ("spread")
    "Air temperature [K]": 2.0,
    "Process temperature [K]": 1.5,
    "Rotational speed [rpm]": 179.0,
    "Torque [Nm]": 10.0,
    "Tool wear [min]": 63.0,
}


# ---- Documented failure-mode physics (AI4I 2020) ------------------
# These are the ACTUAL rules the dataset's authors used to decide a
# failure. Our diagnosis engine checks the same conditions to name a
# likely mode.

# Heat Dissipation Failure (HDF): the process cannot shed heat fast enough.
HDF_TEMP_DIFF_K = 8.6          # (process temp - air temp) below this, AND
HDF_ROT_SPEED_RPM = 1380      # rotational speed below this

# Power Failure (PWF): mechanical power = torque x angular velocity.
# The process fails outside a safe power band.
PWF_POWER_MIN_W = 3500
PWF_POWER_MAX_W = 9000

# Overstrain Failure (OSF): tool wear x torque past a limit that depends
# on the product quality variant.
OSF_TOOLWEAR_TORQUE_LIMIT = {"L": 11000, "M": 12000, "H": 13000}   # min*Nm

# Tool Wear Failure (TWF): the tool fails / is replaced somewhere in this
# tool-wear window.
TWF_TOOLWEAR_MIN = 200        # min
TWF_TOOLWEAR_MAX = 240       # min


# ---- Model / risk settings --------------------------------------

# Three-way split: 60% train / 20% validation / 20% test.
#   train       the model learns from these rows only
#   validation  used to check the model while building it (tuning, threshold
#               choice) WITHOUT touching the test set
#   test        the final untouched hold-out - scored once, and the only
#               rows the dashboard lets you inspect
# Both fractions are of the WHOLE dataset. The split is stratified, so the
# ~3.4% failure rate is preserved in all three parts.
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# The failure model outputs a probability 0.0-1.0. These cut points turn it
# into Low / Medium / High, shared by the dashboard badge and the decision
# engine so their labels always line up.
RISK_MEDIUM = 0.40
RISK_HIGH = 0.70

# For the per-reading "what's driving risk" heuristic: a sensor must be at
# least this many standard deviations from its healthy mean before it
# counts as a contributor.
CONTRIBUTION_MIN_SIGMAS = 1.5


# ---- Anomaly detector -------------------------------------------
# The Isolation Forest trains only on non-failed rows. contamination is
# how much of that training data it should still treat as borderline
# unusual; a small value keeps normal readings quiet.
ANOMALY_CONTAMINATION = 0.02


# ---- Dashboard status badge -----------------------------------
# A quick rule-based Healthy / Warning / Critical flag, independent of the
# ML model. Built from the documented physics above so it stays defensible.
WARNING_TEMP_DIFF_K = 9.5     # heat gap getting tight (HDF territory is 8.6 K)
WARNING_TOOLWEAR_MIN = 190   # approaching the tool-wear failure window


# ---- Cost assumptions (DEMO figures) --------------------------------
# Every rupee value below is a CONFIGURABLE PLANNING ASSUMPTION for the
# demo - not a real quote and not learned from the dataset. A real site
# would replace these with its own shop rates. They let the OPTIMIZE step
# put a number on "fix it now" vs "run it to failure".

CURRENCY = "Rs"                       # label shown in front of amounts

DOWNTIME_COST_PER_HOUR = 5000        # lost production for every hour the machine is stopped
PLANNED_DOWNTIME_HOURS = 2          # length of a scheduled maintenance stop
FAILURE_DOWNTIME_HOURS = 8         # length of an unplanned breakdown (waiting on parts + techs)
EMERGENCY_REPAIR_MULTIPLIER = 2.0   # a rush repair costs this many times the planned repair
DEFAULT_REPAIR_COST = 20000        # generic minor repair + inspection, when no mode is diagnosed


# ---- Repair vs replace (replacement.py) ----------------------------
# The OPTIMIZE step also asks: is this machine still worth fixing? The
# rule of thumb: replace once a single repair costs more than a set share
# of a NEW equivalent machine.
#
# We don't have a milling-machine price list, so we LEARN a price model
# from data.csv - ~1,700 real used-equipment listings. That data is
# heavy-equipment resale in USD, not milling machines: it proves the
# method and gives a demo figure, on the same footing as the rupee
# assumptions above. See replacement.py for the honesty notes.

EQUIPMENT_PRICE_PATH = "data.csv"     # the second dataset, sits next to the .py files
REPLACEMENT_REF_YEAR = 2020          # newest listing year - used to turn "year" into "age"
USD_TO_RS = 83                       # demo FX rate to keep every amount in one currency

REPLACE_COST_FRACTION = 0.5          # repair past this share of a new machine -> replace

# The stand-in "our machine" - the most common category / manufacturer /
# region codes in data.csv. Opaque integers; this is a chosen placeholder.
REPLACEMENT_PROFILE = {"category": 24, "manufacturer": 30, "region": 9}

# Fallback if the price model can't be trained (missing file, etc.).
DEFAULT_REPLACEMENT_COST = 4_600_000  # Rs (~ USD 56k median listing x USD_TO_RS)
