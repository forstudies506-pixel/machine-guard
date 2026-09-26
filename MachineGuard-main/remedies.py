# remedies.py
# ---------------------------------------------------------------------------
# Failure mode -> likely root causes, the fix for each, and what that fix
# roughly costs (parts + labour).
#
# This is a KNOWLEDGE BASE, not a model.
#   * "likelihood" - illustrative FMEA-style priors (Failure Mode and
#     Effects Analysis) for CNC / milling maintenance. NOT computed from
#     the AI4I dataset - it carries no root-cause information. A real
#     deployment would learn them from the plant's own maintenance logs.
#   * "cost" - a CONFIGURABLE DEMO figure (parts + labour), in the ballpark
#     of real Indian CNC / VMC shop quotes so the dashboard "bill" and the
#     repair-vs-replace call are meaningful. Still not an actual quote.
#
# The diagnosis engine names the MODE from sensor physics; this file turns
# that mode into a ranked checklist with a price tag on each line.
# ---------------------------------------------------------------------------


REMEDIES = {
    "Possible Heat Dissipation Failure": [
        {"cause": "Coolant flow blocked or coolant level low", "likelihood": 35,
         "fix": "Check the coolant pump, filters and nozzle aim; top up coolant",
         "parts": "Coolant pump service + filter kit + coolant", "cost": 28000},
        {"cause": "Clogged heat exchanger / dirty cooling fins", "likelihood": 25,
         "fix": "Clean the radiator and fins; confirm the cooling fan runs",
         "parts": "Chiller / heat-exchanger service + cooling fan", "cost": 45000},
        {"cause": "Spindle speed too low for the cut (weak forced cooling)", "likelihood": 20,
         "fix": "Raise spindle speed or reduce depth of cut",
         "parts": "CAM re-programming (engineer time), no parts", "cost": 6000},
        {"cause": "Ambient / enclosure temperature too high", "likelihood": 15,
         "fix": "Improve shop ventilation; check the enclosure A/C",
         "parts": "Enclosure A/C service / panel cooler", "cost": 32000},
        {"cause": "Faulty temperature sensor / degraded thermal contact", "likelihood": 5,
         "fix": "Verify the sensor against a reference and re-seat it",
         "parts": "PT100 temperature sensor + labour", "cost": 9000},
    ],
    "Possible Power Failure (underpower)": [
        {"cause": "Spindle drive or motor fault (loss of torque)", "likelihood": 40,
         "fix": "Read the VFD fault codes; inspect motor windings and brushes",
         "parts": "Spindle drive / VFD board + commissioning", "cost": 165000},
        {"cause": "Slipping or broken drive belt / coupling", "likelihood": 25,
         "fix": "Check belt tension and coupling grub screws",
         "parts": "Drive belt + coupling insert + labour", "cost": 9000},
        {"cause": "Tool broken or disengaged mid-cut", "likelihood": 15,
         "fix": "Inspect the tool, re-seat it in the collet, check the length offset",
         "parts": "Replacement end mill + re-setup", "cost": 6000},
        {"cause": "Incoming supply voltage sag or phase loss", "likelihood": 15,
         "fix": "Check the mains supply, main contactor and phase monitor",
         "parts": "Electrician call-out + phase monitor + contactor", "cost": 14000},
        {"cause": "Encoder / feedback fault under-driving the spindle", "likelihood": 5,
         "fix": "Check encoder wiring; run the drive's self-diagnostics",
         "parts": "Spindle encoder + alignment + labour", "cost": 70000},
    ],
    "Possible Power Failure (overpower)": [
        {"cause": "Cut too aggressive (feed, depth or width too high)", "likelihood": 40,
         "fix": "Reduce the feed rate or the depth of cut",
         "parts": "CAM feed/speed re-programming (engineer time)", "cost": 6000},
        {"cause": "Dull or chipped tool forcing higher torque", "likelihood": 25,
         "fix": "Replace the tool; inspect the cutting edge",
         "parts": "Replacement cutting tool", "cost": 4500},
        {"cause": "Workpiece harder than the program assumes", "likelihood": 15,
         "fix": "Verify the stock material; re-tune speeds and feeds",
         "parts": "Material re-verification + re-tune", "cost": 5000},
        {"cause": "Poor lubrication raising friction", "likelihood": 12,
         "fix": "Check the way lube and spindle lubrication",
         "parts": "Way lube + spindle lube system service", "cost": 18000},
        {"cause": "Workpiece or fixture shifted, tool binding", "likelihood": 8,
         "fix": "Re-check the workholding and alignment",
         "parts": "Re-fixture + alignment check", "cost": 7000},
    ],
    "Possible Overstrain Failure": [
        {"cause": "Tool worn past its service life", "likelihood": 40,
         "fix": "Replace the tool; shorten the tool-change interval",
         "parts": "Replacement tool + tighter change interval", "cost": 5000},
        {"cause": "Feed rate too high for a worn tool", "likelihood": 25,
         "fix": "Lower the feed; enable a wear-compensated feed override",
         "parts": "Control feed-override setup (engineer time)", "cost": 4000},
        {"cause": "Wrong tool grade or coating for the material", "likelihood": 15,
         "fix": "Switch to the correct insert grade / coating",
         "parts": "Correct insert grade set", "cost": 9000},
        {"cause": "Not enough coolant at the cutting edge", "likelihood": 12,
         "fix": "Re-aim the nozzle; raise the coolant pressure",
         "parts": "Nozzle kit + pump pressure service", "cost": 15000},
        {"cause": "Excessive or uneven stock on the raw part", "likelihood": 8,
         "fix": "Add a roughing pass; inspect the casting",
         "parts": "Add roughing pass (programming)", "cost": 5000},
    ],
    "Possible Tool Wear Failure": [
        {"cause": "Tool has reached end of life", "likelihood": 55,
         "fix": "Replace the tool on schedule; do not run past ~200 min of wear",
         "parts": "Replacement end mill / insert set", "cost": 5000},
        {"cause": "Tool-life counter not reset after the last change", "likelihood": 15,
         "fix": "Reset the counter; audit the tool-change log",
         "parts": "Process audit (engineer time), no parts", "cost": 4000},
        {"cause": "Abrasive material accelerating wear", "likelihood": 12,
         "fix": "Shorten the tool-change interval for this part number",
         "parts": "Tooling review + shorter interval", "cost": 4000},
        {"cause": "Coolant concentration too low", "likelihood": 10,
         "fix": "Check the coolant mix ratio with a refractometer",
         "parts": "Coolant concentrate + refractometer check", "cost": 4000},
        {"cause": "Built-up edge from an incorrect surface speed", "likelihood": 8,
         "fix": "Increase the surface speed slightly",
         "parts": "Surface-speed re-programming, no parts", "cost": 4000},
    ],
}


def get_remedies(diagnosis_headline):
    """
    Return the ranked (cause, likelihood, fix, parts, cost) list for a
    diagnosed failure mode, or an empty list if the mode has no known cause
    list (no fault identified, or a genuinely random failure).
    """
    return REMEDIES.get(diagnosis_headline, [])


def repair_bill(diagnosis_headline):
    """
    Turn the remedy list for a diagnosed mode into an itemised "bill".

    Returns a dict, or None if the mode has no remedy list:
      rows      the ranked remedies, each with its own "cost"
      total     cost if EVERY listed fix were carried out (worst case)
      expected  probability-weighted cost = sum(likelihood/100 * cost).
                This is the single number to plan around: the causes more
                likely to be the real problem pull the estimate their way.
    """
    rows = get_remedies(diagnosis_headline)
    if not rows:
        return None

    total = sum(r["cost"] for r in rows)
    expected = sum(r["likelihood"] / 100.0 * r["cost"] for r in rows)

    return {"rows": rows, "total": total, "expected": round(expected)}
