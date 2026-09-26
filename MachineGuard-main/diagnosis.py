# diagnosis.py
# ---------------------------------------------------------------------------
# Rule-based diagnosis: a first guess at WHICH failure mode a reading looks
# like, using the AI4I 2020 dataset's OWN documented failure criteria.
#
# It is NOT a trained model, and it does NOT look at the true "Failure Type"
# column (that is a target - reading it would be cheating). It works purely
# from the five sensor values, checking the same physics the dataset used.
# Every result is phrased as a POSSIBLE cause to guide a human inspection.
# ---------------------------------------------------------------------------

import math

import config


def _physics(latest):
    """Derive the quantities the documented failure criteria are defined on."""
    air = latest["Air temperature [K]"]
    proc = latest["Process temperature [K]"]
    rpm = latest["Rotational speed [rpm]"]
    torque = latest["Torque [Nm]"]
    wear = latest["Tool wear [min]"]

    return {
        "temp_diff": proc - air,                     # K - heat the process can shed
        "rpm": rpm,
        "torque": torque,
        "wear": wear,
        # mechanical power = torque x angular velocity (rad/s)
        "power_w": torque * rpm * 2.0 * math.pi / 60.0,
        # overstrain index used by the dataset
        "overstrain": wear * torque,                  # min*Nm
    }


def diagnose(latest):
    """
    Check the reading against each documented failure mode.

    Returns a dict:
      headline  -> the single most likely mode (a string)
      findings  -> list of {fault, reasons, strength}, strongest first
      symptoms  -> the derived quantities, as readable strings
    """
    p = _physics(latest)

    # The product quality variant (L/M/H) sets the overstrain limit.
    product_type = str(latest[config.TYPE_COLUMN])
    osf_limit = config.OSF_TOOLWEAR_TORQUE_LIMIT.get(product_type, 11000)

    findings = []

    # --- Heat Dissipation Failure -----------------------------------
    # The process can't shed heat: the air/process gap is small AND the
    # spindle turns slowly (little forced cooling). BOTH must hold.
    if (p["temp_diff"] < config.HDF_TEMP_DIFF_K
            and p["rpm"] < config.HDF_ROT_SPEED_RPM):
        findings.append({
            "fault": "Possible Heat Dissipation Failure",
            "reasons": [
                f"process-to-air temperature gap is only {p['temp_diff']:.1f} K "
                f"(needs to stay above {config.HDF_TEMP_DIFF_K} K to shed heat)",
                f"rotational speed is {p['rpm']:.0f} rpm "
                f"(below {config.HDF_ROT_SPEED_RPM} rpm - little forced cooling)",
            ],
        })

    # --- Power Failure --------------------------------------------
    # Mechanical power outside the safe band.
    if p["power_w"] < config.PWF_POWER_MIN_W:
        findings.append({
            "fault": "Possible Power Failure (underpower)",
            "reasons": [
                f"mechanical power is {p['power_w']:.0f} W, below the "
                f"{config.PWF_POWER_MIN_W} W minimum "
                f"(torque {p['torque']:.1f} Nm at {p['rpm']:.0f} rpm)",
            ],
        })
    elif p["power_w"] > config.PWF_POWER_MAX_W:
        findings.append({
            "fault": "Possible Power Failure (overpower)",
            "reasons": [
                f"mechanical power is {p['power_w']:.0f} W, above the "
                f"{config.PWF_POWER_MAX_W} W maximum "
                f"(torque {p['torque']:.1f} Nm at {p['rpm']:.0f} rpm)",
            ],
        })

    # --- Overstrain Failure ----------------------------------
    # Tool wear x torque past the limit for this product variant.
    if p["overstrain"] > osf_limit:
        findings.append({
            "fault": "Possible Overstrain Failure",
            "reasons": [
                f"tool wear x torque = {p['overstrain']:.0f} min*Nm, past the "
                f"{osf_limit} limit for a type-{product_type} product "
                f"(wear {p['wear']:.0f} min, torque {p['torque']:.1f} Nm)",
            ],
        })

    # --- Tool Wear Failure --------------------------------
    # The tool is in the window where it tends to fail / be replaced.
    if config.TWF_TOOLWEAR_MIN <= p["wear"] <= config.TWF_TOOLWEAR_MAX:
        findings.append({
            "fault": "Possible Tool Wear Failure",
            "reasons": [
                f"tool wear is {p['wear']:.0f} min, inside the "
                f"{config.TWF_TOOLWEAR_MIN}-{config.TWF_TOOLWEAR_MAX} min window "
                f"where the tool tends to fail",
            ],
        })

    # strength = how many conditions back the finding; strongest first.
    for finding in findings:
        finding["strength"] = len(finding["reasons"])
    findings.sort(key=lambda f: f["strength"], reverse=True)

    # --- Fallback ---------------------------------------
    if not findings:
        findings.append({
            "fault": "No Specific Fault Identified",
            "reasons": ["no documented failure condition is met for this reading"],
            "strength": 0,
        })

    symptoms = [
        f"temp gap {p['temp_diff']:.1f} K",
        f"speed {p['rpm']:.0f} rpm",
        f"torque {p['torque']:.1f} Nm",
        f"power {p['power_w']:.0f} W",
        f"tool wear {p['wear']:.0f} min",
        f"wear x torque {p['overstrain']:.0f} min*Nm "
        f"(type {product_type} limit {osf_limit})",
    ]

    return {
        "headline": findings[0]["fault"],
        "findings": findings,
        "symptoms": symptoms,
    }
