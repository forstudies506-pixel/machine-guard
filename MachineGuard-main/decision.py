# decision.py
# ---------------------------------------------------------------------------
# Maintenance decision engine.
#
# Every other module produces ONE signal (a risk number, an anomaly flag,
# a likely fault, a status). This file combines all of them into ONE
# recommended next step for the operator.
#
# It is a transparent rule ladder, checked most-urgent-first. It is
# DECISION SUPPORT to prompt a human, not a safety guarantee.
# ---------------------------------------------------------------------------

import config


def decide(failure_risk, anomaly_result, diagnosis, status):
    """
    Combine every signal into one recommended action.

    Inputs:
      failure_risk    -> float 0.0-1.0 from the failure model
      anomaly_result  -> dict from anomaly.interpret_anomaly()
      diagnosis       -> dict from diagnosis.diagnose()
      status          -> "Healthy" / "Warning" / "Critical" (rule-based badge)

    Returns a dict:
      action   -> "Continue Operation" / "Reduce Load" /
                  "Schedule Maintenance" / "Immediate Inspection"
      urgency  -> "low" / "medium" / "high"  (for colour-coding)
      reasons  -> list of short strings explaining the call
    """
    is_anomaly = anomaly_result["is_anomaly"]
    has_named_fault = diagnosis["headline"] != "No Specific Fault Identified"

    reasons = []

    # --- 1. Immediate Inspection -----------------------------------------
    # The machine is already in a critical state, or the model puts the
    # failure risk very high. Either way: stop and look now.
    if status == "Critical" or failure_risk >= config.RISK_HIGH:
        if status == "Critical":
            reasons.append("machine status is Critical on the rule-based check")
        if failure_risk >= config.RISK_HIGH:
            reasons.append(
                f"failure risk is {failure_risk * 100:.0f}%, at or above the "
                f"{config.RISK_HIGH * 100:.0f}% inspection line"
            )
        return {"action": "Immediate Inspection", "urgency": "high", "reasons": reasons}

    # --- 2. Schedule Maintenance ---------------------------------------
    # Clear degradation: risk is elevated, OR the anomaly detector and the
    # diagnosis engine independently agree something is wrong.
    if failure_risk >= config.RISK_MEDIUM or (is_anomaly and has_named_fault):
        if failure_risk >= config.RISK_MEDIUM:
            reasons.append(
                f"failure risk is {failure_risk * 100:.0f}%, at or above the "
                f"{config.RISK_MEDIUM * 100:.0f}% scheduling line"
            )
        if is_anomaly and has_named_fault:
            reasons.append(
                "the anomaly detector flagged this reading and the diagnosis "
                f"engine points to '{diagnosis['headline']}'"
            )
        return {"action": "Schedule Maintenance", "urgency": "medium", "reasons": reasons}

    # --- 3. Reduce Load ------------------------------------------------
    # Early warning: something is off, but the failure risk is still low.
    # Easing the load buys time before maintenance.
    if status == "Warning" or is_anomaly:
        if status == "Warning":
            reasons.append("machine status is Warning on the rule-based check")
        if is_anomaly:
            reasons.append("the anomaly detector flagged this reading as unusual")
        return {"action": "Reduce Load", "urgency": "medium", "reasons": reasons}

    # --- 4. Continue Operation ---------------------------------------
    # Nothing tripped.
    return {
        "action": "Continue Operation",
        "urgency": "low",
        "reasons": ["all checks are within normal limits"],
    }
