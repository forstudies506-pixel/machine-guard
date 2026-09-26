# cost.py
# ---------------------------------------------------------------------------
# The OPTIMIZE step - put money on the decision.
#
# We compare two courses of action for the machine in front of us:
#
#   A) MAINTAIN NOW
#        planned repair cost  +  a short planned stop
#
#   B) RUN IT TO FAILURE
#        the EXPECTED cost of a breakdown
#        = failure_risk  x  (emergency repair  +  a long unplanned stop)
#      A breakdown only costs us if it actually happens, so we weight its
#      impact by the model's failure probability.
#
# "avoided" = B - A. Positive means acting now is the cheaper expected path.
#
# Every rupee figure comes from config.py and is a CONFIGURABLE DEMO
# ASSUMPTION - not a real quote.
#
# repair_vs_replace() adds the second money question: even if we fix it,
# is this machine still worth keeping? The replacement cost it needs is
# LEARNED by replacement.py from data.csv (see that file's honesty notes).
# ---------------------------------------------------------------------------

import config


def estimate_costs(failure_risk, repair_cost=None):
    """
    failure_risk : 0.0-1.0, the failure model's probability for this reading.
    repair_cost  : rupee cost to fix the diagnosed problem. Pass the
                   probability-weighted figure from remedies.repair_bill();
                   if None, config.DEFAULT_REPAIR_COST is used.

    Returns a dict:
      maintain_now             planned repair + planned downtime
      run_to_failure_expected  failure_risk x breakdown impact
      avoided                  run_to_failure_expected - maintain_now
      recommendation           "Maintain now" / "Running on is cheaper for now"
      breakdown                every line item, for display / teaching
    """
    if repair_cost is None:
        repair_cost = config.DEFAULT_REPAIR_COST

    # ---- Option A: maintain now ----
    planned_downtime = config.PLANNED_DOWNTIME_HOURS * config.DOWNTIME_COST_PER_HOUR
    maintain_now = repair_cost + planned_downtime

    # ---- Option B: run to failure ----
    emergency_repair = repair_cost * config.EMERGENCY_REPAIR_MULTIPLIER
    failure_downtime = config.FAILURE_DOWNTIME_HOURS * config.DOWNTIME_COST_PER_HOUR
    breakdown_impact = emergency_repair + failure_downtime

    # A breakdown only hurts if it happens - weight it by the risk.
    run_to_failure_expected = failure_risk * breakdown_impact

    avoided = run_to_failure_expected - maintain_now

    return {
        "maintain_now": round(maintain_now),
        "run_to_failure_expected": round(run_to_failure_expected),
        "avoided": round(avoided),
        "recommendation": (
            "Maintain now" if avoided > 0 else "Running on is cheaper for now"
        ),
        "breakdown": {
            "repair_cost": round(repair_cost),
            "planned_downtime": round(planned_downtime),
            "emergency_repair": round(emergency_repair),
            "failure_downtime": round(failure_downtime),
            "breakdown_impact": round(breakdown_impact),
            "failure_risk": float(failure_risk),
        },
    }


def repair_vs_replace(repair_cost, replacement_cost=None):
    """
    repair_cost      : rupee cost to fix the diagnosed problem (pass the
                       probability-weighted figure from repair_bill()).
    replacement_cost : rupee price of a NEW equivalent machine. Pass the
                       value from replacement.estimate_replacement_cost();
                       if None, config.DEFAULT_REPLACEMENT_COST is used.

    The fleet-management rule of thumb: once one repair costs more than
    config.REPLACE_COST_FRACTION of a new machine, replacing is the
    better call.

    Returns a dict:
      verdict          "Repair" / "Replace"
      repair_cost      echoed back, rounded
      replacement_cost echoed back, rounded
      threshold        REPLACE_COST_FRACTION x replacement_cost
      ratio            repair_cost / replacement_cost (0-1+)
      fraction         REPLACE_COST_FRACTION, for the UI to show the line
    """
    if replacement_cost is None:
        replacement_cost = config.DEFAULT_REPLACEMENT_COST

    threshold = config.REPLACE_COST_FRACTION * replacement_cost
    verdict = "Replace" if repair_cost > threshold else "Repair"

    return {
        "verdict": verdict,
        "repair_cost": round(repair_cost),
        "replacement_cost": round(replacement_cost),
        "threshold": round(threshold),
        "ratio": round(repair_cost / replacement_cost, 3) if replacement_cost else 0.0,
        "fraction": config.REPLACE_COST_FRACTION,
    }
