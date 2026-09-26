import pandas as pd
from sklearn.ensemble import IsolationForest

import config


def train_anomaly_model(df):
    """
    Train the anomaly detector (an Isolation Forest).

    It learns ONLY from the rows that did NOT fail - the dataset's picture
    of normal operation. Anything that looks unlike that is flagged as an
    anomaly, WITHOUT the detector ever being shown a failure.
    """

    # Every non-failed reading. This is "what a healthy machine looks like".
    healthy = df[df[config.TARGET_COLUMN] == 0]
    normal_data = healthy[config.SENSOR_COLUMNS]

    anomaly_model = IsolationForest(
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=42
    )

    anomaly_model.fit(normal_data)

    return anomaly_model


def detect_anomaly(anomaly_model, latest):
    """
    Check the latest reading against the healthy baseline.

    Returns:
      prediction     ->  1  means "looks normal"
                        -1  means "looks anomalous"
      anomaly_score  ->  higher = more normal, below 0 = anomalous
    """

    # One-row DataFrame with the same column names used during training,
    # so the sensors line up and scikit-learn does not warn.
    machine_input = pd.DataFrame(
        [[latest[column] for column in config.SENSOR_COLUMNS]],
        columns=config.SENSOR_COLUMNS
    )

    prediction = anomaly_model.predict(machine_input)[0]
    anomaly_score = anomaly_model.decision_function(machine_input)[0]

    return prediction, anomaly_score


def interpret_anomaly(prediction, anomaly_score):
    """
    Turn the raw Isolation Forest output into plain-language fields the
    dashboard can display directly.

    prediction     :  1 = normal, -1 = anomaly  (Isolation Forest convention)
    anomaly_score  :  decision_function value.
                      >= 0  -> resembles the healthy baseline
                      <  0  -> deviates from it; the lower, the more unusual
    """

    is_anomaly = bool(prediction == -1)

    if is_anomaly:
        label = "Anomaly Detected"
        explanation = (
            "The latest reading does not match the machine's healthy "
            "baseline behaviour."
        )
    else:
        label = "Normal"
        explanation = (
            "The latest reading is consistent with the machine's healthy "
            "baseline behaviour."
        )

    return {
        "is_anomaly": is_anomaly,
        "label": label,
        "explanation": explanation,
        "score": float(anomaly_score),
    }
