from flask import Flask, request, jsonify
import pandas as pd
import joblib
import uuid, time

app = Flask(__name__)

print("📂 Loading model artifacts...")
classifier = joblib.load("classifier.joblib")
preprocessor = joblib.load("preprocessor.joblib")
print("✅ Artifacts loaded!")

# ---- training columns ----
NUMERIC_COLS = [
    "loan_amnt", "term", "int_rate", "annual_inc", "emp_length", "fico_avg",
    "delinq_2yrs", "pub_rec", "inq_last_6mths", "open_acc", "total_acc",
    "credit_utilization", "dti_capped", "payment_burden", "long_term_loan",
    "thin_file", "risk_flag_count"
]
CATEGORICAL_COLS = ["grade", "purpose", "home_ownership"]
REQUIRED_COLS = NUMERIC_COLS + CATEGORICAL_COLS

MODEL_VERSION = "v1"


def decision_band(pd_score: float):
    if pd_score < 0.15:
        return "Low Risk", "APPROVE"
    elif pd_score < 0.30:
        return "Medium Risk", "MANUAL_REVIEW"
    return "High Risk", "REJECT"


def validate_payload(payload: dict):
    missing = [c for c in REQUIRED_COLS if c not in payload]

    bad_numeric = []
    for c in NUMERIC_COLS:
        if c in payload:
            try:
                float(payload[c])
            except Exception:
                bad_numeric.append(c)

    bad_cat = []
    for c in CATEGORICAL_COLS:
        if c in payload and (payload[c] is None or str(payload[c]).strip() == ""):
            bad_cat.append(c)

    return missing, bad_numeric, bad_cat


def build_row(payload: dict):
    """Build 1-row dict in exact expected column order + cast numerics."""
    row = {c: payload[c] for c in REQUIRED_COLS}
    for c in NUMERIC_COLS:
        row[c] = float(row[c])
    return row


def score(row_dict: dict) -> float:
    df = pd.DataFrame([row_dict])
    X = preprocessor.transform(df)
    return float(classifier.predict_proba(X)[0, 1])


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}, 200


@app.post("/validate")
def validate():
    payload = request.get_json(force=True, silent=True) or {}
    missing, bad_numeric, bad_cat = validate_payload(payload)

    return jsonify({
        "missing_fields": missing,
        "bad_numeric_fields": bad_numeric,
        "bad_categorical_fields": bad_cat,
        "required_fields": REQUIRED_COLS
    }), 200


@app.post("/predict")
def predict():
    request_id = str(uuid.uuid4())
    t0 = time.time()

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "invalid_json", "request_id": request_id}), 400

    missing, bad_numeric, bad_cat = validate_payload(payload)
    if missing or bad_numeric or bad_cat:
        return jsonify({
            "error": "invalid_input",
            "missing_fields": missing,
            "bad_numeric_fields": bad_numeric,
            "bad_categorical_fields": bad_cat,
            "request_id": request_id
        }), 400

    try:
        row = build_row(payload)
        pd_score = score(row)
    except Exception as e:
        return jsonify({
            "error": "prediction_failed",
            "details": str(e),
            "request_id": request_id
        }), 500

    risk, decision = decision_band(pd_score)
    latency_ms = int((time.time() - t0) * 1000)

    return jsonify({
        "pd": round(pd_score, 6),
        "risk_category": risk,
        "decision": decision,
        "latency_ms": latency_ms,
        "model_version": MODEL_VERSION,
        "request_id": request_id
    }), 200


@app.post("/sanity")
def sanity():
    """
    Quick behavior checks:
    - FICO up should reduce PD
    - FICO down should increase PD
    - DTI up should increase PD
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "invalid_json"}), 400

    missing, bad_numeric, bad_cat = validate_payload(payload)
    if missing or bad_numeric or bad_cat:
        return jsonify({
            "error": "invalid_input",
            "missing_fields": missing,
            "bad_numeric_fields": bad_numeric,
            "bad_categorical_fields": bad_cat
        }), 400

    base = build_row(payload)

    high_fico = dict(base); high_fico["fico_avg"] = 780.0
    low_fico  = dict(base); low_fico["fico_avg"]  = 600.0
    high_dti  = dict(base); high_dti["dti_capped"] = 30.0

    try:
        pd_base = score(base)
        pd_fico_780 = score(high_fico)
        pd_fico_600 = score(low_fico)
        pd_dti_30 = score(high_dti)
    except Exception as e:
        return jsonify({"error": "sanity_failed", "details": str(e)}), 500

    return jsonify({
        "pd_base": round(pd_base, 6),
        "pd_fico_780": round(pd_fico_780, 6),
        "pd_fico_600": round(pd_fico_600, 6),
        "pd_dti_30": round(pd_dti_30, 6),
        "expected_behavior": {
            "fico_780_should_be_lower_than_base": pd_fico_780 < pd_base,
            "fico_600_should_be_higher_than_base": pd_fico_600 > pd_base,
            "dti_30_should_be_higher_than_base": pd_dti_30 > pd_base
        }
    }), 200


if __name__ == "__main__":
    # Local dev only (Docker/AWS will use gunicorn)
    app.run(host="0.0.0.0", port=8080, debug=False)
