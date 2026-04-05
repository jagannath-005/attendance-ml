import numpy as np
import pandas as pd
import joblib

# ─── LOAD MODELS ONCE AT STARTUP ────────────────────────────────────────────

# Load once when Flask starts — not on every request
# This keeps the API fast
rf_model       = joblib.load("models/cancellation_model.pkl")
risk_model     = joblib.load("models/risk_model.pkl")
risk_scaler    = joblib.load("models/risk_scaler.pkl")
cluster_model  = joblib.load("models/cluster_model.pkl")
cluster_scaler = joblib.load("models/cluster_scaler.pkl")

# ─── FUNCTION 1: CANCELLATION PREDICTION ────────────────────────────────────

def predict_cancellations(upcoming_classes):
    """
    Input: list of dicts, each with keys:
        name, day, subject, faculty, slot, week, days_to_exam
    
    Output: same list with cancel_probability added to each dict
    
    Example input:
        [{"name": "DBS Monday", "day": 0, "subject": 0,
          "faculty": 1, "slot": 0, "week": 10, "days_to_exam": 30}]
    
    Example output:
        [{"name": "DBS Monday", ..., "cancel_probability": 0.40,
          "likely_cancelled": True}]
    """
    results = []
    for cls in upcoming_classes:
        features = pd.DataFrame([[
            cls["day"], cls["subject"], cls["faculty"],
            cls["slot"], cls["week"],   cls["days_to_exam"]
        ]], columns=["day_of_week", "subject", "faculty",
                     "time_slot",  "week",    "days_to_exam"])

        cancel_prob = round(rf_model.predict_proba(features)[0][0], 2)

        results.append({
            "name":               cls["name"],
            "cancel_probability": float(cancel_prob),
            "likely_cancelled":   bool(cancel_prob > 0.30)
        })
    return results


# ─── FUNCTION 2: BUNK PLAN ───────────────────────────────────────────────────

def compute_bunk_plan(subjects, min_threshold=75):
    """
    Input: list of dicts, each with keys:
        name, attended, conducted, planned_remaining
    
    Output: list of dicts with bunk plan per subject

    Example input:
        [{"name": "DBS", "attended": 18,
          "conducted": 24, "planned_remaining": 8}]

    Example output:
        [{"subject": "DBS", "current_pct": 75.0,
          "safe_bunks": 2, "must_still_attend": 6}]
    """
    results = []
    for s in subjects:
        attended  = s["attended"]
        conducted = s["conducted"]
        remaining = s["planned_remaining"]

        current_pct      = (attended / conducted) * 100
        total_at_end     = conducted + remaining
        must_attend_total = (min_threshold / 100) * total_at_end
        still_need       = max(0, must_attend_total - attended)
        safe_bunks       = max(0, int(remaining - still_need))

        results.append({
            "subject":          s["name"],
            "current_pct":      round(current_pct, 1),
            "safe_bunks":       safe_bunks,
            "must_still_attend": int(still_need)
        })
    return results


# ─── FUNCTION 3: RISK SCORING ────────────────────────────────────────────────

def predict_risk(subjects, min_threshold=75):
    """
    Input: list of dicts with name, attended, conducted
    Output: list of dicts with risk label per subject

    Example output:
        [{"subject": "DBS", "current_pct": 75.0,
          "risk_label": "Warning", "risk_code": 1}]
    """
    risk_labels = {0: "Safe", 1: "Warning", 2: "Critical"}
    results = []

    for s in subjects:
        current_pct      = (s["attended"] / s["conducted"]) * 100
        subjects_at_risk = 1 if current_pct < min_threshold else 0

        features = pd.DataFrame(
            [[current_pct, subjects_at_risk]],
            columns=["avg_pct", "subjects_at_risk"]
        )
        features_scaled = risk_scaler.transform(features)
        risk_code       = int(risk_model.predict(features_scaled)[0])

        results.append({
            "subject":     s["name"],
            "current_pct": round(current_pct, 1),
            "risk_label":  risk_labels[risk_code],
            "risk_code":   risk_code   # 0=Safe, 1=Warning, 2=Critical
        })
    return results


# ─── FUNCTION 4: STUDENT PROFILE ─────────────────────────────────────────────

def predict_student_profile(subjects):
    """
    Input: list of dicts with name, attended, conducted
    Output: dict with cluster profile and advice

    Example output:
        {"cluster": 0, "profile": "Borderline student",
         "advice": "You are close to the limit..."}
    """
    profiles = {
        0: ("Borderline student",  "You are close to the limit in several subjects. Be careful."),
        1: ("At-risk student",     "Your attendance is dangerously low. Attend all remaining classes."),
        2: ("Consistent attender", "Your attendance is healthy. You have room to be flexible."),
    }

    attendance_pcts = [
        (s["attended"] / s["conducted"]) * 100
        for s in subjects
    ]

    student_df = pd.DataFrame(
        [attendance_pcts],
        columns=["dbs_pct", "os_pct", "ai_pct", "po_pct", "daa_pct"]
    )
    student_scaled = cluster_scaler.transform(student_df)
    cluster        = int(cluster_model.predict(student_scaled)[0])

    profile_name, advice = profiles[cluster]
    return {
        "cluster": cluster,
        "profile": profile_name,
        "advice":  advice
    }


# ─── FUNCTION 5: WHAT-IF SIMULATION ──────────────────────────────────────────

def simulate_whatif(subjects, bunk_scenario, min_threshold=75):
    """
    Input:
        subjects: list of dicts with name, attended, conducted, planned_remaining
        bunk_scenario: dict of {subject_name: planned_bunks}
    Output: list of dicts with projected attendance per subject

    Example input:
        bunk_scenario = {"DBS": 2, "OS": 1, "AI": 0, "P&O": 1, "DAA": 2}

    Example output:
        [{"subject": "DBS", "planned_bunks": 2,
          "projected_pct": 75.0, "safe": True}]
    """
    results = []
    for s in subjects:
        planned_bunks       = bunk_scenario.get(s["name"], 0)
        new_total_conducted = s["conducted"] + s["planned_remaining"]
        new_attended        = s["attended"] + (s["planned_remaining"] - planned_bunks)
        projected_pct       = (new_attended / new_total_conducted) * 100

        results.append({
            "subject":       s["name"],
            "planned_bunks": planned_bunks,
            "projected_pct": round(projected_pct, 1),
            "safe":          projected_pct >= min_threshold
        })
    return results


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This block only runs when you run api_interface.py directly
    # When Person B imports this file into Flask, this block is skipped

    subjects = [
        {"name": "DBS",  "attended": 18, "conducted": 22, "planned_remaining": 8},
        {"name": "OS",   "attended": 16, "conducted": 20, "planned_remaining": 8},
        {"name": "AI",   "attended": 22, "conducted": 24, "planned_remaining": 8},
        {"name": "P&O",  "attended": 14, "conducted": 22, "planned_remaining": 10},
        {"name": "DAA",  "attended": 20, "conducted": 23, "planned_remaining": 8},
    ]

    upcoming = [
        {"name": "DBS Monday 9am",   "day": 0, "subject": 0, "faculty": 1,
         "slot": 0, "week": 10, "days_to_exam": 30},
        {"name": "DAA Friday 10am",  "day": 4, "subject": 4, "faculty": 2,
         "slot": 1, "week": 10, "days_to_exam": 30},
    ]

    print("Cancellations:", predict_cancellations(upcoming))
    print("Bunk plan:",     compute_bunk_plan(subjects))
    print("Risk:",          predict_risk(subjects))
    print("Profile:",       predict_student_profile(subjects))
    print("What-if:",       simulate_whatif(subjects, {"DBS": 2, "OS": 3}))