import numpy as np
import pandas as pd
import joblib

# ─── LOAD ALL SAVED MODELS ──────────────────────────────────────────────────

# These were trained in train_models.py and saved as .pkl files
# Loading them here means we never retrain — just use what was learned
rf_model       = joblib.load("models/cancellation_model.pkl")
risk_model     = joblib.load("models/risk_model.pkl")
risk_scaler    = joblib.load("models/risk_scaler.pkl")
cluster_model  = joblib.load("models/cluster_model.pkl")
cluster_scaler = joblib.load("models/cluster_scaler.pkl")

print("All models loaded successfully.\n")

# ─── STUDENT INPUT DATA ─────────────────────────────────────────────────────

# This represents one student's current attendance situation
# In the real system, Person C's frontend will send this via the Flask API
# For now we hardcode it to test

# Subject names matching our encoding: 0=DBS, 1=OS, 2=AI, 3=P&O, 4=DAA
subjects = [
    {"name": "DBS",  "attended": 18, "conducted": 22, "planned_remaining": 8},
    {"name": "OS",   "attended": 16, "conducted": 20, "planned_remaining": 8},
    {"name": "AI",   "attended": 22, "conducted": 24, "planned_remaining": 8},
    {"name": "P&O",  "attended": 14, "conducted": 19, "planned_remaining": 10},
    {"name": "DAA",  "attended": 20, "conducted": 23, "planned_remaining": 8},
]

MIN_THRESHOLD = 75  # minimum attendance % required

# ─── PART 1: CANCELLATION PREDICTION ────────────────────────────────────────

print("=" * 55)
print("PART 1: Upcoming class cancellation probabilities")
print("=" * 55)

# Simulate this week's upcoming classes
# In real system this comes from the timetable input
upcoming_classes = [
    {"name": "DBS  Monday 9am",    "day": 0, "subject": 0, "faculty": 1, "slot": 0, "week": 10, "days_to_exam": 30},
    {"name": "OS   Monday 10am",   "day": 0, "subject": 1, "faculty": 0, "slot": 1, "week": 10, "days_to_exam": 30},
    {"name": "AI   Tuesday 11am",  "day": 1, "subject": 2, "faculty": 2, "slot": 2, "week": 10, "days_to_exam": 30},
    {"name": "P&O  Wednesday 9am", "day": 2, "subject": 3, "faculty": 1, "slot": 0, "week": 10, "days_to_exam": 30},
    {"name": "DAA  Friday 10am",   "day": 4, "subject": 4, "faculty": 2, "slot": 1, "week": 10, "days_to_exam": 30},
]

for cls in upcoming_classes:
    # Build the feature row exactly as the model was trained on
    features = pd.DataFrame([[
        cls["day"], cls["subject"], cls["faculty"],
        cls["slot"], cls["week"],   cls["days_to_exam"]
    ]], columns=["day_of_week", "subject", "faculty", "time_slot", "week", "days_to_exam"])

    # predict_proba returns [[prob_cancelled, prob_held]]
    # index [0][0] = probability of class being cancelled
    cancel_prob = rf_model.predict_proba(features)[0][0]

    # Flag classes likely to be cancelled (>30% chance)
    flag = "⚠ Likely cancelled" if cancel_prob > 0.30 else "  Likely held"
    print(f"{cls['name']:<25} Cancel prob: {cancel_prob:.0%}   {flag}")

# ─── PART 2: BUNK OPTIMIZATION ──────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 2: Safe bunk plan per subject")
print("=" * 55)

bunk_plan = []

for s in subjects:
    attended  = s["attended"]
    conducted = s["conducted"]
    remaining = s["planned_remaining"]

    # Current attendance percentage
    current_pct = (attended / conducted) * 100

    # Total classes at end of semester
    total_at_end = conducted + remaining

    # Minimum classes you must attend out of total
    must_attend_total = (MIN_THRESHOLD / 100) * total_at_end

    # How many more you still need to attend
    still_need = max(0, must_attend_total - attended)

    # Out of remaining classes, how many can you safely skip?
    safe_bunks = max(0, int(remaining - still_need))

    bunk_plan.append({
        "subject":     s["name"],
        "current_pct": round(current_pct, 1),
        "safe_bunks":  safe_bunks,
        "still_need":  int(still_need)
    })

    print(f"{s['name']:<6} Current: {current_pct:.1f}%  "
          f"Safe bunks: {safe_bunks}  "
          f"Must still attend: {int(still_need)}")

# ─── PART 3: RISK SCORING ────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3: Risk level per subject")
print("=" * 55)

risk_labels = {0: "Safe", 1: "Warning", 2: "Critical"}

for s in subjects:
    current_pct = (s["attended"] / s["conducted"]) * 100

    # Count subjects below threshold (here we check just this subject)
    subjects_at_risk = 1 if current_pct < MIN_THRESHOLD else 0

    # Build feature array — must match what risk_model was trained on
    features = pd.DataFrame([[current_pct, subjects_at_risk]],
                         columns=["avg_pct", "subjects_at_risk"])

    # Scale it — risk model needs scaled input
    features_scaled = risk_scaler.transform(features)

    # Predict risk label
    risk = risk_model.predict(features_scaled)[0]
    label = risk_labels[risk]

    status = "🔴" if risk == 2 else ("🟡" if risk == 1 else "🟢")
    print(f"{s['name']:<6} {current_pct:.1f}%   {status} {label}")

# ─── PART 4: STUDENT PROFILE (CLUSTERING) ───────────────────────────────────

print("\n" + "=" * 55)
print("PART 4: Student profile")
print("=" * 55)

# Build attendance % array for this student across all subjects
student_attendance = np.array([[
    (s["attended"] / s["conducted"]) * 100
    for s in subjects
]])

# Scale and predict cluster
student_df = pd.DataFrame(student_attendance,
                          columns=["dbs_pct", "os_pct", "ai_pct", "po_pct", "daa_pct"])
student_scaled = cluster_scaler.transform(student_df)
cluster = cluster_model.predict(student_scaled)[0]

profiles = {
    0: ("Borderline student",   "You are close to the limit in several subjects. Be careful."),
    1: ("At-risk student",      "Your attendance is dangerously low. Attend all remaining classes."),
    2: ("Consistent attender",  "Your attendance is healthy. You have room to be flexible."),
}

profile_name, advice = profiles[cluster]
print(f"Your profile: {profile_name}")
print(f"Advice: {advice}")

# ─── PART 5: WHAT-IF SIMULATION ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 5: What-if simulation")
print("=" * 55)
print("What happens if you bunk these many classes?\n")

# Define a bunk scenario to simulate
# Change these numbers to test different scenarios
bunk_scenario = {
    "DBS": 2,
    "OS":  3,
    "AI":  1,
    "P&O": 2,
    "DAA": 2,
}

for s in subjects:
    planned_bunks = bunk_scenario.get(s["name"], 0)
    attended      = s["attended"]
    conducted     = s["conducted"]
    remaining     = s["planned_remaining"]

    # If you bunk some of the remaining classes, your total conducted goes up
    # but your attended stays the same
    new_total_conducted = conducted + remaining
    new_attended        = attended + (remaining - planned_bunks)
    projected_pct       = (new_attended / new_total_conducted) * 100

    status = "SAFE" if projected_pct >= MIN_THRESHOLD else "UNSAFE"
    marker = "✓" if status == "SAFE" else "✗"

    print(f"{s['name']:<6} Bunk {planned_bunks} → Projected: {projected_pct:.1f}%  {marker} {status}")