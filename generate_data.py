import numpy as np
import pandas as pd

# Fix the random seed so you get the same data every time you run this
# Without this, the data changes every run and your results become inconsistent
np.random.seed(42)

# How many class records to simulate
n = 600

# ─── FEATURES ──────────────────────────────────────────────────────────────

# Day of week: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
# np.random.choice picks randomly from the list, n times
day_of_week = np.random.choice([0, 1, 2, 3, 4], n)

# Subject encoded as numbers: 0=DBS, 1=OS, 2=AI, 3=P&O, 4=DAA 
subject = np.random.choice([0, 1, 2, 3, 4], n)

# Faculty encoded: 0, 1, 2, 3 — faculty 2 cancels more often (you'll see below)
faculty = np.random.choice([0, 1, 2, 3], n)

# Time slot: 0=9m, 1=10am, 2=11am, 3=12pm
time_slot = np.random.choice([0, 1, 2, 3], n)

# Which week of the semester (1 to 15)
week = np.random.randint(1, 16, n)

# How many days until the next exam
# randint(low, high) picks integers between low and high-1
days_to_exam = np.random.randint(0, 60, n)

# ─── CANCELLATION PROBABILITY ──────────────────────────────────────────────

# Base probability that any class gets cancelled
# 0.15 means 15% chance of cancellation by default
cancel_prob = np.full(n, 0.15)

# Friday classes cancel more — add 10% extra chance
cancel_prob += 0.10 * (day_of_week == 4)

# Faculty 2 is unreliable — add 12% extra
cancel_prob += 0.12 * (faculty == 2)

# Early morning (9am) classes cancel more — students + faculty skip
cancel_prob += 0.08 * (time_slot == 0)

# Near exams, faculty are more likely to show up — reduce cancellation
# days_to_exam < 14 means within 2 weeks of exam
cancel_prob -= 0.07 * (days_to_exam < 14)

# Clip keeps values between 0.05 and 0.60
# Without this, probabilities could go below 0 or above 1, which is invalid
cancel_prob = np.clip(cancel_prob, 0.05, 0.60)

# ─── SIMULATE WHETHER CLASS HAPPENED ───────────────────────────────────────

# np.random.binomial(1, p) flips a weighted coin for each class
# p here = probability class WAS held = 1 - cancel_prob
# Result: 1 = class happened, 0 = cancelled
class_held = np.random.binomial(1, 1 - cancel_prob)

# ─── BUILD THE DATAFRAME ────────────────────────────────────────────────────

# A DataFrame is a table. You build it from a dictionary where:
# key = column name, value = the array of values
df = pd.DataFrame({
    "day_of_week":    day_of_week,
    "subject":        subject,
    "faculty":        faculty,
    "time_slot":      time_slot,
    "week":           week,
    "days_to_exam":   days_to_exam,
    "class_held":     class_held
})

# ─── INSPECT THE DATA ───────────────────────────────────────────────────────

print("Shape:", df.shape)           # (rows, columns)
print("\nFirst 5 rows:")
print(df.head())                    # Shows top 5 rows

print("\nClass held distribution:")
print(df["class_held"].value_counts())  # How many 1s vs 0s

print("\nCancellation rate by day:")
print(df.groupby("day_of_week")["class_held"].mean().round(2))

# ─── SAVE TO CSV ────────────────────────────────────────────────────────────

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/class_history.csv", index=False)
print("\nSaved to data/class_history.csv")