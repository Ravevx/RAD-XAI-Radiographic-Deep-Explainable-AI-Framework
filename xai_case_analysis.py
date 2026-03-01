
import os
import pandas as pd
import numpy as np

CSV_PATH   = "models/xai_quality_full.csv"
OUTPUT_CSV = "models/xai_case_study.csv"
THRESHOLD  = 0.5

df = pd.read_csv(CSV_PATH)

#Derive prediction from prob 
df["Pred_Label"] = (df["Pred_Prob_Pneumonia"] >= THRESHOLD).astype(int)
df["Correct"]    = (df["Pred_Label"] == df["True_Label"]).astype(int)

# Define 4 clinical scenarios
scenarios = {
    "TP — Pneumonia predicted Pneumonia": df[
        (df["True_Label"] == 1) & (df["Pred_Label"] == 1)
    ],
    "TN — Normal predicted Normal": df[
        (df["True_Label"] == 0) & (df["Pred_Label"] == 0)
    ],
    "FN — Pneumonia predicted Normal (MISSED)": df[
        (df["True_Label"] == 1) & (df["Pred_Label"] == 0)
    ],
    "FP — Normal predicted Pneumonia (FALSE ALARM)": df[
        (df["True_Label"] == 0) & (df["Pred_Label"] == 1)
    ],
}

print("=" * 65)
print("CASE COUNT PER SCENARIO (all models combined)")
print("=" * 65)
for name, subset in scenarios.items():
    unique_imgs = subset["Filename"].nunique()
    print(f"  {name}")
    print(f"    → {unique_imgs} unique images across all models")

#Pick best representative image per scenario 
def pick_best_case(subset, n=1):
    """
    From a scenario subset, find images where ALL 3 models
    fall in the same scenario, sorted by avg Faithfulness_Drop desc.
    Returns top n filenames.
    """
    # Count how many models per filename are in this scenario
    counts = subset.groupby("Filename")["Model"].count()
    unanimous = counts[counts == 3].index.tolist()

    if not unanimous:
        # Fall back to images with at least 2 models in scenario
        unanimous = counts[counts >= 2].index.tolist()
    if not unanimous:
        unanimous = counts.index.tolist()

    # Among unanimous, pick highest avg faithfulness
    faith_avg = (
        subset[subset["Filename"].isin(unanimous)]
        .groupby("Filename")["Faithfulness_Drop"]
        .mean()
        .sort_values(ascending=False)
    )
    return faith_avg.head(n).index.tolist()

print("\n" + "=" * 65)
print("SELECTED REPRESENTATIVE CASES")
print("=" * 65)

selected_rows = []
case_names = {}

for scenario_name, subset in scenarios.items():
    if len(subset) == 0:
        print(f"\n No cases found for: {scenario_name}")
        continue

    best = pick_best_case(subset, n=1)
    case_img = best[0]
    case_names[scenario_name] = case_img

    case_df = df[df["Filename"] == case_img].copy()
    case_df["Scenario"] = scenario_name

    print(f"\n {scenario_name}")
    print(f"     Image: {case_img}")
    print(f"     {'Model':<14} | {'Prob':>6} | {'Conc':>6} | {'Faith':>6} | {'Agr':>6} | Quadrant")
    print(f"     {'-'*65}")
    for _, row in case_df.iterrows():
        print(f"     {row['Model']:<14} | {row['Pred_Prob_Pneumonia']:>6.3f} | "
              f"{row['Concentration']:>6.3f} | {row['Faithfulness_Drop']:>6.3f} | "
              f"{row['GC_GCpp_Agreement']:>6.3f} | {row['Focus_Quadrant']}")

    selected_rows.append(case_df)

case_study_df = pd.concat(selected_rows, ignore_index=True)
col_order = [
    "Scenario", "Filename", "True_Name", "Model",
    "Pred_Prob_Pneumonia", "Pred_Name", "Correct",
    "Concentration", "Faithfulness_Drop", "GC_GCpp_Agreement",
    "Top_Row", "Top_Col", "Focus_Quadrant"
]
case_study_df = case_study_df[[c for c in col_order if c in case_study_df.columns]]
case_study_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

print("\n" + "=" * 65)
print("FULL DATASET SUMMARY — Mean XAI metrics per model")
print("=" * 65)
summary = (
    df.groupby(["Model", "True_Name"])[
        ["Faithfulness_Drop", "Concentration", "GC_GCpp_Agreement"]
    ]
    .mean()
    .round(3)
)
print(summary.to_string())

print("\n" + "=" * 65)
print("CORRECT vs WRONG — Does being right mean better heatmaps?")
print("=" * 65)
corr_summary = (
    df.groupby(["Model", "Correct"])[
        ["Faithfulness_Drop", "GC_GCpp_Agreement"]
    ]
    .mean()
    .round(3)
)
corr_summary.index = corr_summary.index.set_levels(
    ["Wrong ❌", "Correct ✅"], level=1
)
print(corr_summary.to_string())

print("\n" + "=" * 65)
print("UNTRUSTWORTHY PREDICTIONS")
print("Correct prediction BUT faithfulness<0.15 AND agreement<0.30")
print("= model got the answer right for the WRONG reason")
print("=" * 65)
untrustworthy = df[
    (df["Correct"] == 1) &
    (df["Faithfulness_Drop"] < 0.15) &
    (df["GC_GCpp_Agreement"] < 0.30)
]
print(f"\n  Total untrustworthy: {len(untrustworthy)} / {len(df[df['Correct']==1])} correct predictions")
print(f"\n  Per model:")
print(untrustworthy.groupby("Model").size().to_string())
print(f"\n  Per class:")
print(untrustworthy.groupby(["Model","True_Name"]).size().to_string())

print("\n" + "=" * 65)
print("FOCUS QUADRANT DISTRIBUTION (where does each model look?)")
print("=" * 65)
quad = (
    df.groupby(["Model", "Focus_Quadrant"])
    .size()
    .unstack(fill_value=0)
)
print(quad.to_string())
