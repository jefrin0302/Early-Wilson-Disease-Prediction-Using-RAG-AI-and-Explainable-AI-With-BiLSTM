#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import os

ART_DIR = "wilson_artifacts"

# Load saved artifacts
preprocessor = joblib.load(os.path.join(ART_DIR, "preprocessor.joblib"))
svm = joblib.load(os.path.join(ART_DIR, "svm.joblib"))
logreg = joblib.load(os.path.join(ART_DIR, "logreg_base.joblib"))
meta = joblib.load(os.path.join(ART_DIR, "meta_model.joblib"))

print("Loaded saved models successfully.\n")

# -------------------------
# Prepare the sample DF
# -------------------------
data = [
    ["Daniel Perry",14,"Female",9.34,229.03,18.5,144.98,73.71,72.04,1.31,3.93,110.55,1.05,127.93,1,5.07,1,80.1,1,1,"West","Low","FALSE",25.21,1],
    ["Lauren Luna",21,"Female",9.14,200.23,16.15,104.19,25.89,67.51,0.22,4.10,89.18,1.10,105.97,1,6.18,0,96.84,1,1,"South","Medium","FALSE",33.51,1],
    ["Andrew Lewis",31,"Male",9.76,267.34,14.37,153.17,45.91,76.68,3.53,5.49,145.40,1.18,85.50,1,4.43,1,69.16,1,1,"West","High","TRUE",26.23,1],
    ["John Nash",44,"Male",13.68,101.04,12.81,72.49,59.51,35.47,1.29,3.76,118.66,1.14,75.28,0,2.81,1,86.72,0,0,"South","High","FALSE",18.70,0],
    ["Elizabeth Blevins",53,"Female",23.50,121.82,20.61,110.77,44.47,24.84,2.65,5.45,161.10,1.22,92.20,0,0.54,1,90.49,0,0,"East","Low","TRUE",37.20,0],
    ["Stephen Davis",59,"Female",22.85,145.98,10.81,120.36,50.11,48.07,1.38,3.95,121.90,1.27,59.52,0,2.34,1,85.77,1,0,"North","Medium","FALSE",28.16,0],
    ["Stephen Wallace",43,"Male",14.53,173.64,8.77,106.67,32.74,60.62,1.20,4.21,132.30,1.04,78.33,0,2.75,0,85.34,0,0,"West","High","FALSE",38.48,0]
]

columns = [
    "Name","Age","Sex","Ceruloplasmin Level","Copper in Blood Serum",
    "Free Copper in Blood Serum","Copper in Urine","ALT","AST",
    "Total Bilirubin","Albumin","Alkaline Phosphatase (ALP)",
    "Prothrombin Time / INR","Gamma-Glutamyl Transferase (GGT)",
    "Kayser-Fleischer Rings","Neurological Symptoms Score","Psychiatric Symptoms",
    "Cognitive Function Score","Family History","ATB7B Gene Mutation","Region",
    "Socioeconomic Status","Alcohol Use","BMI","Is_Wilson_Disease"
]

df = pd.DataFrame(data, columns=columns)

# For prediction, remove Name and Target
df_features = df.drop(columns=["Name","Is_Wilson_Disease"])

print("Input DataFrame:")
print(df_features)
print("\n")

# -------------------------
# RUN HYBRID MODEL
# -------------------------
# preprocess
X = preprocessor.transform(df_features)

# SVM probability
svm_p = svm.predict_proba(X)[:,1]

# Logistic regression probability
log_p = logreg.predict_proba(X)[:,1]

# Stack predictions
stacked = np.column_stack([svm_p, log_p])

# Final probability
final_prob = meta.predict_proba(stacked)[:,1]
final_pred = (final_prob >= 0.5).astype(int)

# -------------------------
# PRINT RESULTS
# -------------------------
print("===== PREDICTION RESULTS =====")
for i in range(len(df)):
    name = df.iloc[i]["Name"]
    label = final_pred[i]
    prob = final_prob[i]

    print(f"{i+1}. {name}")
    print(f"   Predicted: {'Positive' if label==1 else 'Negative'}")
    print(f"   Probability: {prob:.4f}")
    print(f"   Actual Label: {df.iloc[i]['Is_Wilson_Disease']}")
    print("-"*40)
