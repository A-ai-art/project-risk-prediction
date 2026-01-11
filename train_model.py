import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
import os

# --------------------------------------
# 1. LOAD DATASET
# --------------------------------------
df = pd.read_csv(r"C:\Users\anjal\OneDrive\Documents\ml project\data\project_risk_raw_dataset.csv")

# --------------------------------------
# 2. SELECT IMPORTANT FEATURES
# --------------------------------------
features = [
    "Team_Size",
    "Project_Budget_USD",
    "Estimated_Timeline_Months",
    "Complexity_Score",
    "Stakeholder_Count",
    "Past_Similar_Projects",
    "Change_Request_Frequency"
]

X = df[features]
y = df["Risk_Level"]

# --------------------------------------
# 3. SCALE FEATURES
# --------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------
# 4. SPLIT DATA
# --------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --------------------------------------
# 5. TRAIN MODEL
# --------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(xtrain, ytrain)

# --------------------------------------
# 6. SAVE MODEL + SCALER
# --------------------------------------
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/model.joblib")

# Save scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
