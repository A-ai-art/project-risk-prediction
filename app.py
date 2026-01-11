from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model_path = os.path.join("model", "model.joblib")
scaler_path = os.path.join("model", "scaler.pkl")

model = joblib.load(model_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Features order (must match train_model.py)
features = [
    "Team_Size",
    "Project_Budget_USD",
    "Estimated_Timeline_Months",
    "Complexity_Score",
    "Stakeholder_Count",
    "Past_Similar_Projects",
    "Change_Request_Frequency"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float
        arr = []
        for feature in features:
            value = float(request.form[feature])
            arr.append(value)

        arr = np.array([arr])  # shape (1, 7)
        arr_scaled = scaler.transform(arr)
        prediction = model.predict(arr_scaled)[0]

        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
