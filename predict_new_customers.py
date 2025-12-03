import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from src.utils import save_new_customer

# -----------------------------
# Load models & preprocessing
# -----------------------------
nn = load_model("nn_model.h5")

with open("dt_model.pkl", "rb") as f:
    dt = pickle.load(f)

with open("preprocess_objects.pkl", "rb") as f:
    objs = pickle.load(f)

le = objs['le']
scaler = objs['scaler']
numeric_cols = objs['numeric_cols']

# -----------------------------
# User input
# -----------------------------
print("\n--- Enter New Customer Details ---")
new_customer = {}
columns_to_ask = list(le.keys()) + numeric_cols + ['SeniorCitizen']

for col in columns_to_ask:
    if col in numeric_cols + ['SeniorCitizen']:
        while True:
            try:
                new_customer[col] = float(input(f"{col}: "))
                break
            except ValueError:
                print("Enter a valid number!")
    else:
        options = list(le[col].classes_)
        val = input(f"{col} {options}: ")
        if val not in options:
            print(f"Invalid input. Using default: {options[0]}")
            val = options[0]
        new_customer[col] = le[col].transform([val])[0]

# -----------------------------
# Convert to DataFrame & scale
# -----------------------------
new_df = pd.DataFrame([new_customer])
new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
new_df = new_df.astype(np.float32)

# -----------------------------
# Predict churn
# -----------------------------
new_df['ChurnPrediction_NN'] = (nn.predict(new_df) > 0.5).astype(int)
new_df['ChurnPrediction_DT'] = dt.predict(new_df)

# -----------------------------
# Save prediction
# -----------------------------
save_new_customer(new_df)

print("\nPrediction saved! Here's the result:")
print(new_df)
