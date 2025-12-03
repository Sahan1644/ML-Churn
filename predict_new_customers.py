# predict_new_customers.py

import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# ------------------------------
# Load preprocessing objects
# ------------------------------
with open("src/preprocess_objects.pkl", "rb") as f:
    objs = pickle.load(f)

le_dict = objs['le']
target_le = objs['target_le']
scaler = objs['scaler']
numeric_cols = objs['numeric_cols']
column_order = objs['column_order']

# ------------------------------
# Load trained models
# ------------------------------
nn = load_model("src/nn_model.h5")
with open("src/dt_model.pkl", "rb") as f:
    dt = pickle.load(f)

# ------------------------------
# Get user input for new customer
# ------------------------------
print("\n--- Enter New Customer Details ---")

new_customer = {}

# Categorical columns
for col in le_dict.keys():
    options = list(le_dict[col].classes_)
    while True:
        val = input(f"{col} {options}: ")
        if val in options:
            break
        print(f"Invalid input. Choose from {options}")
    new_customer[col] = le_dict[col].transform([val])[0]

# Numeric columns
for col in numeric_cols + ['SeniorCitizen']:
    while True:
        try:
            val = float(input(f"{col}: "))
            new_customer[col] = val
            break
        except ValueError:
            print("Enter a valid number!")

# ------------------------------
# Prepare DataFrame for prediction
# ------------------------------
new_df = pd.DataFrame([new_customer])
new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
new_df = new_df.astype(np.float32)

# ------------------------------
# Predict using models
# ------------------------------
new_df['ChurnPrediction_NN'] = (nn.predict(new_df) > 0.5).astype(int)
X_new = new_df.drop(columns=['ChurnPrediction_NN'], errors='ignore')
X_new = X_new[column_order]  # Ensure correct feature order for DT
new_df['ChurnPrediction_DT'] = dt.predict(X_new)

# Map predictions back to 'Yes'/'No' for readability
new_df['ChurnPrediction_NN'] = target_le.inverse_transform(new_df['ChurnPrediction_NN'])
new_df['ChurnPrediction_DT'] = target_le.inverse_transform(new_df['ChurnPrediction_DT'])

# ------------------------------
# Save predictions to CSV
# ------------------------------
output_file = "predicted_customers.csv"
new_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

print("\nPrediction saved to '{}':".format(output_file))
print(new_df)
