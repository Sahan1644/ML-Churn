import pandas as pd

def save_new_customer(new_df, output_file="predicted_customers.csv"):
    try:
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        combined_df = new_df
    combined_df.to_csv(output_file, index=False)
