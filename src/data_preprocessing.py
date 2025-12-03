# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Drop customerID
    df = df.drop(['customerID'], axis=1)

    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Churn')

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Encode target
    target_le = LabelEncoder()
    df['Churn'] = target_le.fit_transform(df['Churn'])

    # Numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save preprocessing objects
    column_order = X_train.columns.tolist()
    with open("src/preprocess_objects.pkl", "wb") as f:
        pickle.dump({
            'le': le_dict,
            'target_le': target_le,
            'scaler': scaler,
            'numeric_cols': numeric_cols,
            'column_order': column_order
        }, f)

    return X_train, X_test, y_train, y_test
