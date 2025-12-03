import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Drop customerID
    df = df.drop(['customerID'], axis=1)

    # Convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Churn')  # target

    le = {}
    for col in categorical_cols:
        le[col] = LabelEncoder()
        df[col] = le[col].fit_transform(df[col])

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    # Scale numeric columns
    numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, le, scaler, numeric_cols
