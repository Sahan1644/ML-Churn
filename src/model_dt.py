# src/model_dt.py
from sklearn.tree import DecisionTreeClassifier
import pickle

def train_dt(X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    with open("src/dt_model.pkl", "wb") as f:
        pickle.dump(dt, f)
    return dt
