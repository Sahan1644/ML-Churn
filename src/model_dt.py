from sklearn.tree import DecisionTreeClassifier

def train_dt(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    return dt

def predict_dt(dt, X):
    return dt.predict(X)
