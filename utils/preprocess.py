import joblib

def load_scaler():
    return joblib.load("model/scaler.save")
