import numpy as np

def quantum_transform(X):
    return np.hstack([
        np.sin(X),
        np.cos(X),
        np.exp(-X)
    ])
