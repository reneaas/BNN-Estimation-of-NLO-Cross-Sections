import numpy as np



def r2_score(y_true, y_pred):
    residual = y_true - y_pred
    y_mean = np.mean(y_true, axis=0)
    r2 = 1 - np.sum(residual ** 2, axis=0) / np.sum((y_true - y_mean) ** 2, axis=0)
    return r2