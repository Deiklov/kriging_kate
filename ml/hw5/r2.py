import numpy as np


def r2(y_true, y_pred):
    err = np.sum((y_pred-y_true)**2)/len(y_pred)
    ans = 1-err/np.std(y_true)**2
    return ans
