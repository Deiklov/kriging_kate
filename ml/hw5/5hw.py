import numpy as np
np.random.seed(42)

class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_:np.array = None

    def fit(self, x: np.array, y: np.array):
        x=np.c_[ x, np.ones(x.shape[0]) ]
        self.coef_=np.linalg.lstsq(x,y)[0]
        

    def predict(self, x: np.array):
        res=np.zeros(x.shape[0])
        for idx,val in enumerate(x):
            res[idx]=np.dot(np.append(val,1),self.coef_)
        return res
