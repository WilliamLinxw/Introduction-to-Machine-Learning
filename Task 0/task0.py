import pandas as pd
import numpy as np

def read_Y_X(filename):
    train = pd.read_csv('./'+filename)
    print(train.head(10))
    X = train.iloc[:,2:].values
    Y = train.iloc[:,1:2].values
    return X, Y

def inverse(X):
    eps = 0.001
    dot = X.T@X
    if np.linalg.matrix_rank(dot) < min(dot.shape):
        dot += eps*np.identity(min(dot.shape))
    inverse = np.linalg.inv(dot)
    return inverse

def closeform(X, Y):
    w_hat = inverse(X)@X.T@Y
    return w_hat