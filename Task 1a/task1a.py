from cmath import sqrt
from bitarray import test
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import csv

def readxy(filename):
    train = pd.read_csv('./'+filename)
    print(train.head(10))
    X = train.iloc[:,1:].values
    Y = train.iloc[:,0:1].values
    return X, Y

X, Y = readxy('train.csv')
train_Y = np.delete(Y, slice(0,None,10), 0)

def fold(X, Y, n, alpha):
    error = 0
    for i in range(n):
        test_Y = Y[i:len(Y):n,:]
        test_X = X[i:len(X):n,:]

        train_Y = np.delete(Y, slice(i,None,n), 0)
        train_X = np.delete(X, slice(i,None,n), 0)
        
        reg = Ridge(alpha)
        reg.fit(train_X, train_Y)


        SE = 0
        for j in range(len(test_Y)):
            y_hat = np.dot(reg.coef_, test_X[j]) + reg.intercept_
            print('y_hat', y_hat)
            print('test y', test_Y[j])
            SE = SE + (test_Y[j] - y_hat)**2
        MSE = SE / len(test_Y)
        RMSE = MSE**0.5
        error += RMSE
    ave_error = error / n
    return ave_error[0]

sample = [fold(X, Y, 10, 0.1), fold(X, Y, 10, 1), fold(X, Y, 10, 10), fold(X, Y, 10, 100), fold(X, Y, 10, 200)]
print(sample)

