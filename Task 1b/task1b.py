import pandas as pd
import numpy as np
from sklearn import linear_model

def readxy(filename):
    train = pd.read_csv('./'+filename)
    print(train.head(10))
    # train = train.sample(frac=1).reset_index(drop=True)
    # print(train.head())
    return train

train = readxy('train.csv')

def feature(train):
    train = train.rename(columns={'x1':'phi1', 'x2':'phi2', 'x3':'phi3', 'x4':'phi4', 'x5':'phi5'})
    train['phi6'] = train['phi1'] * train['phi1']
    train['phi7'] = train['phi2'] * train['phi2']
    train['phi8'] = train['phi3'] * train['phi3']
    train['phi9'] = train['phi4'] * train['phi4']
    train['phi10'] = train['phi5'] * train['phi5']
    train['phi11'] = np.exp(train['phi1'])
    train['phi12'] = np.exp(train['phi2'])
    train['phi13'] = np.exp(train['phi3'])
    train['phi14'] = np.exp(train['phi4'])
    train['phi15'] = np.exp(train['phi5'])
    train['phi16'] = np.cos(train['phi1'])
    train['phi17'] = np.cos(train['phi2'])
    train['phi18'] = np.cos(train['phi3'])
    train['phi19'] = np.cos(train['phi4'])
    train['phi20'] = np.cos(train['phi5'])
    train.insert(train.shape[1], 'phi21', 1)
    print(train.head(10))
    return train
    

train = feature(train)
X = train.iloc[:,2:22].values
print(X.shape)
Y = train.iloc[:,1:2].values
print(Y.shape)

reg = linear_model.LinearRegression()
reg.fit(X, Y)

coef = reg.coef_[0]
coef = np.append(coef, reg.intercept_)
print(reg.coef_[0])
print(reg.intercept_)
print(coef)

np.savetxt('myfile.csv', coef, delimiter=',')