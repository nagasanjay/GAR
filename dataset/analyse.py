import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('dataset/dataset copy.csv', delimiter=',', header=None)

data = data.iloc[:,1:6]

'''
print('\n speed \n')
print(data.iloc[:,0].describe())
print('\n throttle \n')
print(data.iloc[:,1].describe())
print('\n steer \n')
print(data.iloc[:,2].describe())
print('\n break \n')
print(data.iloc[:,3].describe())

print('\n mode \n')
print(data.iloc[:,0].mode())
print(data.iloc[:,1].mode())
print(data.iloc[:,2].mode())
print(data.iloc[:,3].mode())
'''
print(data.shape)
X = pd.DataFrame()
Y = pd.DataFrame()

X['speed'] = data.iloc[:,0]
X['dir'] = data.iloc[:,4]

Y['throttle'] = data.iloc[:, 1]
Y['steer'] = data.iloc[:, 2]
Y['brake'] = data.iloc[:,3].apply(lambda y: 0.0 if y < 0.5 else 1.0)

print(type(X), X.describe(), X.shape)
print(type(Y), Y.describe(), Y.shape)

print(X.head())
print(Y.head())

c , b = np.histogram(Y['brake'])
print(c)
print(b)
plt.bar(b[:-1], b, weights=c)