import pandas as pd

data = pd.read_csv('dataset/dataset.csv', delimiter=',', header=None)

data = data.iloc[:,1:5]
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