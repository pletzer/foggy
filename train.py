import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow import keras
import random
from sklearn.metrics import confusion_matrix

random.seed(123)
np.random.seed(345)


def shuffle(x, y):
    nrow, ncol = x.shape
    inds = [i for i in range(nrow)]
    random.shuffle(inds)
    print(f'shuffled indices = {inds}')
    xShuffled = np.zeros((nrow, ncol), np.float32)
    yShuffled = np.zeros((nrow,), np.float32)
    for i in range(nrow):
        xShuffled[i, :] = x[ inds[i] , :]
        yShuffled[i] = y[ inds[i] ]
    return xShuffled, yShuffled


# read the data 
df = pd.read_csv('export_dataframe_subset_blind_with_fg.csv')
print(f"number of cases with/without fog {(df['fg'] == 1).sum()}/{(df['fg'] == 0).sum()}")

for fld in ['msl_p?', 'r_p?_l0']:
    for i in range(4):
        colname = fld.replace('?', str(i))
        print(f'max/min of {colname}: {df[colname].max()}/{df[colname].min()}')


# normalize
df2 = df.drop(columns="filename")

# normalize between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df2.values)

dfNorm = pd.DataFrame(x_scaled, columns=df2.columns)

print('after normalization')
cols = []
for fld in ['msl_p?', 'r_p?_l0',]:
    for i in range(4): # number of points
        colname = fld.replace('?', str(i))
        print(f'max/min of {colname}: {dfNorm[colname].max()}/{dfNorm[colname].min()}')
        cols.append(colname)

# create input
x = np.zeros((dfNorm.shape[0], len(cols)), np.float32)
y = np.array(dfNorm['fg'])

# choose n samples of each fog/non-fog cases
n = 1000 #45
rs = dfNorm.groupby(['fg']).apply(lambda x: x.sample(n, replace=True))
xSelect = np.zeros((rs.shape[0], len(cols)), np.float32)
ySelect = np.array(rs['fg'])

for i in range(len(cols)):
    colname = cols[i]
    x[:, i] = dfNorm[colname]
    xSelect[:, i] = rs[colname]

print(f'xSelect = {xSelect}')
print(f'ySelect = {ySelect}')

print(f'x = {x}')
print(f'y = {y}')


# shuffle selected samples
xTrain, yTrain = shuffle(xSelect, ySelect)

print(f'total    set has {(y == 1).sum()}/{(y == 0).sum()} cases with/without fog')
print(f'training set has {(yTrain == 1).sum()}/{(yTrain == 0).sum()} cases with/without fog')

# build model
model = keras.models.Sequential()
model.add( keras.layers.Dense(8, activation='relu') )
model.add( keras.layers.Dropout(0.1) )
model.add( keras.layers.Dense(16, activation='relu') )
model.add( keras.layers.Dropout(0.1) )
model.add( keras.layers.Dense(16, activation='relu') )
model.add( keras.layers.Dropout(0.1) )
model.add( keras.layers.Dense(8, activation='relu') )
#model.add( keras.layers.Dense(8, activation='relu') )
#model.add( keras.layers.Dense(8, activation='relu') )
model.add( keras.layers.Dense(1) ) #, activation='softmax'))
model.compile(optimizer='sgd',
              loss='mean_squared_error', 
              metrics=['accuracy'])
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy', 
#              metrics=['accuracy'])

# train
model.fit(xTrain, yTrain, epochs=1000)

print(model.summary())

# test/predict
yPred = np.round( model.predict(x).reshape(y.shape) )
print(f'yPred = {yPred}')
print(f'yPred max/min= {yPred.max()}/{yPred.min()}')
print(f'y = {y}')
#yPred = min(1.0, max(0., yPred))

num_errors = int(np.abs(yPred - y).sum())
print(f'number of errors = {num_errors} out of {x.shape[0]} ({100*num_errors/x.shape[0]:.1f} %)')

# compute the confusion matrix
confusion_mat = confusion_matrix(y, yPred)
print('confusion matrix:')
print(confusion_mat)

