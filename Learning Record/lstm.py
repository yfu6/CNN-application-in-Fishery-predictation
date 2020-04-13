import math
import keras as K
from pandas import DataFrame
from pandas import concat
from numpy import concatenate
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sklearn.metrics 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from keras.layers import Dropout


dataset = read_csv('main8.csv', header=0, index_col=0)
dataset.info()
values = dataset.values

# 确保所有数据都是float
values = values.astype('float32')
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 框架作为监督学习
value2 = scaled

n_train_hours = int(len(dataset) * 0.8)
k=int(len(dataset) * 0.9)
train = value2[:n_train_hours, :]
test = value2[n_train_hours:k, :]
cheak = value2[k:, :]
# 建立输入和输出集
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
cheak_X, cheak_y = cheak[:, :-1], cheak[:, -1]
# 将输入集转化为3维
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
cheak_X = cheak_X.reshape((cheak_X.shape[0], 1, cheak_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# 构建神经网络模型
model = Sequential()
model.add(LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')


# 调整神经网络
history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# 显示损失值

print(model.summary())
# 进行预测
pyplot.plot(history.history['loss'], label='train')

pyplot.legend()
pyplot.show()

ycheak = model.predict(cheak_X)
cheak_X = cheak_X.reshape((cheak_X.shape[0], cheak_X.shape[2]))

inv_cheak = concatenate((cheak_X[:, 0:],ycheak), axis=1)
inv_cheak = scaler.inverse_transform(inv_cheak)
inv_cheak =inv_cheak[:,6]
cheak_y = cheak_y.reshape((len(cheak_y), 1))

inv_c = concatenate((cheak_X[:, 0:],cheak_y), axis=1)
inv_c = scaler.inverse_transform(inv_c)
inv_c = inv_c[:,6]


pyplot.plot(inv_c)
pyplot.plot(inv_cheak)
pyplot.show()


