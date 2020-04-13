#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import time, datetime
df_data_5minute=pd.read_csv('黄金主力5分钟数据.csv')
'''
或者使用JQdata
from jqdatasdk import *
#jqdata的账号密码
auth('邮箱：', 'jiaohiabin@ruc.edu.cn')
df_data_5minute= get_price('AU9999.XSGE',   start_date='2016-01-01', end_date='2018-01-01', frequency='5m')
'''


# In[55]:


df_data_5minute.head()


# In[56]:


df_data_5minute


# In[57]:


df_data_5minute.drop('Unnamed: 0', axis=1, inplace=True)
df_data_5minute


# In[58]:


df=df_data_5minute
close = df['close']
df.drop(labels=['close'], axis=1,inplace = True)
df.insert(0, 'close', close)
df


# In[59]:


data_train =df.iloc[:int(df.shape[0] * 0.7), :]
data_test = df.iloc[int(df.shape[0] * 0.7):, :]
print(data_train.shape, data_test.shape)


# In[60]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
import time
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)


# In[61]:


data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)


# In[62]:


data_train


# In[63]:


from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

output_dim = 1
batch_size = 256
epochs = 60
seq_len = 5
hidden_size = 128


TIME_STEPS = 5
INPUT_DIM = 6

lstm_units = 64
X_train = np.array([data_train[i : i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0]- seq_len)])
X_test = np.array([data_test[i : i + seq_len, :] for i in range(data_test.shape[0]- seq_len)])
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[67]:


inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
#drop1 = Dropout(0.3)(inputs)

x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
#x = Conv1D(filters=128, kernel_size=5, activation='relu')(output1)#embedded_sequences
x = MaxPooling1D(pool_size = 5)(x)
x = Dropout(0.2)(x)

print(x.shape)


# In[68]:


lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
#lstm_out = LSTM(lstm_units,activation='relu')(x)
print(lstm_out.shape)


# In[70]:


output = Dense(1, activation='sigmoid')(lstm_out)
#output = Dense(10, activation='sigmoid')(drop2)

model = Model(inputs=inputs, outputs=output)
print(model.summary())


# In[71]:


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
y_pred = model.predict(X_test)
print('MSE Train loss:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test loss:', model.evaluate(X_test, y_test, batch_size=batch_size))
plt.plot(y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()


# 随着训轮数（epoch）的增加，误差（loss）不断减小 loss: 0.0410左右

# In[ ]:





# In[ ]:




