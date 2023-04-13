import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import pandas as pd
#import lightgbm as lgbm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import os


train = np.load('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_train_data.npy', allow_pickle=True)

from sklearn.preprocessing import MinMaxScaler

x_train = train[0:4000, :-1, 0:8].astype(float)
x_test = train[4000:, :-1, 0:8].astype(float)
y_train = train[0:4000, -1, 8:].astype(float)
y_test = train[4000:, -1, 8:].astype(float)

#x_train 정규화
x_train = np.array(x_train)

num_sample   = x_train.shape[0]
num_sequence = x_train.shape[1]
num_feature  = x_train.shape[2]

scaler = MinMaxScaler()

for i in range(num_sequence):
    scaler.partial_fit(x_train[:, i, :])

results = []
for i in range(num_sequence):
    results.append(scaler.transform(x_train[:, i, :]).reshape(num_sample, 1, num_feature))
x_train = np.concatenate(results, axis=1)

#x_test 정규화
x_test = np.array(x_test)

num_sample   = x_test.shape[0]
num_sequence = x_test.shape[1]
num_feature  = x_test.shape[2]

scaler = MinMaxScaler()

for i in range(num_sequence):
    scaler.partial_fit(x_test[:, i, :])

results = []
for i in range(num_sequence):
    results.append(scaler.transform(x_test[:, i, :]).reshape(num_sample, 1, num_feature))
x_test = np.concatenate(results, axis=1)

'''#y_train 정규화
y_train = np.array(y_train)
scaler = MinMaxScaler()
y_train = scaler.fit(y_train)

#y_test 정규화
y_test = np.array(y_test)
scaler = MinMaxScaler()
y_test = scaler.fit(y_test)'''

#0~28044
#28044~35056
print(train.shape)

x_train = train[0:4000, :-1, 0:8].astype(float)
x_test = train[4000:, :-1, 0:8].astype(float)
y_train = train[0:4000, -1, 8:].astype(float)
y_test = train[4000:, -1, 8:].astype(float)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(20, 8)))
model.add(Dropout(0.05))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.05))
model.add(Dense(128, activation='linear'))
model.add(Dense(6, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

with tf.device("/device:gpu:0"):
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=60)
#batch_size = 32로 고정

scores = model.evaluate(x_test, y_test)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1]*100))
pred = model.predict(x_test)


model.save('RL_train1_norm12.h5')


