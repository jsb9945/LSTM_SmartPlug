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


train = np.load('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/train_data.npy', allow_pickle=True)


#0~28044
#28044~35056
print(train.shape)

x_train = train[0:108620, :-1, 0:7].astype(float)
x_test = train[108620:, :-1, 0:7].astype(float)
y_train = train[0:108620, -1, 7:].astype(float)
y_test = train[108620:, -1, 7:].astype(float)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(20, 7)))
model.add(LSTM(60, return_sequences=False))
model.add(Dense(6, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

with tf.device("/device:gpu:0"):
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=30)
#batch_size = 32로 고정

scores = model.evaluate(x_test, y_test)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1]*100))
pred = model.predict(x_test)


model.save('model_window_20_noaircon_plus.h5')


