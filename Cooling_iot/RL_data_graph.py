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


vali = np.load('E:/IoT/Temperature_LSTM/검증데이터/train, test split/RL_train_test_split/train_data.npy', allow_pickle=True)


x_test = vali[:, 0:20, 0:8].astype(float)
y_test = vali[:, 20, 8:].astype(float)

'''x_test = np.array(x_test)

num_sample   = x_test.shape[0]
num_sequence = x_test.shape[1]
num_feature  = x_test.shape[2]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

for i in range(num_sequence):
    scaler.partial_fit(x_test[:, i, :])

results = []
for i in range(num_sequence):
    results.append(scaler.transform(x_test[:, i, :]).reshape(num_sample, 1, num_feature))
x_test = np.concatenate(results, axis=1)'''


x = x_test[4020:4250, :].astype(float)
y = y_test[4020:4250, :].astype(float)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(x_test[4020:4250])

model = keras.models.load_model('RL_train1_norm10.h5')
model.summary()

pred = model.predict(x)

pred_0 = []
pred_1 = []
pred_2 = []
pred_3 = []
pred_4 = []
pred_5 = []


for i in range(len(pred)):
    for j in range(len(pred[i])):
        pred[i][j] = (pred[i][j]-32)*5/9
        if j==0:
            pred_0.append(pred[i][j])
        elif j==1:
            pred_1.append(pred[i][j])
        elif j==2:
            pred_2.append(pred[i][j])
        elif j==3:
            pred_3.append(pred[i][j])
        elif j==4:
            pred_4.append(pred[i][j])
        elif j==5:
            pred_5.append(pred[i][j])



y_0 = []
y_1 = []
y_2 = []
y_3 = []
y_4 = []
y_5 = []


for i in range(len(y)):
    for j in range(len(y[i])):
        y[i][j] = (y[i][j] - 32) * 5 / 9
        if j==0:
            y_0.append(y[i][j])
        elif j==1:
            y_1.append(y[i][j])
        elif j==2:
            y_2.append(y[i][j])
        elif j==3:
            y_3.append(y[i][j])
        elif j==4:
            y_4.append(y[i][j])
        elif j==5:
            y_5.append(y[i][j])


plt.figure(figsize = (12,9))
plt.title("node4")
plt.plot(pred_3, label='pred')
plt.plot(y_3, label='actual')
plt.xlabel('Time(m)', labelpad=15, loc='right')
plt.ylabel('Temperature(°C)', labelpad=15, loc='top')
plt.legend()
plt.show()


'''plt.subplot(325)
plt.title("node1")
#plt.ylim(72,76)
plt.plot(y_0, label='True')
plt.plot(pred_0, label='Prediction')
plt.legend()


plt.subplot(323)
plt.title("node2")
#plt.ylim(72,76)
plt.plot(y_1, label='True')
plt.plot(pred_1, label='Prediction')
plt.legend()

plt.subplot(321)
plt.title("node3")
#plt.ylim(74,78)
plt.plot(y_2, label='True')
plt.plot(pred_2, label='Prediction')
plt.legend()

plt.subplot(322)
plt.title("node4")
#plt.ylim(78,82)
plt.plot(y_3, label='True')
plt.plot(pred_3, label='Prediction')
plt.legend()

plt.subplot(324)
plt.title("node5")
#plt.ylim(74,78)
plt.plot(y_4, label='True')
plt.plot(pred_4, label='Prediction')
plt.legend()

plt.subplot(326)
plt.title("node6")
#plt.ylim(74,78)
plt.plot(y_5, label='True')
plt.plot(pred_5, label='Prediction')
plt.legend()

plt.show()'''
