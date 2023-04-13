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



valid_noaircon = pd.read_csv('E:/IoT/Temperature_LSTM/검증데이터/last/noaircon1_valid_data.csv', header=None)
valid_R_L_3 = pd.read_csv('E:/IoT/Temperature_LSTM/검증데이터/last/R_L_3_valid_data.csv', header=None)


# window 나누기
seq_len = 20
sequence_length = seq_len + 1

result_valid_noaircon = []
for index in range(valid_noaircon[0].count() - sequence_length):
    result_valid_noaircon.append(valid_noaircon[index: index + sequence_length])

result_valid_R_L_3 = []
for index in range(valid_R_L_3[0].count() - sequence_length):
    result_valid_R_L_3.append(valid_R_L_3[index: index + sequence_length])

from sklearn.model_selection import train_test_split

result_valid_noaircon = np.array(result_valid_noaircon)
result_valid_R_L_3 = np.array(result_valid_R_L_3)

train = np.concatenate((result_valid_noaircon,result_valid_R_L_3,
                        ), axis=0)
#print(result_cooling_L_1.shape)

#np.random.shuffle(train)

print(train.dtype)

for j in train:
    for i in j:
        if i[0] == 'data_noaircon':
            i[0] = 0
        elif i[0] == 'data_cooling_L_1':
            i[0] = 11
        elif i[0] == 'data_cooling_L_2':
            i[0] = 12
        elif i[0] == 'data_cooling_L_3':
            i[0] = 13
        elif i[0] == 'data_after_cooling_L_1':
            i[0] = 14
        elif i[0] == 'data_after_cooling_L_2':
            i[0] = 15
        elif i[0] == 'data_after_cooling_L_3':
            i[0] = 16
        elif i[0] == 'data_cooling_R_1':
            i[0] = 21
        elif i[0] == 'data_cooling_R_2':
            i[0] = 22
        elif i[0] == 'data_cooling_R_3':
            i[0] = 23
        elif i[0] == 'data_after_cooling_R_1':
            i[0] = 24
        elif i[0] == 'data_after_cooling_R_2':
            i[0] = 25
        elif i[0] == 'data_after_cooling_R_3':
            i[0] = 26
        elif i[0] == 'data_cooling_B_1':
            i[0] = 31
        elif i[0] == 'data_cooling_B_2':
            i[0] = 32
        elif i[0] == 'data_cooling_B_3':
            i[0] = 33
        elif i[0] == 'data_after_cooling_B_1':
            i[0] = 34
        elif i[0] == 'data_after_cooling_B_2':
            i[0] = 35
        elif i[0] == 'data_after_cooling_B_3':
            i[0] = 36

np.save('E:/IoT/Temperature_LSTM/검증데이터/train, test split/train_data2.npy', train)

