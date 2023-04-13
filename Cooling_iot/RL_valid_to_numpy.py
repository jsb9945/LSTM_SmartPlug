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


L3_R0_valid = pd.read_csv('E:/IoT/Temperature_LSTM/검증데이터/last/RL_valid/L3_R0_valid.csv', header=None)
L1_R2_valid = pd.read_csv('E:/IoT/Temperature_LSTM/검증데이터/last/RL_valid/L1_R2_valid.csv', header=None)
L1_R1_valid = pd.read_csv('E:/IoT/Temperature_LSTM/검증데이터/last/RL_valid/L1_R1_valid.csv', header=None)
L0_R3_valid = pd.read_csv('E:/IoT/Temperature_LSTM/검증데이터/last/RL_valid/L0_R3_valid.csv', header=None)
L0_R2_valid = pd.read_csv('E:/IoT/Temperature_LSTM/검증데이터/last/RL_valid/L0_R2_valid.csv', header=None)

# window 나누기
seq_len = 20
sequence_length = seq_len + 1

result_L3_R0_valid = []
for index in range(L3_R0_valid[0].count() - sequence_length):
    result_L3_R0_valid.append(L3_R0_valid[index: index + sequence_length])

result_L1_R2_valid = []
for index in range(L1_R2_valid[0].count() - sequence_length):
    result_L1_R2_valid.append(L1_R2_valid[index: index + sequence_length])

result_L1_R1_valid = []
for index in range(L1_R1_valid[0].count() - sequence_length):
    result_L1_R1_valid.append(L1_R1_valid[index: index + sequence_length])

result_L0_R3_valid = []
for index in range(L0_R3_valid[0].count() - sequence_length):
    result_L0_R3_valid.append(L0_R3_valid[index: index + sequence_length])

result_L0_R2_valid = []
for index in range(L0_R2_valid[0].count() - sequence_length):
    result_L0_R2_valid.append(L0_R2_valid[index: index + sequence_length])


from sklearn.model_selection import train_test_split

result_valid_R_L_3 = np.array(result_L3_R0_valid)
result_L1_R2_valid = np.array(result_L1_R2_valid)
result_L1_R1_valid = np.array(result_L1_R1_valid)
result_L0_R3_valid = np.array(result_L0_R3_valid)
result_L0_R2_valid = np.array(result_L0_R2_valid)

train = np.concatenate((result_valid_R_L_3,result_L1_R2_valid,result_L1_R1_valid,result_L0_R3_valid,result_L0_R2_valid
                        ), axis=0)
print(train.shape)

#np.random.shuffle(train)

print(train.dtype)


np.save('E:/IoT/Temperature_LSTM/검증데이터/train, test split/RL_train_test_split/train_data.npy', train)
