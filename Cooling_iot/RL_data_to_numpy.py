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

df_noaircon = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_noaircon.csv', header = None)

df_cooling_L_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_cooling_L_1.csv', header = None)
df_cooling_L_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_cooling_L_2.csv', header = None)
df_cooling_L_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_cooling_L_3.csv', header = None)

df_cooling_R_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_cooling_R_1.csv', header = None)
df_cooling_R_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_cooling_R_2.csv', header = None)
df_cooling_R_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_cooling_R_3.csv', header = None)

# window 나누기
seq_len = 20
sequence_length = seq_len + 1

result_noaircon = []
for index in range(df_noaircon[0].count() - sequence_length):
    result_noaircon.append(df_noaircon[index: index + sequence_length])

result_cooling_L_1 = []
for index in range(df_cooling_L_1[0].count() - sequence_length):
    result_cooling_L_1.append(df_cooling_L_1[index: index + sequence_length])

result_cooling_L_2 = []
for index in range(df_cooling_L_2[0].count() - sequence_length):
    result_cooling_L_2.append(df_cooling_L_2[index: index + sequence_length])

result_cooling_L_3 = []
for index in range(df_cooling_L_3[0].count() - sequence_length):
    result_cooling_L_3.append(df_cooling_L_3[index: index + sequence_length])

result_cooling_R_1 = []
for index in range(df_cooling_R_1[0].count() - sequence_length):
    result_cooling_R_1.append(df_cooling_R_1[index: index + sequence_length])

result_cooling_R_2 = []
for index in range(df_cooling_R_2[0].count() - sequence_length):
    result_cooling_R_2.append(df_cooling_R_2[index: index + sequence_length])

result_cooling_R_3 = []
for index in range(df_cooling_R_3[0].count() - sequence_length):
    result_cooling_R_3.append(df_cooling_R_3[index: index + sequence_length])


result_noaircon = np.array(result_noaircon)
result_cooling_L_1 = np.array(result_cooling_L_1)
result_cooling_L_2 = np.array(result_cooling_L_2)
result_cooling_L_3 = np.array(result_cooling_L_3)
result_cooling_R_1 = np.array(result_cooling_R_1)
result_cooling_R_2 = np.array(result_cooling_R_2)
result_cooling_R_3 = np.array(result_cooling_R_3)

train = np.concatenate((result_noaircon,
                        result_cooling_L_1,result_cooling_L_2,result_cooling_L_3,
                        result_cooling_R_1, result_cooling_R_2, result_cooling_R_3,
                        ), axis=0)
np.random.shuffle(train)

np.save('E:/IoT/Temperature_LSTM/학습데이터/RL_data/RL_train_data.npy', train)

