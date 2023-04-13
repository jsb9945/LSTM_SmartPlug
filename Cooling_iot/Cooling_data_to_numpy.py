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



df_noaircon = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_noaircon_last2.csv', header = None)
df_after_cooling_L_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_L_1_last2.csv', header = None)
df_after_cooling_L_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_L_2_last2.csv', header = None)
df_after_cooling_L_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_L_3_last2.csv', header = None)
df_cooling_L_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_L_1_last2.csv', header = None)
df_cooling_L_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_L_2_last2.csv', header = None)
df_cooling_L_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_L_3_last2.csv', header = None)

df_after_cooling_R_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_R_1_last2.csv', header = None)
df_after_cooling_R_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_R_2_last2.csv', header = None)
df_after_cooling_R_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_R_3_last2.csv', header = None)
df_cooling_R_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_R_1_last2.csv', header = None)
df_cooling_R_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_R_2_last2.csv', header = None)
df_cooling_R_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_R_3_last2.csv', header = None)

df_after_cooling_B_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_B_1_last2.csv', header = None)
df_after_cooling_B_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_B_2_last2.csv', header = None)
df_after_cooling_B_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_after_cooling_B_3_last2.csv', header = None)
df_cooling_B_1 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_B_1_last2.csv', header = None)
df_cooling_B_2 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_B_2_last2.csv', header = None)
df_cooling_B_3 = pd.read_csv('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/data_cooling_B_3_last2.csv', header = None)

# window 나누기
seq_len = 20
sequence_length = seq_len + 1

result_noaircon = []
for index in range(df_noaircon[0].count() - sequence_length):
    result_noaircon.append(df_noaircon[index: index + sequence_length])

result_after_cooling_L_1 = []
for index in range(df_after_cooling_L_1[0].count() - sequence_length):
    result_after_cooling_L_1.append(df_after_cooling_L_1[index: index + sequence_length])

result_after_cooling_L_2 = []
for index in range(df_after_cooling_L_2[0].count() - sequence_length):
    result_after_cooling_L_2.append(df_after_cooling_L_2[index: index + sequence_length])

result_after_cooling_L_3 = []
for index in range(df_after_cooling_L_3[0].count() - sequence_length):
    result_after_cooling_L_3.append(df_after_cooling_L_3[index: index + sequence_length])

result_cooling_L_1 = []
for index in range(df_cooling_L_1[0].count() - sequence_length):
    result_cooling_L_1.append(df_cooling_L_1[index: index + sequence_length])

result_cooling_L_2 = []
for index in range(df_cooling_L_2[0].count() - sequence_length):
    result_cooling_L_2.append(df_cooling_L_2[index: index + sequence_length])

result_cooling_L_3 = []
for index in range(df_cooling_L_3[0].count() - sequence_length):
    result_cooling_L_3.append(df_cooling_L_3[index: index + sequence_length])
#
result_after_cooling_R_1 = []
for index in range(df_after_cooling_R_1[0].count() - sequence_length):
    result_after_cooling_R_1.append(df_after_cooling_R_1[index: index + sequence_length])

result_after_cooling_R_2 = []
for index in range(df_after_cooling_R_2[0].count() - sequence_length):
    result_after_cooling_R_2.append(df_after_cooling_R_2[index: index + sequence_length])

result_after_cooling_R_3 = []
for index in range(df_after_cooling_R_3[0].count() - sequence_length):
    result_after_cooling_R_3.append(df_after_cooling_R_3[index: index + sequence_length])

result_cooling_R_1 = []
for index in range(df_cooling_R_1[0].count() - sequence_length):
    result_cooling_R_1.append(df_cooling_R_1[index: index + sequence_length])

result_cooling_R_2 = []
for index in range(df_cooling_R_2[0].count() - sequence_length):
    result_cooling_R_2.append(df_cooling_R_2[index: index + sequence_length])

result_cooling_R_3 = []
for index in range(df_cooling_R_3[0].count() - sequence_length):
    result_cooling_R_3.append(df_cooling_R_3[index: index + sequence_length])
#
result_after_cooling_B_1 = []
for index in range(df_after_cooling_B_1[0].count() - sequence_length):
    result_after_cooling_B_1.append(df_after_cooling_B_1[index: index + sequence_length])

result_after_cooling_B_2 = []
for index in range(df_after_cooling_B_2[0].count() - sequence_length):
    result_after_cooling_B_2.append(df_after_cooling_B_2[index: index + sequence_length])

result_after_cooling_B_3 = []
for index in range(df_after_cooling_L_3[0].count() - sequence_length):
    result_after_cooling_B_3.append(df_after_cooling_B_3[index: index + sequence_length])

result_cooling_B_1 = []
for index in range(df_cooling_B_1[0].count() - sequence_length):
    result_cooling_B_1.append(df_cooling_B_1[index: index + sequence_length])

result_cooling_B_2 = []
for index in range(df_cooling_B_2[0].count() - sequence_length):
    result_cooling_B_2.append(df_cooling_B_2[index: index + sequence_length])

result_cooling_B_3 = []
for index in range(df_cooling_B_3[0].count() - sequence_length):
    result_cooling_B_3.append(df_cooling_B_3[index: index + sequence_length])

from sklearn.model_selection import train_test_split
result_noaircon, result_noaircon_plt = train_test_split(result_noaircon, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_L_1, result_after_cooling_L_1_plt = train_test_split(result_after_cooling_L_1, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_L_2, result_after_cooling_L_2_plt = train_test_split(result_after_cooling_L_2, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_L_3, result_after_cooling_L_3_plt = train_test_split(result_after_cooling_L_3, test_size=0.2, shuffle=False, random_state=1)
result_cooling_L_1, result_cooling_L_1_plt = train_test_split(result_cooling_L_1, test_size=0.2, shuffle=False, random_state=1)
result_cooling_L_2, result_cooling_L_2_plt = train_test_split(result_cooling_L_2, test_size=0.2, shuffle=False, random_state=1)
result_cooling_L_3, result_cooling_L_3_plt = train_test_split(result_cooling_L_3, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_R_1, result_after_cooling_R_1_plt = train_test_split(result_after_cooling_R_1, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_R_2, result_after_cooling_R_2_plt = train_test_split(result_after_cooling_R_2, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_R_3, result_after_cooling_R_3_plt = train_test_split(result_after_cooling_R_3, test_size=0.2, shuffle=False, random_state=1)
result_cooling_R_1, result_cooling_R_1_plt = train_test_split(result_cooling_R_1, test_size=0.2, shuffle=False, random_state=1)
result_cooling_R_2, result_cooling_R_2_plt = train_test_split(result_cooling_R_2, test_size=0.2, shuffle=False, random_state=1)
result_cooling_R_3, result_cooling_R_3_plt = train_test_split(result_cooling_R_3, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_B_1, result_after_cooling_B_1_plt = train_test_split(result_after_cooling_B_1, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_B_2, result_after_cooling_B_2_plt = train_test_split(result_after_cooling_B_2, test_size=0.2, shuffle=False, random_state=1)
result_after_cooling_B_3, result_after_cooling_B_3_plt = train_test_split(result_after_cooling_B_3, test_size=0.2, shuffle=False, random_state=1)
result_cooling_B_1, result_cooling_B_1_plt = train_test_split(result_cooling_B_1, test_size=0.2, shuffle=False, random_state=1)
result_cooling_B_2, result_cooling_B_2_plt = train_test_split(result_cooling_B_2, test_size=0.2, shuffle=False, random_state=1)
result_cooling_B_3, result_cooling_B_3_plt = train_test_split(result_cooling_B_3, test_size=0.2, shuffle=False, random_state=1)

#model
result_noaircon = np.array(result_noaircon)
result_after_cooling_L_1 = np.array(result_after_cooling_L_1)
result_after_cooling_L_2 = np.array(result_after_cooling_L_2)
result_after_cooling_L_3 = np.array(result_after_cooling_L_3)
result_cooling_L_1 = np.array(result_cooling_L_1)
result_cooling_L_2 = np.array(result_cooling_L_2)
result_cooling_L_3 = np.array(result_cooling_L_3)
result_after_cooling_R_1 = np.array(result_after_cooling_R_1)
result_after_cooling_R_2 = np.array(result_after_cooling_R_2)
result_after_cooling_R_3 = np.array(result_after_cooling_R_3)
result_cooling_R_1 = np.array(result_cooling_R_1)
result_cooling_R_2 = np.array(result_cooling_R_2)
result_cooling_R_3 = np.array(result_cooling_R_3)
result_after_cooling_B_1 = np.array(result_after_cooling_B_1)
result_after_cooling_B_2 = np.array(result_after_cooling_B_2)
result_after_cooling_B_3 = np.array(result_after_cooling_B_3)
result_cooling_B_1 = np.array(result_cooling_B_1)
result_cooling_B_2 = np.array(result_cooling_B_2)
result_cooling_B_3 = np.array(result_cooling_B_3)
#print(result_cooling_L_1.shape)

train = np.concatenate((result_noaircon,result_after_cooling_L_1,result_after_cooling_L_2,result_after_cooling_L_3,
                        result_cooling_L_1,result_cooling_L_2,result_cooling_L_3,
                        result_after_cooling_R_1, result_after_cooling_R_2, result_after_cooling_R_3,
                        result_cooling_R_1, result_cooling_R_2, result_cooling_R_3,
                        result_after_cooling_B_1,result_after_cooling_B_2,result_after_cooling_B_3,
                        result_cooling_B_1,result_cooling_B_2,result_cooling_B_3,
                        ), axis=0)
np.random.shuffle(train)

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

np.save('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/train_data.npy', train)

result_noaircon_plt = np.array(result_noaircon_plt)
result_after_cooling_L_1_plt = np.array(result_after_cooling_L_1_plt)
result_after_cooling_L_2_plt = np.array(result_after_cooling_L_2_plt)
result_after_cooling_L_3_plt = np.array(result_after_cooling_L_3_plt)
result_cooling_L_1_plt = np.array(result_cooling_L_1_plt)
result_cooling_L_2_plt = np.array(result_cooling_L_2_plt)
result_cooling_L_3_plt = np.array(result_cooling_L_3_plt)
result_after_cooling_R_1_plt = np.array(result_after_cooling_R_1_plt)
result_after_cooling_R_2_plt = np.array(result_after_cooling_R_2_plt)
result_after_cooling_R_3_plt = np.array(result_after_cooling_R_3_plt)
result_cooling_R_1_plt = np.array(result_cooling_R_1_plt)
result_cooling_R_2_plt = np.array(result_cooling_R_2_plt)
result_cooling_R_3_plt = np.array(result_cooling_R_3_plt)
result_after_cooling_B_1_plt = np.array(result_after_cooling_B_1_plt)
result_after_cooling_B_2_plt = np.array(result_after_cooling_B_2_plt)
result_after_cooling_B_3_plt = np.array(result_after_cooling_B_3_plt)
result_cooling_B_1_plt = np.array(result_cooling_B_1_plt)
result_cooling_B_2_plt = np.array(result_cooling_B_2_plt)
result_cooling_B_3_plt = np.array(result_cooling_B_3_plt)

vali = np.concatenate((result_noaircon_plt,result_after_cooling_L_1_plt,result_after_cooling_L_2_plt,result_after_cooling_L_3_plt,
                        result_cooling_L_1_plt,result_cooling_L_2_plt,result_cooling_L_3_plt,
                        result_after_cooling_R_1_plt, result_after_cooling_R_2_plt, result_after_cooling_R_3_plt,
                        result_cooling_R_1_plt, result_cooling_R_2_plt, result_cooling_R_3_plt,
                        result_after_cooling_B_1_plt,result_after_cooling_B_2_plt,result_after_cooling_B_3_plt,
                        result_cooling_B_1_plt,result_cooling_B_2_plt,result_cooling_B_3_plt,
                        ), axis=0)



for j in vali:
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

np.save('E:/IoT/Temperature_LSTM/학습데이터/last_data/last2/vali_data.npy', vali)