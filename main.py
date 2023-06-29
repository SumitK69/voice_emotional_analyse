import json
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import wave
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import IPython.display as ipd
#! CNN model imports
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv1D, MaxPooling1D, Flatten, Activation, Dropout, Dense
from keras.optimizers import RMSprop


mylist = glob('.\\Data\\*\\*.wav')
print(type(mylist))

# obj = wave.open(mylist[0], "rb")
# print("no. of channel:", obj.getnchannels())
# print("sample width:", obj.getsampwidth())
# print("frame rate:", obj.getframerate())
# print("no. of frames", obj.getnframes())
# print("parameters: ", obj.getparams())


# data, sampling_rate = librosa.load('RawData/03-01-01-01-01-01-01.wav')
# plt.figure(figsize=(15, 5))
# librosa.display.waveshow(data, sr=sampling_rate)
# plt.show()


feeling_list = []
for item in mylist:

    if (item[-18:-16] == '01' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_neutral')
    if (item[-18:-16] == '01' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_neutral')
    if (item[-18:-16] == '02' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_calm')
    if (item[-18:-16] == '02' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_calm')
    if (item[-18:-16] == '03' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_happy')
    if (item[-18:-16] == '03' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_happy')
    if (item[-18:-16] == '04' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_sad')
    if (item[-18:-16] == '04' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_sad')
    if (item[-18:-16] == '05' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_angry')
    if (item[-18:-16] == '05' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_angry')
    if (item[-18:-16] == '06' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_fearful')
    if (item[-18:-16] == '06' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_fearful')
    if (item[-18:-16] == '07' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_disgust')
    if (item[-18:-16] == '07' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_disgust')
    if (item[-18:-16] == '08' and int(item[-6:-4]) % 2 != 0):
        feeling_list.append('male_surprised')
    if (item[-18:-16] == '08' and int(item[-6:-4]) % 2 == 0):
        feeling_list.append('female_surprised')

label = pd.DataFrame(feeling_list, columns=['labels'])


temp_df = pd.DataFrame()

for index, y in enumerate(mylist):
    X, sample_rate = librosa.load(
        y, duration=3, res_type='kaiser_fast', offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, n_mfcc=25), axis=0)
    feature = mfccs
    temp_df[index] = [feature]

df = temp_df.transpose()

df.rename(columns={0: 'features'}, inplace=True)

df2 = pd.DataFrame(df['features'].values.tolist())


newdf = pd.concat([df2, label], axis=1)


newdf = shuffle(newdf)


rnewdf = newdf.fillna(0)

# train and test split

X = rnewdf.iloc[:, :-1]
Y = rnewdf.iloc[:, -1:]

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(f"xtrain: {X_train.shape}, x_test:{X_test.shape}")
print(f"ytrain: {Y_train.shape}, y_test:{Y_test.shape}")

#! building CNN model

lb = LabelEncoder()

Y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
Y_test = np_utils.to_categorical(lb.fit_transform(Y_test))


X_traincnn = np.expand_dims(X_train, axis=2)
X_testcnn = np.expand_dims(X_test, axis=2)

print(f"xtraincnn:{X_traincnn.shape}")
print(f"xtraincnn:{X_testcnn.shape}")

model = Sequential()

model.add(Conv1D(256, 5, padding='same', input_shape=(256, 1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = RMSprop(learning_rate=0.00001, decay=1e-6)
# print(model.summary())

model.compile(loss='categorical-crossentropy',
              optimizer=opt, metrics=['accuracy'])

print(f"xtraincnn{X_traincnn.shape}")
print(f"xtestcnn{X_testcnn.shape}")
print(f"Ytrain{Y_train.shape}")
print(f"ytest{Y_test.shape}")


cnnhistory = model.fit(X_traincnn, Y_train, batch_size=16,
                       epochs=700, validation_data=(X_testcnn, Y_test))
