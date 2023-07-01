from keras.models import model_from_json
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


print(type(df))
print(df.shape)
print(df[:10])

df.rename(columns={0: 'features'}, inplace=True)


df2 = pd.DataFrame(df['features'].values.tolist())


newdf = pd.concat([df2, label], axis=1)


newdf = shuffle(newdf)


rnewdf = newdf.fillna(0)


newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]


train_features = train.iloc[:, :-1]
train_labels = train.iloc[:, -1:]
test_features = test.iloc[:, :-1]
test_labels = test.iloc[:, -1:]

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# print(type(X_train))

#! building CNN model
X_train = np.array(train_features)
y_train = np.array(train_labels)
X_test = np.array(test_features)
y_test = np.array(test_labels)


lb = LabelEncoder()

Y_train = np_utils.to_categorical(lb.fit_transform(y_train))
Y_test = np_utils.to_categorical(lb.fit_transform(y_test))

print(type(Y_train))


X_traincnn = np.expand_dims(X_train, axis=2)
X_testcnn = np.expand_dims(X_test, axis=2)

print(f"xtraincnn:{X_traincnn.shape}")
print(f"xtestcnn:{X_testcnn.shape}")


model = Sequential()

model.add(Conv1D(256, 5, padding='same', input_shape=(130, 1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(16))
model.add(Activation('softmax'))
opt = RMSprop(learning_rate=0.00001, decay=1e-6)
print(model.summary())

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer="adam", metrics=['accuracy'])

# print(f"xtraincnn{X_traincnn.shape}")
# print(f"xtestcnn{X_testcnn.shape}")
# print(f"Ytrain{Y_train.shape}")
# print(f"ytest{Y_test.shape}")


cnnhistory = model.fit(x=X_traincnn, y=Y_train,
                       epochs=30, validation_data=(X_testcnn, Y_test))

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# loading json and creating model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                     optimizer="adam", metrics=['accuracy'])
score = loaded_model.evaluate(X_testcnn, Y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
