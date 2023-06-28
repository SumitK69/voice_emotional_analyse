import wave
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import IPython.display as ipd


mylist = glob('.\\Data\\*\\*.wav')
print(type(mylist))
obj = wave.open(mylist[0], "rb")

print("no. of channel:", obj.getnchannels())
print("sample width:", obj.getsampwidth())
print("frame rate:", obj.getframerate())
print("no. of frames", obj.getnframes())
print("parameters: ", obj.getparams())


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

label = pd.DataFrame(feeling_list)


# for index, y in enumerate(mylist):
#     X, sample_rate = librosa.load(
#         y, duration=3, res_type='kaiser_fast', offset=0.5)
#     sample_rate = np.array(sample_rate)
#     mfccs = np.mean(librosa.feature.mfcc(y=X, n_mfcc=25), axis=0)
