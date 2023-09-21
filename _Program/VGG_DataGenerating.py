import os
import numpy as np
from scipy.io import wavfile

'''
关于数据处理，依然需要一个数据生成程序。
因为我们需要根据已有的数据进行切分。
'''

# 文件路径
filePath = 'F:\\recording_data\\右侧单牙\\L6.wav'

# 牙齿标号
teethLabel = 6

# 读入双声道数据，切分成 3000 采样点为一组的数据。双声道认为是同一位置的两个通道。
frequency, audio = wavfile.read(filePath)
windowLength = 3000
audioLength = len(audio)

# 创建文件，防止读取错误。
try:
    data = np.load('cnn_data.npy')
    label = np.load('cnn_label.npy')
except:
    open('cnn_data.npy', 'ab')
    data = []
    open('cnn_label.npy', 'ab')
    label = []

'''
怎么定义存储的数据结构？
分两个文件存储。一个存储数据，一个存储标签。
'''
for i in range(0, audioLength, windowLength):
    if i + windowLength > audioLength:
        break
    # 左右声道分别添加进去
    data = np.append(data, audio[i:i + windowLength, 0])
    data = np.append(data, audio[i:i + windowLength, 1])
    label = np.append(label, teethLabel)

os.remove('cnn_data.npy')
np.save('cnn_data.npy', data)
os.remove('cnn_label.npy')
np.save('cnn_label.npy', label)

data = np.load('cnn_data.npy')
print(len(data))
label = np.load('cnn_label.npy')
print(len(label))
