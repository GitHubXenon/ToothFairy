import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
from tensorflow.keras.optimizers import SGD

'''
该程序构造卷积神经网络，自动提取特征并实现牙齿分类。
关于卷积引包名称：在 Keras2.0 之前使用的是 ConvolutionND，N=1,2,3，在 Keras2.0 之后就被改作了 ConvND。
'''

print('正在读入数据...')
data = np.load('cnn_data.npy')
label = np.load('cnn_label.npy')
windowLength = 3000

classNumber = 6

# 生成虚拟数据
# 第一维表示数据的组数
xTrain = np.array([])
yTrain = np.array([])
for i in range(0, len(label)):
    softmaxVector = np.zeros(classNumber, dtype=float)
    softmaxVector[int(label[i] - 1)] = 1
    yTrain = np.append(yTrain, softmaxVector)
    xTrain = np.append(xTrain, data[i:i + 2 * windowLength])

# 以下内容获得的是测试集的数据
print('正在准备测试集...')
xTest = np.array([])
yTest = np.array([])
index = np.random.randint(len(label), size=100)  # 长度 100 的一维随机数数组
for i in index:
    # print(i)
    softmaxVector = np.zeros(classNumber, dtype=float)
    softmaxVector[int(label[i] - 1)] = 1
    yTest = np.append(yTest, softmaxVector)
    xTest = np.append(xTest, data[i:i + 2 * windowLength])
    # 获得测试集后，要从训练集中删除该数据。
    xTrain = np.delete(xTrain, range(i, i + 2 * windowLength))
    yTrain = np.delete(yTrain, range(i, i + classNumber))

# 第一维：组数 | 中间维：数据维数 | 最后维：通道数
xTrain = xTrain.reshape((int(len(xTrain) / 2 / windowLength), windowLength, 2))
# keras 需要的数据中，第一个参数是组数。
yTrain = yTrain.reshape((int(len(yTrain) / classNumber), classNumber))

# 第一维：组数 | 中间维：数据维数 | 最后维：通道数
xTest = xTest.reshape((int(len(xTest) / 2 / windowLength), windowLength, 2))
# keras 需要的数据中，第一个参数是组数。
yTest = yTest.reshape((int(len(yTest) / classNumber), classNumber))

print('正在构建模型...')
model = Sequential()
model.add(Conv1D(32, 16, activation='relu', input_shape=(3000, 2)))
model.add(Conv1D(32, 16, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.15))

model.add(Conv1D(64, 16, activation='relu'))
model.add(Conv1D(64, 16, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classNumber, activation='softmax'))

sgd = SGD(lr=0.00000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print('开始训练...')
model.fit(xTrain, yTrain, batch_size=32, epochs=200)

print('评估中...')
score = model.evaluate(xTest, yTest, batch_size=32)

#


