import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Load wave file
wave_file = wave.open(r"/Users/wangyang/Library/CloudStorage/OneDrive-个人/_数据/【右下】/L1.wav", "rb")

# Get wave parameters
nchannels = wave_file.getnchannels()
sampwidth = wave_file.getsampwidth()
fs = wave_file.getframerate()
nframes = wave_file.getnframes()

# Read wave data
wave_data = wave_file.readframes(nframes)
wave_data = np.frombuffer(wave_data, dtype=np.int16)

# Close wave file
wave_file.close()

# Split channels
if nchannels == 2:
    channel1 = wave_data[0::2]
    channel2 = wave_data[1::2]
else:
    print("Error: wave file must have two channels")

# Plot signal series of the two channels
# time = np.arange(0, nframes / fs, 1 / fs)
# matplotlib.use('TkAgg')
# plt.figure()
# plt.plot(time, channel1, label='Channel 1')
# plt.plot(time, channel2, label='Channel 2')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.show()

# 一维数组，长度 84480
# print(len(channel1), "   ", channel1)
# print(len(channel2), "   ", channel2)

# Calculate correlation coefficient
correlation = np.corrcoef(channel1, channel2)[0, 1]
# print("Correlation coefficient between channel 1 and channel 2:", correlation)

# Calculate correlation coefficient for a window of 4410
loc = 0
win = 4410
profile = []
for i in range(20):
    # print(i)
    left_idx = int((i + 1 - 1) * win / 2)
    right_idx = int((i + 1 + 1) * win / 2)

    # 切割序列
    Sl_ = channel1[loc + left_idx:loc + right_idx]
    Sr_ = channel2[loc + left_idx:loc + right_idx + win]
    corr_sequence = []
    # 先将 corr 的序列计算出来，再从中找到 argmax
    for k in range(len(Sl_)):
        # 计算 Sl[0:end] 及 Sr[k:end] 之间的相关系数
        # pearson 相关
        correlation = np.corrcoef(Sl_, Sr_[k:k + win])[0, 1]
        # print(correlation)
        # corr = Sl_.corr(Sr_[k:], "pearson")
        # 这样 k 值，就等于对应计算结果的索引值（corr_sequence 中的索引值）
        corr_sequence.append(correlation)

    print(corr_sequence)
    DeltaT_slide = np.argmax(corr_sequence)
    print(DeltaT_slide, ":", corr_sequence[DeltaT_slide])
    profile.append(DeltaT_slide)

print(profile)
print(np.mean(profile))
