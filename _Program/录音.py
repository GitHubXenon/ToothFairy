import numpy as np
from matplotlib import pyplot as plt
import pyaudio
import wave

CHUNK = 1024  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 量化位数
fs = 44100
# 持续时长
duration = 20
# 声道数
channels = 1
n = duration * fs
t = np.arange(1, n) / fs
wave_output_file = 'record.wav'
print('这段音频有几秒：', duration)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=channels, rate=fs,
                input=True, frames_per_buffer=CHUNK)
print('开始录制：')

frames = []
for i in range(0, int(fs / CHUNK * duration)):
    data = stream.read(CHUNK)
    frames.append(data)

print('录制结束')
stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(wave_output_file, 'wb')  # 打开这个文件，以二进制写入的方式
wf.setnchannels(channels)  # 设置声道
wf.setsampwidth(p.get_sample_size(FORMAT))  # 设置采样位宽
wf.setframerate(fs)  # 设置采样率
wf.writeframes(b''.join(frames))  # 把所有的帧连成一段语音
wf.close()
