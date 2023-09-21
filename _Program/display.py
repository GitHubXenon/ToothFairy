# 实时显示信号的时域和频域，不做具体位置判断！

import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import random_tools as rt
import view as v
import tooth_research as tr
from scipy.io import wavfile
import utility_functions as uf
from fractions import Fraction

# 刷新时间间隔，60 FPS = 0.0166，所以应当比这个数小。
# interval = Fraction(1, 100)
interval = 1 / 100

src_path = r"F:\python生成的数据\real_time.wav"
obj_curve_path = r"F:\python生成的数据\curve.wav"

# src_path = r"/Volumes/UGREEN/real_time.wav"
# obj_curve_path = r"/Volumes/UGREEN/curve.wav"

# fs, signal = wavfile.read(obj_curve_path)
fs, signal = wavfile.read(src_path)

sd_flag = False

if sd_flag:
    # 静态全局显示
    v.show_am_time(signal[..., 0])
    v.show_am_freq(signal[..., 0])
    v.show_am_time(signal[..., 1])
    v.show_am_freq(signal[..., 1])
else:
    """
    MacOS 复制路径快捷键 ⌥⌘C option + command + c
    获取数据 rt.get_rand_len_signal()
    窗口分为两种：
    一种是调整窗口，一种是显示窗口。
    为了使得信号变化显示得较为平滑，需要有较高的调整粒度。
    换句话说，至少调整窗口要比显示窗口小。
    """

    # 窗口起始索引
    wnd_start = 0
    # 窗口结束索引
    wnd_end = 0
    # 识别窗口长度，单位采样点，窗口长度 0.1s，是刷新速率的 10 倍。
    wnd_len = uf.time2samp(interval * 10, fs)
    print("窗口长度：", wnd_len)
    data_wnd = []
    horiz_wnd = []

    # 查看位置标识符
    # 查看左右声道，True 表示左声道
    lr_flag = False
    # 查看时频域，True 表示时域
    tf_flag = False

    print("预处理数据...")

    # 先排除一侧声道
    if lr_flag:
        signal = signal[..., 0]
    else:
        signal = signal[..., 1]

    # 先将数据写入数组，再在后面绘制图形。
    if tf_flag:
        """
        对于时域信号：
        将其分解为二维数组
        但是显示的时候是一次显示三个窗口。
        """
        for i in range(int((len(signal) - 1) / wnd_len)):
            # i 每次移动一个窗口的长度
            wnd_start = i * wnd_len
            wnd_end = (i + 1) * wnd_len
            data_wnd.append(signal[wnd_start:wnd_end])
            # 横轴单位是秒，有时候会出来 4411 的长度。小数步进不定长问题。
            # 先简单处理下，回来再研究。
            horiz = np.arange(uf.samp2time(wnd_start), uf.samp2time(wnd_end), np.around(interval * 10 / wnd_len, 10))
            if len(horiz) > 4410:
                horiz = horiz[:-1]
            horiz_wnd.append(horiz)
            # @TODO 出现了超出的部分
            if len(horiz_wnd[-1]) > 4410:
                print("超出显示：")
                print(len(horiz_wnd[-1]))
                print(uf.samp2time(wnd_start))
                print(uf.samp2time(wnd_end))
                print(interval * 10 / wnd_len)
                print(interval * 10 / wnd_len * 1e13)
                # 输出 2.2675736961451248e-05，也就是说，显示出来已经是极限精度了。
                # 输出 226757369.61451247
    else:
        # 显示频域
        for i in range(int((len(signal) - 1) / wnd_len)):
            wnd_start = i * wnd_len
            wnd_end = (i + 1) * wnd_len
            am, freq, phi = v.get_fft_result(signal[wnd_start:wnd_end], fs)
            data_wnd.append(am)
            # 此时横轴表示频率
            horiz_wnd.append(freq)

    print("数据处理完成。")

    # 开启 interactive mode，才能实时显示。
    plt.ion()
    plt.figure("real time display", (4, 2.5))

    # 时间轴，长度应当为常量，显示最新 2s 内的信号。实际上就是一个时间窗口。
    # time_axis = [0]
    # t_now = 0
    # m = [sin(t_now)]

    # for i in range(2000):
    #     # 清空画布上所有内容
    #     plt.clf()
    #     # 随着循环增加，这个数从 0 变化到 200
    #     t_now = i * 0.1
    #     # 模拟数据增量流入，保存历史数据
    #     t.append(t_now)
    #     m.append(sin(t_now))  # 模拟数据增量流入，保存历史数据
    #     plt.plot(t, m, '-r')  # -r 意思是红色线
    #     # 方法一：使用 draw 函数，一般适用于 Ubuntu
    #     # plt.draw()
    #     # time.sleep(0.01)
    #     # 方法二：如果出不了动态效果，就将上两行替换为，一般适用于 MacOS 或 Windows
    #     plt.pause(0.01)

    if tf_flag:

        # 绘制时域图，三个识别窗口合并作为一个显示窗口。
        for i in range(len(data_wnd)):
            # wnd_start = i * wnd_len
            # wnd_end = (i + 1) * wnd_len
            # data_wnd = signal[wnd_start:wnd_end]
            # time = np.arange(wnd_start / fs, wnd_end / fs, wnd_len)
            plt.clf()
            if tf_flag:
                plt.plot(horiz_wnd[i], data_wnd[i], c="orange")
            else:
                plt.plot(horiz_wnd[i], data_wnd[i], c="blue")
            plt.ylim((-230, 230))
            plt.pause(interval)
    else:
        # 绘制频域图
        for i in range(int((len(signal) - 1) / wnd_len)):
            plt.clf()
            if tf_flag:
                plt.plot(horiz_wnd[i], data_wnd[i], c="orange")
            else:
                plt.plot(horiz_wnd[i], data_wnd[i], c="blue")
            plt.ylim((0, 200))
            plt.xlim((0, 1000))
            plt.pause(interval)
