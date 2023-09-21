# 该工具类用于实现滤波器功能，主要使用巴特沃斯滤波。

import numpy as np

import power as p
import view as v
import utility_functions as uf
from scipy import signal
from matplotlib import pyplot as plt

# 参数为待处理数据，低截止、高截止、采样率、阶数、是否画图。低截止 <= 0 表示低通，高截止 <= 0 表示高通。

# 【非法倍数阈值】若滤波后，数据出现超过原始数据的该倍数，则认为没有滤波成功。


invalid_th = 10 ** 2


# 【非法数据判断】
# data_ 表示处理后的数据
# data 表示原始数据
# return False 表示不合法，return True 表示合法
def data_invalid(data_, data):
    if np.isnan(data_).any():
        return False
    for i in range(0, len(data)):
        if abs(data_[i]) > abs(data[i]) * invalid_th and data_[i] != 0:
            # 同时还要排除零值
            return False
        return True


# 【巴特沃斯滤波器】
# low_cut <= 0 表示低通滤波，此时 high_cut 的值表示截止频率
# high_cut <= 0 表示高通滤波，此时 low_cut 的值表示截止频率
# low_cut > high_cut 表示带阻滤波，此时两个参数分别为两边的截止频率
def butter_filter(data, low_cut, high_cut, fs=44100, n=-1, draw=False):
    # 滤波器的阶数不能超过带宽，也就是说可以阶数可以自适应。
    data = np.array(data)
    # 滤波器只能接收整型数据
    data = np.int32(data)
    data_ = np.array([np.nan])
    # 归一化的截止频率
    w, h, b, a = 0, 0, 0, 0
    if low_cut <= 0 and high_cut <= 0 or low_cut == high_cut:
        print('截止频率参数错误')
        return data
    low_cut_ = 2 * low_cut / fs
    high_cut_ = 2 * high_cut / fs
    if low_cut <= 0:
        # 初始化默认阶数
        if n <= -1:
            n = np.rint(high_cut / 2)
        # 低通滤波
        while not data_invalid(data_, data):
            # 模拟滤波器仅用于画图
            b, a = signal.butter(n, high_cut, 'low', analog=True)
            if np.isnan(a).any() or np.isinf(a).any():
                n -= 1
                continue
            w, h = signal.freqs(b, a)
            # 默认是数字滤波器，Wn 介于 [0, 1] 之间。即归一化的截止频率。
            b, a = signal.butter(n, high_cut_, 'low')
            if n == 0:
                # 若 n 为 0 阶，则返回原始数据
                print('零阶滤波器，已返回原始信号。')
                return data
            data_ = signal.filtfilt(b, a, data, axis=0)
            n -= 1
    elif high_cut <= 0:
        # 初始化默认阶数
        if n <= -1:
            n = np.rint(low_cut / 2)
        # 高通滤波
        while not data_invalid(data_, data):
            b, a = signal.butter(n, low_cut, 'high', analog=True)
            if np.isnan(a).any() or np.isinf(a).any():
                n -= 1
                continue
            w, h = signal.freqs(b, a)
            b, a = signal.butter(n, low_cut_, 'high')
            # 这一行会报错 ValueError: The length of `a` must be at least 2.
            # 原因是 n = 0
            if n == 0:
                # 若 n 为 0 阶，则返回原始数据
                print('零阶滤波器，已返回原始信号。')
                return data
            data_ = signal.filtfilt(b, a, data, axis=0)
            n -= 1
    elif low_cut > high_cut:
        if n <= -1:
            n = np.rint(low_cut - high_cut)
        # 带阻滤波
        while not data_invalid(data_, data):
            b, a = signal.butter(n, [high_cut, low_cut], 'stop', analog=True)
            if np.isnan(a).any() or np.isinf(a).any():
                n -= 1
                continue
            print(b)
            print(a)
            w, h = signal.freqs(b, a)
            b, a = signal.butter(n, [high_cut_, low_cut_], 'stop')
            if n == 0:
                # 若 n 为 0 阶，则返回原始数据
                print('零阶滤波器，已返回原始信号。')
                return data
            data_ = signal.filtfilt(b, a, data, axis=0)
            n -= 1
    else:
        if n <= -1:
            n = np.rint(high_cut - low_cut)
        # 带通滤波
        while not data_invalid(data_, data):
            b, a = signal.butter(n, (low_cut, high_cut), 'bandpass', analog=True)
            # 校验是否为无效值
            if np.isnan(a).any() or np.isinf(a).any():
                n -= 1
                continue
            w, h = signal.freqs(b, a)
            b, a = signal.butter(n, (low_cut_, high_cut_), 'bandpass')
            if n == 0:
                # 若 n 为 0 阶，则返回原始数据
                print('零阶滤波器，已返回原始信号。')
                return data
            data_ = signal.filtfilt(b, a, data, axis=0, padlen=len(data) - 1)
            n -= 1
    if draw and type(b) != int and type(a) != int:
        print('滤波器阶数为：')
        print(n + 1)
        plt.figure('butter filter')
        plt.title('butter filter')
        plt.plot(w, abs(h))
        plt.show()
    return data_


# 获取最值池化(最值降采样)
def get_max_pling(data, wnd=1):
    print('功能尚待开发')
    return


# 【获取平滑卷积】通过滑动窗口，获得每一段窗口的均值，返回一个均值数组
# wnd 表示窗口大小，默认 wnd = 1，表示直接返回原始数据
def get_conv_smooth(data, wnd=1):
    if wnd <= 0:
        print("窗口值错误！已设置窗口值为 1")
        wnd = 1
    data = np.array(data, dtype=float)
    data_mean = np.array([], dtype=float)
    if len(data) >= 1e5:
        print("正在计算平滑卷积：")
    for i in range(0, len(data) - wnd):
        if len(data) >= 1e5:
            v.show_progress(i, len(data) - wnd)
        data_mean = np.append(data_mean, np.mean(data[i:i + wnd]))
    v.show_progress()
    return data_mean


# 【获取信号的平滑率】值介于 [0, 1] 之间，越接近 1 越平滑。
# data 为一维时域信号
# 返回值分别为平滑度和极值点坐标集合
def get_smooth_deg(data):
    if len(data) <= 3:
        print("输入数据太短，无法计算平滑度.")
        return 0, np.array([])
    data = np.array(data, dtype=float)
    diff = np.diff(data)
    zero_num = 0
    extrem_idx = np.array([], dtype=int)
    for i in range(0, len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            extrem_idx = np.append(extrem_idx, i)
            zero_num += 1
    smooth_deg = np.around(1 - (zero_num / len(data)), 6)
    return smooth_deg, extrem_idx


# 【升采样平滑】
# sub_dg 表示升采样程度
def get_up_sampling_smooth(x, y, sub_dg=100):
    if not uf.is_ary(x) or not uf.is_ary(y):
        print("get_up_sampling_smooth 错误！x，y 必须为数组！")
        return [], []
    if len(x) != len(y):
        print("get_up_sampling_smooth 错误！x，y 不等长！")
        return [], []
    smooth_wnd = int(sub_dg / 2)
    x_ = np.array([], dtype=float)
    y_ = np.array([], dtype=float)
    for i in range(len(x) - 1):
        pre_idx = i
        next_idx = i + 1
        x_local, y_local = uf.divide_2points((x[pre_idx], y[pre_idx]), (x[next_idx], y[next_idx]), sub_dg)
        x_ = np.append(x_, x_local)
        y_ = np.append(y_, y_local)
    x_ = np.append(x_, x[-1])
    y_ = np.append(y_, y[-1])
    x_ = x_[0:-smooth_wnd]
    y_ = get_conv_smooth(y_, smooth_wnd)
    return x_, y_


# 【获取一定平滑度下的所有极值点索引】
# data 表示原始数据
# smooth_th 表示平滑门限值，当平滑度大于该值后迭代结束。
# wnd 表示初始窗口大小
def get_end_idx(data, smooth_th=0.99, wnd=2):
    data = np.array(data, dtype=float)
    mean_extrem_idx = np.array([], dtype=int)
    end_idx = np.array([0], dtype=int)
    smooth_deg = 0.
    last_smooth_deg = 0.
    step = 1
    print("开始迭代平滑卷积：")
    while smooth_deg < smooth_th:
        mean = get_conv_smooth(data, wnd)
        smooth_deg, mean_extrem_idx = get_smooth_deg(mean)
        if smooth_deg <= last_smooth_deg:
            step = 1
            wnd += step
        else:
            # 差越小，步进越大
            step = int(1 / 2 / uf.sigmoid(smooth_deg - last_smooth_deg, 0.375, 0.106032))
            if step <= 0:
                step = 1
            wnd += step
            pass
        last_smooth_deg = smooth_deg
        v.show_progress(smooth_deg, smooth_th)
    wnd -= step
    v.show_progress(smooth_deg, smooth_th)
    # 换行显示
    print()
    print("迭代结束，窗口大小为：", end="")
    print(wnd)
    end_idx = np.append(end_idx, mean_extrem_idx + int(wnd / 2))
    end_idx = np.append(end_idx, len(data) - 1)
    return end_idx


"""
注意:
模糊极值点的获取方法重点是降采样
而降采样还可以使用均值池化,最大值池化等方法
"""


# × 暂不可用
# 【离散滤波器】通过向信号中添加调整过幅频相的正弦来实现滤波
# def disc_filter(data, fs, low_cut, high_cut):
#     data = np.array(data, dtype=float)
#     norm_am, freq = sv.get_fft_result(data, fs)
#     freq, cut_l_idx, cut_r_idx = sv.get_freq_range(freq, low_cut, high_cut)
#     x = np.arange(0, len(data))
#     power = p.get_freq_power(data, low_cut, high_cut, fs)
#     # 遍历频率区间
#
#     while power > 500:
#         # print('freq 长度：')
#         # print(len(freq))
#         for i in range(0, len(freq)):
#             am = norm_am[cut_l_idx + i]
#             data = data - algo.sin(am, freq[i], x, fs)
#             # 重新获取调整后的频率，且要根据频率进行截取。
#             norm_am, freq = sv.get_fft_result(data, fs)
#             freq, cut_l_idx, cut_r_idx = sv.get_freq_range(freq, low_cut, high_cut)
#             # 若减去了反向的频率，那么再重新加回来
#             if norm_am[cut_l_idx + i] > 1.5 * am:
#                 # print('出现了反向的情况')
#                 data = data + 2 * algo.sin(am, freq[i], x, fs)
#         power = p.get_freq_power(data, low_cut, high_cut, fs)
#         print('本次迭代的功率为：')
#         print(power)
#     return data


# 【小波滤波器】通过离散小波变换来进行滤波
def dwt_filter(data, low_cut, high_cut, fs=44100):
    # 频率最大值
    freq_max = int(fs / 2)
    # 频率区间
    freq_inr = np.array([0, freq_max])
    # 各区间的 dwt 数组，dwt 的结果是 ndarray 类型
    dwt_r = []
    # 插入值的方法：np.insert(ndarray, index, elements, axis)
    """
        这个循环怎么设计？
        先算分解后的频率范围
        算一个lowcut或highcut到
    """
    while True:
        break
    return


"""
现在任务：
先看看平滑滤波后的频谱

"""

"""
获得指定频率区间的 dwt 数组
查看、调整、复原
两个数据结构：
1. 频率区间数组
2. 滤波后数组的数
"""

"""
怎么获得目标频率的 sin 信号？
假设目标频率是 obj_freq
使用 y = np.sin(obj_freq*2*np.pi*x)
但是输入的自变量 x 有要求，必须将步长转换成时间步长。
时间步长 t_step = 1/fs
那么 x = x*t_step

但是这个滤波有个问题：
有可能进行了反向滤波
因为当 y=-sin 的时候，频谱中依然会得到相同的结果
所以滤波后需要进行检验，该频率是否增大了指定振幅
"""


# 【均值池化滤波器】
# 支持窗口为小数值，wnd 默认是 1，等同于不进行滤波。
# 变频功能：窗口大小即为变频倍数。例如 5Hz 变为 6Hz，则 wnd=1.2
def get_mean_pooling(data, wnd=1.):
    if not uf.is_ary(data):
        print("get_mean_pooling 错误！请输入数组参数！")
        return data
    if len(data) <= wnd:
        print("get_mean_pooling 错误！数组长度短于窗口长度！")
        return data
    data = np.array(data, dtype=float)
    if uf.is_flt(wnd):
        wnd_dec = wnd % 1
    else:
        wnd_dec = 0
    wnd_int = int(wnd)
    data_ = np.array([], dtype=float)
    if len(data) > 1e5:
        print("正在执行均值池化（变频）：")
    if wnd_dec:
        # 窗口为小数的情况
        i = 0
        while i < len(data) - wnd:
            if len(data) > 1e5:
                v.show_progress(i, len(data) - wnd)

            pre_edge = i
            next_edge = i + wnd

            if next_edge > len(data) - 1:
                break
            pre_prop = pre_edge % 1
            pre_idx = int(np.ceil(pre_edge))
            next_prop = next_edge % 1
            next_idx = int(next_edge)

            # 前小数
            if pre_prop:
                sum_arr = [uf.get_prop_mean(data[pre_idx - 1], data[pre_idx], pre_prop)]
            else:
                sum_arr = []

            # 中整数
            sum_arr = np.append(sum_arr, data[pre_idx:next_idx + 1])

            # 后小数
            if next_prop:
                sum_arr = np.append(sum_arr, uf.get_prop_mean(data[next_idx], data[next_idx + 1], next_prop))
            data_ = np.append(data_, np.mean(sum_arr))
            i += wnd
    else:
        # 窗口为整数的情况
        i = 0
        while i < len(data) - wnd_int:
            data_ = np.append(data_, np.mean(data[i:i + wnd_int]))
            i += wnd_int
    if len(data) > 1e5:
        v.show_progress()
    return data_


'''
有两种情况，滤波器需要降阶：
1. 滤波后数据出现 NaN
2. 滤波后数据出现大于一个数量级（100 倍）
'''
# 参考网址：
# https://blog.csdn.net/aaalswaaa1/article/details/120750826
# https://www.cnpython.com/qa/1205339
