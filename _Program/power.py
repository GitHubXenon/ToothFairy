# 该代码实现对一个或一组 wav 的功率调整，可以分左右声道调整。
import numpy as np
import filter as f
import view as v
import utility_functions as uf

# 迭代结束的倍数阈值区间（差距在该倍数区间以内时结束迭代）
mpl_th = [0.99, 1.01]


# 【单声道功率计算】
# 该函数可以在其他模块中调用（输入的 data 必须为单声道）
def get_power(data):
    data = np.array(data, dtype=float)
    return np.sum(data ** 2) / len(data)


# 【单声道频率范围功率计算】
# low_cut <= 0 表示低通功率，此时 high_cut 的值表示截止频率
# high_cut <= 0 表示高通功率，此时 low_cut 的值表示截止频率
# low_cut > high_cut 表示带阻功率，此时两个参数分别为两边的截止频率
def get_freq_power_by_filter(data, low_cut=-1, high_cut=-1, fs=44100):
    # 先带通滤波，再计算功率
    data = np.array(data, dtype=float)
    if low_cut <= 0 and high_cut <= 0 or low_cut == high_cut:
        print("输入频率范围错误，已返回全频域功率。")
        return get_power(data)
    if low_cut <= 0:
        low_cut = -1
    elif high_cut <= 0:
        high_cut = -1
    n = np.rint(abs((high_cut - low_cut) / 2))
    # 滤波不对
    return get_power(f.butter_filter(data, low_cut, high_cut, fs, n))


# 通过频谱计算功率
def get_freq_power_by_spectrum(data, low_cut, high_cut, fs=44100):
    am, freq, phi = v.get_fft_result(data, fs)
    low_idx = uf.get_nearest_idx(freq, low_cut)
    high_idx = uf.get_nearest_idx(freq, high_cut)
    # 注意这里有个除 2！解释如下
    return np.sum(am[low_idx:high_idx + 1] ** 2) / 2


"""
频谱与功率谱的关系：
FFT 计算频谱是 fft(s)/N
功率谱是 S=[fft(s)/N]^2
那么通过功率谱计算信号功率就是 P=sum(abs(S))

这里为什么要除以 2？
原计算公式中，FFT 结果不取半，那么某频率的功率就是 a^2 + a^2 = 2a^2
取半后，直接算平方和就是 (2a)^2 = 4a^2
"""


def get_am_by_power(power):
    return


# 【平均振幅调整】
def am_adjust():
    return


# 【全信号功率调整】
# data 必须为一维数组，即单声道数据
# object_power 表示要调整的目标功率
def power_adjust(data, object_power):
    # if object_power == 0:
    #     print("错误！目标功率为 0！已返回原始值")
    #     return np.int32(np.rint(data))
    # 封装为 ndarray
    data = np.array(data, dtype=float)
    # 初始倍数
    multiple = get_power(data) / object_power
    counter = 0
    while not mpl_th[0] < multiple < mpl_th[1]:
        # 计算功率倍率
        if multiple == 0:
            print("功率调整中出现除零错误！")
            return
        data = data / np.sqrt(multiple)
        multiple = get_power(data) / object_power
        counter += 1
        if counter > 100:
            break
    return np.int32(np.rint(data))


# 【某频率范围功率调整】
# data 必须为一维数组，即单声道数据
# object_power 表示要调整的目标功率
# low_cut, high_cut 表示截止频率
# rounding 表示是否取整？默认是 False，返回浮点数，能够进行高精度运算。
# keep_origin_power 表示是否让信号保持原始功率（局部功率调整会导致整段信号的功率都发生变化）
def freq_power_adjust(data, obj_power, low_cut, high_cut, fs=44100, rounding=False, keep_origin_power=False):
    """
        调整方案：
        0. 先判断输入参数合法性。
        1. 先带通滤波，获得 data_。
        2. 将 data_ 进行功率调整。
        3. 再对同样的位置进行带阻滤波（low_cut 和 high_cut 交换位置），并相加到 data_。
        4. 计算相加后的指定频带功率值，通过倍数阈值来判断是否需要重新迭代。
        4. 根据需要重新调整整体功率。
    """
    data = np.array(data, dtype=float)
    data_ = data
    multiple = 100
    if obj_power <= 0:
        print("目标功率参数不正确，已返回原始数据。")
        return data
    if low_cut <= 0 and high_cut <= 0:
        print("截止频率输入不正确，已返回原始数据。")
        return data
    if low_cut <= 0:
        low_cut = -1
    elif high_cut <= 0:
        high_cut = -1
    # while not multiple_th[0] < multiple < multiple_th[1]:
    # 获取滤波后的数据，滤两遍
    data_ = f.butter_filter(f.butter_filter(data, low_cut, high_cut, fs), low_cut, high_cut, fs)
    # 对滤波后的数据进行功率调整
    data_ = power_adjust(data_, obj_power)
    # 将带通调整后的数据和带阻数据相加
    # 这个带阻信号，有可能反相？不可能，因为不是同频信号。
    data_ = data_ + f.butter_filter(f.butter_filter(data, high_cut, low_cut, fs), high_cut, low_cut, fs)
    # print("滤波后的功率：")
    # print(get_power(data_))
    # print("相加后的总功率：")
    # print(get_power(data_))
    # print("相加后，指定频带功率：")
    # print(get_freq_power(data_, low_cut, high_cut))
    """
        迭代过程中，是需要增加带宽还是增加其他信息？
    """
    if keep_origin_power:
        data_ = power_adjust(data_, get_power(data))
    if rounding:
        data_ = np.int32(np.rint(data_))
    return data_


# 【频率成分的功率按比例调整】
def freq_proportion_adjust(data, object_power, low_cut, high_cut):
    print("功能尚未开发，敬请期待...")
    return


# 【正弦修正器】暂时没写完
# 利用正弦函数和滑动窗口对信号进行功率调整
def sin_power_adjust(data, obj_power, freq, fs=44100):
    # time = np.arange(0, len(data)) / fs
    # wnd_len = algo.get_nearest_idx(time, 2 / freq)
    # start_idx = 0
    # end_idx = start_idx + wnd_len
    # low_cut = freq - 20
    # high_cut = freq + 20
    # data_ = f.butter_filter(data, low_cut, high_cut, fs)
    # data_ = power_adjust(data_, obj_power)
    # while True:
    #     x = range(start_idx, end_idx)
    #     phi = algo.get_nearest_phi(data[start_idx:end_idx], am, freq, fs, True)
    #     y = algo.sin(x, am, freq, phi, fs)
    #     data[start_idx:end_idx] = data[start_idx:end_idx] - y
    #     start_idx += wnd_len
    #     end_idx += wnd_len
    #     if end_idx >= len(data) + wnd_len:
    #         break
    #     elif end_idx > len(data):
    #         end_idx = len(data)
    # return data_
    pass


'''
类型转换问题：
不能通过 dtype 属性来转换类型，dtype 属性不能随便乱用！
转换类型必须要通过 np.int64()、np.float64() 等函数来执行
Numpy 根本不存在浮点数溢出的问题
'''
