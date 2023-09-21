import numpy as np
from sklearn import svm
from scipy.optimize import curve_fit

import view as v
from scipy.io import wavfile
import os
import filter as f
from decimal import Decimal
import types
from scipy.optimize import curve_fit
import random_tools as rt
from scipy.optimize import fsolve
import sympy
from sympy.abc import p, m, n
import matplotlib.pyplot as plt
from random import sample
import pandas as pd


# 文本文字颜色
class FontColor:
    PINK_HEADER = '\033[95m'  # pink
    BLUE_OK = '\033[94m'  # blue
    GREEN_OK = '\033[92m'  # green
    YELLOW_WARNING = '\033[93m'  # yellow
    RED_FAIL = '\033[91m'  # red
    BLACK_NORMAL = '\033[0m'  # black
    BLACK_BOLD = '\033[1m'  # black + bold
    BLACK_UNDERLINE = '\033[4m'  # black + underline


class Signal:
    # label 用于数据分类用，作为可选项
    def __init__(self, data_1, data_2, fs, label=0):
        self.data_1 = data_1
        self.data_2 = data_2
        self.fs = fs
        self.label = label


# 读取文件并转换为 Signal 对象
def file2signal(path, label=0):
    fs, signal = wavfile.read(path)
    return Signal(signal[..., 0], signal[..., 1], fs, label)


# 将文件夹读入为 signal_list
def dir2_signal_list(dir_path, label=0):
    signal_list = []
    if not os.path.exists(dir_path):
        print('路径不存在！')
        return None
    file_list = os.listdir(dir_path)
    print('正在执行数据读入（共找到', len(file_list), '个文件）')
    i = 0
    for f_path in file_list:
        v.show_progress(i, len(file_list))
        signal_list.append(file2signal(dir_path + '/' + f_path, label))
        i += 1
    v.show_progress(1, 1)
    return signal_list


# 【列表转字符串】
def list2str(_list):
    _str = '['
    for i in range(0, len(_list)):
        _str += str(_list[i])
        if i != len(_list) - 1:
            _str += ', '
    _str += ']'
    return _str


# 【矩阵转字符串】
def mat2str(mat):
    _str = '['
    for i in range(0, len(mat)):
        _str += list2str(mat[i])
        if i != len(mat) - 1:
            _str += ', '
    _str += ']'
    return _str


# 【数字字符串转列表字符串】
def num_list2str_list(_list):
    str_list = []
    for num in _list:
        str_list.append(str(num))
    return str_list


# 代码示例
# print(FontColor.PINK_HEADER)  # 此后打印的字体都是粉色
# print(FontColor.BLACK_NORMAL) # 此后打印的字体都是常规黑色

# 获取随机扭曲函数
# 单调且凹凸区间有限的
# samp_num 表示采样个数
# cvx_num 表示凸点
# def get_rand_distort_curve(p1=(0, 0), p2=(1, 1), samp_num=100, cvx_num=2):
#     """
#     :param p1:
#     :param p2:
#     :param samp_num:
#     :param cvx_num:
#     :return:
#     算法思想：
#     先分割凹凸区间点，在中间随机取点，且保证满足凹凸一致原则。
#     不断细分下去。不需要曲线拟合。
#     """
#
#     # mid_x 表示凹凸区间的分割点
#     if p1[0] > p2[0]:
#         mid_x = rt.mean_rand([0] * (cvx_num - 0), p2[0], p1[0])
#     else:
#         mid_x = rt.mean_rand([0 * (cvx_num - 0)], p1[0], p2[0])
#     # 在以下算法中，mid_x 和 mid_y 表示曲线细分算法。
#     mid_x = np.sort(mid_x)
#     mid_x = np.insert(mid_x, 0, p1[0])
#     mid_x = np.append(mid_x, p2[0])
#     mid_y = []
#     line = get_line(p1, p2)
#     for j in range(1, len(mid_x) - 1):
#         height = abs(line(mid_x[j - 1]) - line(mid_x[j + 1]))
#         mid_y.append(rt.mean_rand(line(mid_x[j]), min_val=-height / 2, max_val=height / 2))
#
#     mid_y = np.insert(mid_y, 0, p1[1])
#     mid_y = np.append(mid_y, p2[1])
#
#     k = 0
#     insert_sum = 0
#     for j in range(len(mid_x) - 1):
#         diff_y = np.diff(mid_y)
#         # 凸
#         if diff_y[k] > diff_y[k + 1]:
#             mid_p = get_mid_point((mid_x[k], mid_y[k]), (mid_x[k + 1], mid_y[k + 1]), trans_type=TRANS_CONVEXITY)
#         else:
#             mid_p = get_mid_point((mid_x[k], mid_y[k]), (mid_x[k + 1], mid_y[k + 1]), trans_type=TRANS_CONCAVE)
#         mid_x = np.insert(mid_x, k, )
#         # 不能用 j 的原因是 mid_y 和 mid_x 在不断变化
#         insert_sum += 1
#         k += insert_sum + 1
#
#     return


# 上凸
TRANS_CONVEXITY = 1
# 下凹
TRANS_CONCAVE = 2
# 直线
TRANS_LINEAR = 3


# 获取随机中间点
# rand 表示随机倍数，若 rand = 0 表示无随机。
def get_mid_point(p1=(0, 0), p2=(1, 1), trans_type=TRANS_LINEAR):
    y_mid = (p1[1] + p2[1]) / 2
    y_height = abs(p1[1] - p2[1])
    # if p1[1] > p2[1]:
    #     min_val = y_mid - p1[1]
    #     max_val = p2[1] - y_mid
    # else:
    #     min_val = y_mid - p1[1]
    #     max_val = p2[1] - y_mid
    if trans_type == TRANS_LINEAR:
        p_mid = [(p1[0] + p2[0]) / 2, rt.mean_rand(y_mid, min_val=-y_height / 2, max_val=y_height / 2)]
    elif trans_type == TRANS_CONVEXITY:
        p_mid = [(p1[0] + p2[0]) / 2, rt.mean_rand(y_mid, min_val=0, max_val=y_height / 2)]
    elif trans_type == TRANS_CONCAVE:
        p_mid = [(p1[0] + p2[0]) / 2, rt.mean_rand(y_mid, min_val=-y_height / 2, max_val=0)]
    else:
        p_mid = [(p1[0] + p2[0]) / 2, y_mid]
    return p_mid


# 过渡函数
# steep_dg 表示陡峭程度，越大表示越直
def get_trans_func(p1=(0, 0), p2=(1, 1), trans_type=TRANS_LINEAR, steep_dg=100):
    k = (p2[1] - p1[1]) / (p2[0] - p1[0]) / steep_dg
    if trans_type == TRANS_LINEAR:
        def linear(x):
            # 两点式
            return k * (x - p1[0]) + p1[1]

        return linear
    # 需要分为单增单减两种情况
    # 务必保证首尾两个点是端点值。
    # 保证平滑端接近于 0，例如 y'(p2[1]) = 0.05
    if (k > 0 and trans_type == TRANS_CONVEXITY) or (k < 0 and trans_type == TRANS_CONCAVE):
        # 单增凸型、单减凹型：平直点在后
        p_end = p2
    elif (k < 0 and trans_type == TRANS_CONVEXITY) or (k > 0 and trans_type == TRANS_CONCAVE):
        # 单减凸型、单增凹型：平直点在前
        p_end = p1
    else:
        p_end = (0, 0)
        print("警告！未找到对应的函数曲线。")

    # 解方程
    p_opt = sympy.solve([p / (p1[0] + m) + n - p1[1],
                         p / (p2[0] + m) + n - p2[1],
                         -p / (p_end[0] + m) ** 2 - k])

    # 注意：得到的 dict 只能用标识符取值，不能用字符串。
    # 返回的函数应当改为常量，不能继续用 sympy 对象。（命名空间原因）
    p_val = float(p_opt[0][p])
    m_val = float(p_opt[0][m])
    n_val = float(p_opt[0][n])

    def curve(x):
        return p_val / (x + m_val) + n_val

    return curve


# 时间单位转为采样点单位
def time2samp(t, fs=44100):
    return int(t * fs)


# 采样单位转为时间单位
def samp2time(x, fs=44100):
    return x / fs


# 小数步进不定长问题解决方案
def get_range(start, end, step):
    """
    【小数步进不定长问题解决方案】
    :param start:
    :param end:
    :param step:
    :return:
    为什么会出现不定长问题？
    举例：
    print(len(np.arange(1, 4, 1)))              长度为 3
    print(len(np.arange(1, 4, 0.9999999999)))   长度为 4
    也就是说，在整数情况下，终点与尾元素之间，应当相差正好一个步长。
    例如尾元素是 3，终点是 4，应当相差 1. 这就导致了数组步进 1 是取不到终点值的。
    由于无法整除等原因，步长可能是无限小数，该数必然比分数实际值要小，导致多取了一个。
    判断是否会出现不定长的方法是：判断步长是否为无限小数。
    """
    return


# 用于调整一组采样点中，导数剧烈变化的情况
# 还没写完
def samp_derivative_adjust(x, y):
    if len(x) != len(y):
        print("错误！x 和 y 长度不一致！")
        print("x 长度：", len(x))
        print("y 长度：", len(y))
        return None

    d = np.diff(x) / np.diff(y)
    plt.figure(0, (3, 3.5))
    plt.rc("font", family="Times New Roman", size=12)
    plt.plot(d)
    plt.show()

    dd = np.diff(d)
    plt.figure(0, (3, 3.5))
    plt.rc("font", family="Times New Roman", size=12)
    plt.plot(dd)
    plt.show()

    # 1, 2, 3, 4 ｜ d1 表示 12 ｜ dd1 表示 123
    # -5 +4 -3

    # for j in range(len(dd) - 2):
    # 中间点同时与两侧点异号，那么该点的斜率改为两侧斜率的平均值。
    # if dd[j + 1] * dd[j] < 0 and dd[j + 1] * dd[j + 2]:
    # j = 0 时 12 之间的值到 23 之间的值出现变化，同时覆盖 4 个点
    # y[j + 1]-y[j]=

    return


# 值截断
def truncation(data, value=1):
    for i in range(len(data)):
        if data[i] > value:
            data[i] = value
    return data


# 超过该值的后续所有值都截断
def all_truncation(data, value=1):
    for i in range(len(data)):
        if data[i] > value:
            data[i:] = value
    return data


# 调整凸度，暂时失败
def convex(data, multi=2):
    data = np.array(data)
    center = len(data) / 2
    for j in range(len(data)):
        data[j] = data[j] * multi / np.log(abs(center - j + 20))
    return data


# 光滑采样曲线的随机波动
def curve_fluctuation(data, rate=0.5):
    # idx = np.arange(0, len(data))
    # fluct_idx = sample(idx, int(rate * len(data)))
    for j in range(1, len(data) - 1):
        if rt.prop_rand(rate):
            data[j] = rt.mean_rand(data[j], min_val=(data[j] - data[j - 1]) * 0.7,
                                   max_val=(data[j + 1] - data[j]) * 0.7)
    return data


# 根据点集获取多项式曲线
def get_poly_curve(x, y, order=0):
    if len(x) != len(y):
        print("错误！x 和 y 长度不一致！")
        return None
    if order <= 0:
        order = len(x)
    p_param = np.polyfit(x, y, int(order / 2))
    curve = np.poly1d(p_param)
    # print("拟合参数为：", p_param)
    return curve


# 两个函数拼接，将两个定义域相接的函数拼接为同一个函数
# 两个定义域相接处。需要平滑功能？
# x_lim 表示两个函数的定义域
def func_splicing(func1, func2, x_lim1=(0, 1), x_lim2=(1, 2)):
    if x_lim1[0] > x_lim2[1]:
        print("定义域顺序错误！")
        return 0
    mid = (x_lim1[1] + x_lim2[0]) / 2

    def func(x):
        # 输入的 x 可能为数组
        if is_num(x):
            if x_lim1[0] <= x <= mid:
                return func1(x)
            elif mid < x <= x_lim2[1]:
                return func2(x)
        else:
            # 若为数组则循环递归
            y = np.array([])
            for e in x:
                y = np.append(y, func(e))
            return y

    x_lim = (x_lim1[0], x_lim2[1])
    return func, x_lim


# 多函数拼接
def multi_func_splicing(funcs, x_lims):
    if len(funcs) != len(x_lims):
        print("错误！函数与定义域维度不一致！")
        return 0
    else:
        func = 0
        x_lim = 0
        for i in range(len(funcs) - 1):
            if x_lim == 0:
                func, x_lim = func_splicing(funcs[i], funcs[i + 1], x_lims[i], x_lims[i + 1])
            else:
                func, x_lim = func_splicing(func, funcs[i + 1], x_lim, x_lims[i + 1])
        return func, x_lim


# 输入参数是函数 + 范围
# wave_order 表示随机波动阶数，默认采用 7 阶多项式
def wave_curve(func, x_lim=(0, 1), wave_order=7):
    """
    :param func: 原始曲线
    :param x_lim: 曲线的定义域
    :param wave_order: 波动阶数
    :return:
    """
    if not isinstance(func, types.FunctionType):
        print("错误！请输入函数！")
        return 0
    if x_lim[1] < x_lim[0]:
        # 返回一个新 list
        x_lim = exchange_val(x_lim[0], x_lim[1])
    # 步长均分 100 份采样
    step = (x_lim[1] - x_lim[0]) / 100
    samp_x = np.arange(x_lim[0], x_lim[1], step)
    samp_y = func(samp_x)
    samp_y = rt.gauss_rand(samp_y, float_range=np.mean(samp_y) * wave_order / 1000)

    p_param = np.polyfit(samp_x, samp_y, wave_order)
    curve = np.poly1d(p_param)
    return curve


# 曲线采样
def get_curve_samp(curve, start, end, samp_num=100):
    step = (end - start) / samp_num
    x = np.arange(start, end, step)
    y = curve(x)
    return x, y


# 降低极值点
# rd_dg 表示降低程度，rd_dg=0 表示不动，rd_dg=1 完全削平。
def reduce_extremum(data, rd_dg=0.8):
    length = len(data)
    data_diff = np.diff(data)
    for j in range(1, length - 1):
        if data_diff[j - 1] * data_diff[j] < 0:
            mean = np.mean([data[j - 1], data[j + 1]])
            if data[j] > mean:
                data[j] = data[j] - (data[j] - mean) * rd_dg
            else:
                data[j] = data[j] + (mean - data[j]) * rd_dg
    return data


# 【偏移均值】也就是算得平均之后，向哪边偏移。
# bias < 0.5 表示向 val_2 偏移
# bias > 0.5 向 val_1 偏移
def get_bias_mean(val_1, val_2, bias=0.5):
    if val_1 > val_2:
        return val_2 + (val_1 - val_2) * bias
    else:
        return val_2 - (val_2 - val_1) * bias


# 【返回指定幅频相的 sin 值或数组】
# 前三个参数分别为振幅 AM、频率 Freq、相位 PH
# 输入的自变量 x 是 index，根据 fs 自动转换成时间步长。
# 即 t = 1 / fs, x = x * t
# 注意！幅频相参数只接受标量，不接受向量！
def sin(x, am=1., freq=1., phi=0., fs=44100):
    x = np.array(x, dtype=float)
    # 将 x 从采样点数，转化为时间。
    x = x / fs
    if isinstance(am, types.FunctionType):
        # 如果振幅是函数形式
        am = am(x)
    # else:
    #     return am * np.sin(freq * 2 * np.pi * x + phi)
    if isinstance(freq, types.FunctionType):
        freq = freq(x)
    if freq < 0:
        print("uf.sin 出现负频率，已调整为正频率。")
        freq = -freq
    return am * np.sin(freq * 2 * np.pi * x + phi)


# def sin_func(x, am_func, freq, phi, fs):
#     x = np.array(x, dtype=float)
#     x = x / fs
#     return am_func(x) * np.sin(freq * 2 * np.pi * x + phi)


# 【返回指定幅频相的 cos 值或数组】
def cos(x, am=1., freq=1., phi=0., fs=44100):
    x = np.array(x, dtype=float)
    x = x / fs
    return am * np.cos(freq * 2 * np.pi * x + phi)


# 【向信号中添加指定参数的 sin 信号】
def add_sin(data, am=1., freq=1., phi=0., fs=44100):
    data = np.array(data, dtype=float)
    x = np.arange(0, len(data))
    return data + sin(x, am, freq, phi, fs)


# 【向信号中添加指定参数的 cos 信号】
def add_cos(data, am=1., freq=1., phi=0., fs=44100):
    data = np.array(data, dtype=float)
    x = np.arange(0, len(data))
    return data + cos(x, am, freq, phi, fs)


# 【无溢出乘方】
def pow_no_overflow(x, power):
    return


# 【球面衰减公式】（弃用）
# 参数 b 是 1 或 2.5
# 参数 c 是 个体差异 6K - 8K ？
# 参数 a 是衰减速度，也决定了曲线的凹陷程度，实测 a 从 0.24 到 0.05 不等，那么可以人为指定为 0.1-0.2 不等
# d 是偏移量
def spherical_attenuation(x, c, d):
    # x 数据校验
    if is_ary(x):
        x = np.array(x, dtype=float)

    # 指定参数
    a = 0.034
    b = 1.1

    return c * (x ** (2 * b)) * np.exp(a * (x ** 2)) + d


# 【球面衰减公式】4 参数版（弃用）
def spherical_attenuation_4param(x, a, b, c, d):
    # x 数据校验
    if is_ary(x):
        x = np.array(x, dtype=float)
    x = 8 - x
    return c * (x ** b) * np.exp(a * x) + d


# [6.0686362658724405e-21, 1.9162799762384262, 2.6253189778938997]

#  p q d
# eta = 1
#     beta = 0
#     alpha = 0.2
#     # 初始声强
#     i0 = 1.1e5
#     r = path_distance(x, p, q, d)
#     return eta * i0 * np.power(r, beta) * np.power(np.e, -alpha * r)


# 碗形函数
def bowl(x, hz_offset, hz_scale, vrt_scale):
    a = 0.8
    center = 2
    # center 为倒数对称中心， b 为 center 的三次方
    b = 8
    if is_ary(x):
        # 数组情况，拆分递归
        x = np.array(x, dtype=float)
        y = []
        for i in range(len(x)):
            y = np.append(y, bowl(x[i], hz_offset, hz_scale, vrt_scale))
        return y
    x -= hz_offset
    x /= hz_scale
    if is_num(x) and x >= center:
        return vrt_scale * (np.power((x - center), 2) + a) * b / np.power(x, 3)
    elif is_num(x) and x < center:
        return vrt_scale * (np.power((x - center), 2) + a) * b / np.power((2 * center - x), 3)
    else:
        pass


# 路径距离模型
# 下牙区
def path_distance(x, p, q, d):
    return p * np.log(x) + q * x + d


# 上牙区
def path_distance_2(x, p, q, h, d):
    return np.sqrt((p * np.log(x) + q * x) ** 2 + h ** 2) + d


# 【上深度预测模型】现用
def depth_distinguish_upper(x, p, q, h, d):
    alpha = 0.246
    eta = 1 / (2 * np.pi)
    beta = -1
    # 初始声强
    i0 = 1.1e5
    r = path_distance_2(x, p, q, h, d)
    return eta * i0 * np.power(r, beta) * np.power(np.e, -alpha * r)


# 【下深度预测模型】现用
def depth_distinguish_lower(x, p, q, d):
    eta = 1
    beta = 0
    alpha = 0.246
    # 初始声强
    i0 = 1.1e5
    r = path_distance(x, p, q, d)
    return eta * i0 * np.power(r, beta) * np.power(np.e, -alpha * r)


# 【球面衰减公式】4 参数版画图用版
def spherical_attenuation_show(x):
    # a, b = [0.034, 1.1]
    # c, d = [20.90558018511369, 8284.137649003103]
    # x 数据校验
    if is_ary(x):
        x = np.array(x, dtype=float)
    # return c * (x ** (2 * b)) * np.exp(a * (x ** 2)) + d
    eta = 1
    alpha = 0.246
    beta = 0
    i0 = 1.1e5
    p, q, d = [6.0686362658724405e-21, 1.9162799762384262, 2.6253189778938997]
    r = path_distance(x, p, q, d)

    # 注意 path_distance 中带有 log(x)，因此定义域只有 [0, +∞]

    return eta * i0 * np.power(r, beta) * np.power(np.e, -alpha * r)


def csv2xy(path=r"C:\Users\xenon\Desktop\包络图.csv", show=False):
    csv_data = pd.read_csv(path)
    x = np.array(csv_data["x"])
    y = np.array(csv_data["Curve1"])
    if show:
        print("-------------------- CSV --------------------")
        print("x=")
        print(list2str(x))
        print("y=")
        print(list2str(y))
        print("--------------------以上来自 CSV 数据读取--------------------")
    return x, y


# 将 CSV 数据归零化
# x,y 轴数据浮动于 0-1 之间，首位顶到头 (0, 0)
def csv_zeroing(path=r"C:\Users\xenon\Desktop\包络图.csv", show=False):
    data = pd.read_csv(path)
    # 注意 pandas 不能用 -1 索引访问，原则上 csv 是无穷行。只能用 len - 1 的形式访问。
    if data["Curve1"][0] > data["Curve1"][len(data["Curve1"]) - 1]:
        # 首大于尾
        data["Curve1"] -= data["Curve1"][0]
    else:
        data["Curve1"] -= data["Curve1"][len(data["Curve1"]) - 1]
    # 小于 0 则归零
    for j in range(0, len(data["Curve1"])):
        if data["Curve1"][j] < 0:
            data["Curve1"][j] = 0

    # x 轴归零
    data["x"] -= data["x"][0]
    # xy 归一化
    data["x"] /= np.max(data["x"])
    print(data["Curve1"])
    data["Curve1"] /= np.max(data["Curve1"])

    data.to_csv(path, index=False)
    if show:
        print("-------------------- 改后 CSV 数据 --------------------")
        print("x=")
        print(list2str(data["x"]))
        print("y=")
        print(list2str(data["Curve1"]))
        print("--------------------以上来自 Pandas 内存数据--------------------")
    return


# 包络文件夹读取
def read_evp_from_csv_dir(dir_path=r"C:\Users\xenon\Desktop"):
    # 包络列表，每个元素是二元组，二元组内又包含xy列表。
    evp_list = []
    file_list = os.listdir(dir_path)
    for f_path in file_list:
        data = pd.read_csv(dir_path + '\\' + f_path)
        evp = (np.array(data["x"]), np.array(data["Curve1"]))
        evp_list.append(evp)
    return evp_list


# 全文件夹的 CSV 归零化
def csv_zeroing_dir(dir_path=r"C:\Users\xenon\Desktop"):
    file_list = os.listdir(dir_path)
    j = 0
    for f_path in file_list:
        csv_zeroing(dir_path + '\\' + f_path)
        j += 1
        v.show_progress(j, len(file_list))
    v.show_progress()
    return


# 包络调整，xy表示位置，wh表示宽高
def get_evp(path, x, y, w, h, pooling=1):
    x_data, y_data = csv2xy(path)
    x_data = x_data * w + x
    y_data = y_data * h + y
    x_data = f.get_mean_pooling(x_data, pooling)
    y_data = f.get_mean_pooling(y_data, pooling)
    return x_data, y_data


# 小于某数值过滤器
def low_val_filter(data, val):
    for i in range(0, len(data)):
        if data[i] <= val:
            data[i] = np.mean(data)
    return np.array(data)


# 大于某数值过滤器
def high_val_filter(data, val):
    for i in range(0, len(data)):
        if data[i] >= val:
            data[i] = np.mean(data)
    return np.array(data)


# 矩阵数值过滤器
def mat_val_filter(data, low, high):
    for i in range(0, len(data)):
        data[i] = high_val_filter(data[i], high)
        data[i] = low_val_filter(data[i], low)
    return data


class Signal:

    def __init__(self, l_data, r_data, fs):
        self.l_data = l_data
        self.r_data = r_data
        self.fs = fs


def file2signal(file_path):
    fs, signal = wavfile.read(file_path)
    l_data = np.array(signal[..., 0], dtype=float)
    r_data = np.array(signal[..., 1], dtype=float)
    signal = Signal(l_data, r_data, fs)
    return signal


# 从文件夹中读取全部信号
def read_as_signal_list(dir_path):
    s_list = []
    # 获取文件夹下所有文件名
    file_list = os.listdir(dir_path)
    print('正在执行数据读入（共找到', end='')
    print(len(file_list), end='')
    print('个文件）')
    i = 0
    for f_path in file_list:
        v.show_progress(i, len(file_list))
        s_list.append(file2signal(dir_path + '/' + f_path))
        i += 1
    v.show_progress(1, 1)
    return s_list


# 四分位法排除异常值
def del_outlier(data):
    """
    异常值（离群值）判别方法：
    上界=75%分位数+（75%分位数-25%分位数）*1.5
    下界=25%分位数- （75%分位数-25%分位数）*1.5
    比上界大的,和比下界小的都是异常值.
    :param data:
    :return:
    """
    # 计算四分位点
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # 找出异常值
    for j in range(0, len(data)):
        # 非正常值范围内
        if not q3 + (1.5 * iqr) >= data[j] >= q1 - (1.5 * iqr):
            data = np.delete(data, j)
    return data


"""
一种看参数影响程度的方法：
数量级影响曲线。
通过改变某参数不同的数量级，即 10 倍增长，查看曲线相比于上一次曲线的平方误差。

【参数结果记录】

upper：
n = 0.0001，dmax 自适应 0.8，单增，上凹 12000 - 16000
n = 0.001，dmax 自适应 0.8，单增，上凹 12000 - 16000
n = 0.01，dmax 自适应 0.78，单增，上凹 12500 - 16000
n = 0.1，dmax 自适应 0.65，单减（×），上凹

d_max = 0.9，n 自适应 -2.73，单减（×），上凹
d_max = 0.9，n 自适应 -2.56，单减（×），上凹
d_max = 0.8，n 自适应 -0.01，单增，线性
d_max = 0.7，n 自适应 -2.23，单减（×），上凹
d_max = 0.6，n 自适应 -2.1，单减（×），上凹
d_max = 0.5，n 自适应 0.18，单减（×），上凹

lower：
n = 1，dmax 自适应 9.，曲线单减 17000-12000
n = 0.5，dmax 自适应 9.，曲线单减
n = 0.1，dmax 自适应 10.，曲线水平
n = 0.01，dmax 自适应 10.，曲线水平
n = 0.001，dmax 自适应 10.，曲线水平

dmax = 25，n 自适应 -7，单增，上凹，2300 - 23000，还比较符合
dmax = 20，n 自适应 -5.32，单增，线性
dmax = 15，n 自适应 -2.7，单增，上凸（×）
dmax = 10，n 自适应 0.04，水平（×）
dmax = 5，n 自适应 3.55，单减（×）

我们需要去尝试解释参数值多出来的内容是什么意义。
并且按理讲，上区分度更大，因为衰减的原因更多。
接近槽牙时，上下牙应当接近，即 x = 8 时，上略小于下。

"""

"""
    首先，alpha 是负值，因为函数需要单调递减。
    当 alpha 为正时，分子位置为指数，增长速度远大于分母
    实测 dmax 影响较大，应当为
    alpha 决定了曲线弯曲程度，以及两端点的差。

    比较合理的参数：
    n = 0.008
    尝试改为 n = 0.002
    d_max = 0.6
    alpha = -0.2
    eta = 1 / 4 / np.pi
    beta = -2
    i0 = 1e5
    w=0.014

    上测8组，下测8组：

    上1：
    n = 0.008
    尝试改为 n = 0.002
    d_max = 0.6
    alpha = -0.2
    eta = 1 / 4 / np.pi
    beta = -2
    i0 = 1e5
    w=0.014

    其中 d_max 偏差较大，原因是需要用来进行拟合
    这就有的写了，因为 d 在指数位置，可以将多出来的 ed次方进行因式分解
    这个因式就表示整体的放缩程度

    alpha 加个负号
    解释为什么使用 9 次多项式，波动少，奇数次
    描述二阶差分平方和算法的缺点：慢、低
    解释为什么 dmax 会偏大？其中含有放缩系数，拟合用。

    测试：
    功能测试 32 分类混淆矩阵，综合识别准确率
    时间测试方案，32 颗牙的时间曲线，两条，一个秒表的，一个算法的
    与 VGG 对比，要把 VGG 的方案和参数写出来，VGG 区内识别准确，上下牙区识别不准

    写 32 分类 VGG 网络判别
"""

"""
为什么不用 arctan？因为没有峰。固有频率需要有个峰。
要求：
a >= 0，a 表示二次函数的偏移量
还需要有个横向拉伸的功能？
不需要横向偏移量？

那么只能将 a、center 设为定值，其余方法就是对其进行放缩

现确定参数：
下 hz_offset, hz_scale, vrt_scale = [455, 2, 5]
上 hz_offset, hz_scale, vrt_scale = [690, 2.5, 4]
"""


def bowl_show(x):
    # hz_offset, hz_scale, vrt_scale = [690, 2.5, 4]
    hz_offset, hz_scale, vrt_scale = [2, 1, 4]
    a = 0.8
    center = 2
    # center 为对称中心 b 为 center 的三次方
    b = 8
    if is_ary(x):
        x = np.array(x, dtype=float)
        y = []
        for i in range(len(x)):
            y = np.append(y, bowl_show(x[i]))
        return y
    x -= hz_offset
    x /= hz_scale
    if is_num(x) and x >= center:
        return vrt_scale * (np.power((x - center), 2) + a) * b / np.power(x, 3)
    elif is_num(x) and x < center:
        return vrt_scale * (np.power((x - center), 2) + a) * b / np.power((2 * center - x), 3)
    else:
        pass


"""
    x = np.arange(1, 9)
    
    下牙区标准采样数据：
    std_y = [4977.122206528136, 6305.612274942442, 8139.3473186985675, 10644.80995295658, 14032.426184540087,
             18570.92266346762, 24603.7086323455, 32569.10568840025]
    拟合参数：
    a=0.2, b=0.95, c=800, d=4000

    上牙区标准采样数据：
    std_y = [6221.034183615129, 6850.63428730222, 7950.457981228854, 9617.893453670782, 11974.798240381508,
             15168.091425622266, 19372.499952423263, 24794.36070972585]
    拟合参数：
    a=0.1, b=1.8, c=200, d=6000

    实测发现 d 决定了曲线的初始高度，4K - 8K 之间都是正常的，所以大范围浮动
    c 是为了适应 ab 的变化，原则上不能大范围浮动。
"""

# 数字或数组（可运算类型）
ALL = 1
# 数字
NUM = 2
# 数组
ARY = 3
# 整型
INT = 4
# 浮点型
FLT = 5
# 复数
CPL = 6


# 【整型判断】
def is_int(data):
    if type(data) == int or type(data) == np.int32 or type(data) == np.int64:
        return True
    else:
        return False


# 【浮点型判断】
def is_flt(data):
    if type(data) == float or type(data) == np.float64:
        return True
    else:
        return False


# 【复数判断】
def is_cpl(data):
    if type(data) == complex or type(data) == np.complex128:
        return True
    else:
        return False


# 【判断是数字】
def is_num(data):
    if is_int(data) or is_flt(data) or is_cpl(data):
        return True
    else:
        return False


# 【数组判断】
# 包括 ndarray、list、tuple、range 类型。
def is_ary(data):
    if type(data) == np.ndarray or type(data) == list or type(data) == tuple or type(data) == range:
        return True
    else:
        return False


def is_list(data):
    # ndarray 不行
    if type(data) == list:
        return True
    else:
        return False


def is_dict(data):
    if type(data).__name__ == "mappingproxy" or type(data).__name__ == "dict":
        return True
    else:
        return False


"""
怎么实现曲线细分？
较少的离散值组成的折线，需要较多的采样点来细分。
1. 升采样（线性插值）
2. 执行平滑卷积
"""


# 【曲线细分】
def curve_subdivision(x, y, multi=10, wnd=5):
    # 先计算采样间隔
    return


"""
求阈值算法：
（一）迭代阈值分割：（未知两组原始数据的分组）
    初始 th 为数据最值（最大和最小）的中间值
    flag：用该 th 分割数据，得到两组数据
    两组数据求均值
    新的 th 是两个均值的均值
    goto flag 重新迭代，直到变化很小为止
（二）最大类间方差 OTSU：
    数据1 的样本数为 w1，平均值为 u1
    数据2 的样本数为 w2，平均值为 u2
    那么数据的总均值为 u = w1u1 + w2u2
        当然，有条件也可以直接用总数除以个数
    方差 g = “未完待续”

（三）最小二乘法？
    一个值位于两组数据中间，使得方差最小
    初始值为均值

"""


# 交换数组中的两个位置的元素
# 可以兼容自己和自己交换
def exchange_idx(arr, idx_1, idx_2):
    tmp = arr[idx_1]
    arr[idx_1] = arr[idx_2]
    arr[idx_2] = tmp
    return arr


def exchange_val(v_1, v_2):
    temp = v_1
    v_1 = v_2
    v_2 = temp
    return v_1, v_2


# 通过变量值，交换数组中两个元素的位置
def exchange_elements(arr, val_1, val_2):
    arr = list(arr)
    if arr.count(val_1) == 0 or arr.count(val_2) == 0:
        print("exchange_elements 错误：未找到元素，已返回原始值。")
        return arr
    idx_1 = arr.index(val_1)
    idx_2 = arr.index(val_2)
    tmp = arr[idx_1]
    arr[idx_1] = arr[idx_2]
    arr[idx_2] = tmp
    return arr


# 用于 rev_cal 的反转计算
PLUS = 1
MINUS = -1
cur_op = PLUS


# 【交替反转计算】
def rev_cal(val_1, val_2, rev=True):
    global cur_op
    # 先判断是否要变换运算方式
    if rev:
        cur_op *= -1
        return rev_cal(val_1, val_2, False)
    if cur_op == PLUS:
        return val_1 + val_2
    elif cur_op == MINUS:
        return val_1 - val_2
    else:
        print('cur_op 参数错误')


# 【获取均值阈值】
# 分别计算两个集合的均值，然后再取平均
# 注意，该算法与整体取平均是不一样的
def get_mean_th(data_1, data_2):
    data_1 = mat2list(data_1)
    data_2 = mat2list(data_2)
    return np.mean(np.append(data_1, data_2))


# 获取 SVM 阈值
# c 为松弛
def get_svm_th(data_1, data_2, c=10):
    # 转为列向量
    x = np.array(data_1)
    x = np.append(x, np.array(data_2))
    x = x.reshape(-1, 1)
    y = [1] * len(data_1)
    y = np.append(y, [-1] * len(data_2))
    # 构造线性分类模型
    clf = svm.SVC(kernel="linear", C=c)
    clf.fit(x, y)
    # 计算阈值
    th = -clf.intercept_ / clf.coef_[0]
    return th


def get_th_accu(data_1, data_2, th, cls=1):
    """
    通过阈值计算两个集合的准确率（预测正确数/总数）
    :param data_1:
    :param data_2:
    :param th:
    :param cls: 谁应当大于阈值谁应当小于阈值，或者说谁是正样本谁是负样本。cls > 0 表示前者为正样本。
    :return:
    """
    correct_num = 0
    if cls > 0:
        correct_num += np.sum(data_1 > th)
        correct_num += np.sum(data_2 < th)
    else:
        correct_num += np.sum(data_1 < th)
        correct_num += np.sum(data_2 > th)
    return correct_num / (len(data_1) + len(data_2))


# 二维数组展开为一维
# 适用于 mat 的两个维度不一致的情况
def mat2list(data):
    data_ = np.array([])
    for j in range(len(data)):
        data_ = np.append(data_, data[j])
    return data_


# 【获取最小二乘法阈值】
# acura 表示精确度，实测精度最大也就到 15（可能是 64 位机器的原因）
def get_mse_th(data_1, data_2, acura=15):
    data_1 = np.array(data_1, dtype=float)
    data_2 = np.array(data_2, dtype=float)
    th = get_mean_th(data_1, data_2)
    mse = np.sum((data_1 - th) ** 2) / len(data_1) + np.sum((data_2 - th) ** 2) / len(data_2)
    last_mse = mse * 2
    mse_div = last_mse / mse
    # 精确度
    acura = 0.1 ** acura
    while not 1 - acura < mse_div < 1 + acura:
        if mse_div < 1:
            # 下降函数用 “绝对值指数”
            th = rev_cal(th, np.abs(np.log(mse_div)) * acura)
        else:
            th = rev_cal(th, np.abs(np.log(mse_div)) * acura, False)
        # 重新计算均方差
        last_mse = mse
        mse = np.sum((data_1 - th) ** 2) / len(data_1) + np.sum((data_2 - th) ** 2) / len(data_2)
        mse_div = last_mse / mse
    return th


# 【获取最小二乘法偏移量】
# 需要补零算法
# 需要一个连续的量？
def get_mse_offset(data_1, data_2, acura=15):
    print('功能尚未开发，敬请期待。。。')
    pass
    return


# fft 相位问题讲解地址： https://www.ilovematlab.cn/thread-528420-1-1.html
# 信号泄露 https://zhuanlan.zhihu.com/p/24318554


"""
注：
复指数信号 A * exp(j * 2pi * f * n + phi)
正弦信号 A * sin(2pi * f * n + phi)
信号经过FFT后，得到的结果是复数数组，具有虚部和实部。
对实部和虚部的平方和求平方根就是幅值。
对实部和虚部的比求反正切就是相位，得到弧度（带 pi 的值）。
相位有取值范围。
"""


# 【将两个声道合并写入文件】
# 注意：data_1 必为左声道，需要在程序中自行处理
# around 表示是否取整
def signal_write(data_1, data_2, fs, f_path, around=False):
    if len(data_1) != len(data_2):
        print('错误：左右声道不一致，分别为：', len(data_1), '\t', len(data_2))
        return
    # 重新归为整数
    if around:
        data_1 = np.rint(data_1)
        data_1 = np.array(data_1, dtype=int)
        data_2 = np.rint(data_2)
        data_2 = np.array(data_2, dtype=int)
    else:
        data_1 = np.array(data_1, dtype=float)
        data_2 = np.array(data_2, dtype=float)
    signal = np.empty((2, len(data_1)), dtype=int)
    signal[0] = data_1
    signal[1] = data_2
    signal = signal.T
    wavfile.write(f_path, fs, signal)
    print('文件完成：', end='')
    print(f_path)
    return


# wav 文件切割（按时间窗口）
def cut_file(path=r"", time_range=(0, 0)):
    if time_range == (0, 0):
        print("需要指定切割范围")
        return
    fs, signal = wavfile.read(path)
    data_1 = signal[..., 0]
    data_1 = data_1[int(time_range[0] * fs):int(time_range[1] * fs)]
    data_2 = signal[..., 1]
    data_2 = data_2[int(time_range[0] * fs):int(time_range[1] * fs)]
    signal_write(data_1, data_2, fs, path)
    return


# 【写入发生错误信号的】
# 文件命名格式：reason + num，写入前应当对已存在的文件名进行查找
# fs <= 0 表示是普通数组，并非时域信号
def error_data(data, reason, path=r".\错误数据", fs=44100):
    print(FontColor.YELLOW_WARNING + "发生错误，原因是：", reason)
    data = np.array(data, dtype=float)
    if fs <= 0:
        # 非音频信号
        i = 1
        while True:
            if os.path.exists(path + '\\' + reason + str(i) + ".npy"):
                i += 1
            else:
                break
        np.save(path + '\\' + reason + str(i) + ".npy", data)
    else:
        # 音频信号
        i = 1
        while True:
            if os.path.exists(path + '\\' + reason + str(i) + ".wav"):
                i += 1
            else:
                break
        wavfile.write(path + '\\' + reason + str(i) + ".wav", fs, data)
    print(FontColor.YELLOW_WARNING + "信号已写入路径：", path)
    print(FontColor.BLACK_NORMAL)
    return


# 将不同长度的信号相加（裁切方式）
def add_diff_len(data_1, data_2):
    if len(data_1) > len(data_2):
        data_1 = data_1[0:len(data_2)]
    else:
        data_2 = data_2[0:len(data_1)]
    return data_1 + data_2


# 【对勾函数】
# d 就是真实的铅直渐近线
# 我们希望的是左边平坦，右边陡
# 会出现拟合或计算失败，因为 x 不能取 0
def tick(x, a=5, b=4500, c=0, d=0, e=1):
    x -= d
    x /= e
    return a * x + b / x + c


"""
在第一象限内
转折点为：(√(b/a), 2√(a*b))，转折点即为极小值点、最小值点
斜渐近线为 y = ax
那么为了实现左平坦右陡峭，应当使得：
    转折点靠右，渐近线斜率大
    a 尽可能大，b 尽可能小
现假设指定斜率为 a，极值点横坐标为 z，那么 z = √(b/a) => b = az^2 
"""

"""
函数不大行，区分度太小
"""


# 【Sigmoid 函数】
# width 表示带宽
def sigmoid(x, offset=0, width=1):
    x -= offset
    return 1 / (1 + np.exp(-(1 / width) * x))


# 【Sigmoid 导数】
def sigmoid_deriv(x, offset=0, width=1):
    return sigmoid(x, offset, width) * (1 - sigmoid(x, offset, width))


# 【获取 sigmoid 最优参数】找到符合该通频带的最优 offset、width 值
# line = True 表示线性下降，False 表示指数下降
# obj_grad 表示，多大的斜率定义了主值区间
# line 表示是否使用线性下降法迭代（用于性能优化）（默认是指数下降法）
# 返回值为两个：偏移量、width
def get_sigmoid_param(proc_range=[-9, 9], obj_grad=0.025, line=False):
    width = abs(proc_range[1] - proc_range[0])
    if width == 0:
        print("输入通频带范围不正确！")
        return 1
    offset = (proc_range[1] + proc_range[0]) / 2
    while True:
        width = np.around(width, 6)
        # 注意，此 width 非彼 width，而只是作为初始迭代参数使用
        if sigmoid_deriv(proc_range[1], offset, width) > 1.1 * obj_grad:
            if line:
                denominator = sigmoid_deriv(proc_range[1], offset, width) - obj_grad + 1
            else:
                denominator = np.exp(sigmoid_deriv(proc_range[1], offset, width) - obj_grad)
            width /= denominator
            continue
        if sigmoid_deriv(proc_range[1], offset, width) < 0.9 * obj_grad:
            if line:
                denominator = obj_grad - sigmoid_deriv(proc_range[1], offset, width) + 1
            else:
                denominator = np.exp(obj_grad - sigmoid_deriv(proc_range[1], offset, width))
            width /= denominator
            continue
        return offset, width


# 【获取有序数组中离目标值的最近点索引】
# 返回值有两个，一个是最近点索引，一个是次近点索引。
# double 表示是否返回最近的两个索引点
# invert 表示倒序查找（用于优化性能）（功能尚待开发）
def get_nearest_idx(data, obj_val, double=False, invert=False):
    data = np.array(data, dtype=float)

    # 以下部分代码省略
    # _1st_idx, _2nd_idx = 0, 0
    # _1st_dis, _2nd_dis = 2147483647, 2147483647
    # for i in range(len(data)):
    #     dis = abs(data[i] - obj_val)
    #     """
    #     如何找最近的两个点？
    #     分为两种情况：
    #         找到比 1st 小的索引，那么 1 存 2，新存 1.
    #         找到比 1st 大，但是比 2nd 小的索引，那么新存 2.
    #     """
    #     if dis < _1st_dis:
    #         # 如果找到恰好等于的点，则两个值均为 i
    #         if dis == 0:
    #             _1st_idx = i
    #             _2nd_idx = i
    #             break
    #             # 跳出循环，交由后面的代码处理
    #         _2nd_dis = _1st_dis
    #         _1st_dis = dis
    #         _2nd_idx = _1st_idx
    #         _1st_idx = i
    #     elif dis < _2nd_dis:
    #         _2nd_dis = dis
    #         _2nd_idx = i

    data = np.abs(data - obj_val)
    return np.argmin(data)

    # if double:
    #     return _1st_idx, _2nd_idx
    # else:
    #     return _1st_idx


"""
在 numpy 中有现成的函数：
np.where()

"""


# 【获取序列峰值，并进行排名】
# 输入参数 idx_range 是序列的 ”索引“ 范围，若要算频率范围需要自行转换
# 从序列中指定索引位置找到峰值，并按照从大到小排序
# 返回值为峰的索引和峰值
def get_peaks(am, idx_range=None):
    """
    【获取序列峰值，并进行排名】
    注意：idx_range 不能默认为 list 类型，否则其执行过程中会保存局部变量。
    :param am: 幅值序列
    :param idx_range: 索引范围
    :return: peak_idx 峰索引数组, peak_val 峰值数组
    """
    # 参数校验
    if idx_range is None:
        idx_range = [0, 0]
    if idx_range[0] == idx_range[1]:
        idx_range[0] = 0
        idx_range[1] = len(am)
    if idx_range[1] == len(am):
        idx_range[1] -= 1
    # 重新封装数据
    am = np.array(am, dtype=float)
    # 注意：diff 的长度比 am 短一个索引。
    diff = np.diff(am)
    # 峰索引
    peak_idx = []
    # 峰值
    peak_val = []
    print("正在获取峰值序列：")
    for i in range(idx_range[0], idx_range[1] - 1):
        v.show_progress(i, idx_range[1] - 1 - idx_range[0])
        # 从一阶差分中找峰值点，i+1 就是原始数据的峰值位置
        if diff[i + 1] < 0 < diff[i]:
            j = 0
            while j < len(peak_val):
                if am[i + 1] > peak_val[j]:
                    peak_val = np.insert(peak_val, j, am[i + 1])
                    peak_idx = np.insert(peak_idx, j, i + 1)
                    break
                j += 1
            if j >= len(peak_val):
                # 如果从头找到尾都没找到，则直接添加
                peak_val = np.append(peak_val, am[i + 1])
                peak_idx = np.append(peak_idx, i + 1)
    v.show_progress()
    # 将 peak_idx 转换为 int 型
    peak_idx = np.array(np.rint(peak_idx), dtype=int)
    return peak_idx, peak_val


# 获取主峰频率
def get_main_peak_freq(data, fs):
    am, freq, phi = v.get_fft_result(data, fs)
    # peak_idx, peak_val = get_peaks(am)
    idx = np.argmax(am)
    return freq[idx]


# 【获取序列谷值，并按绝对值大小排名】
def get_valleys(am, idx_range=[0, 0]):
    am = np.array(-am, dtype=float)
    valley_idx, valley_val = get_peaks(am, idx_range)
    return valley_idx, valley_val


# 【获取调整倍数】
def get_multi(real_1, real_2, obj_multi):
    """
    【获取调整倍数】
    算法解释：
        现有两个真实值：真值1、真值2
        我希望【真值 1】乘上一个【真实倍数】，就能变成【指定倍数】的【真值 2】
        公式表述为：real_1 × real_multi = real_2 × obj_multi
    :param real_1: 真实值 1
    :param real_2: 真实值 2
    :param obj_multi: 目标倍数
    :return: 真实倍数
    """
    return real_2 * obj_multi / real_1


# 【共振峰判别法】
# 判别用代码
def formant_mean(data, fs=44100, low_freq=20, high_freq=45, peak_num=5):
    data = np.array(data, dtype=float)
    norm_am, freq, phi = v.get_fft_result(data, fs)
    freq_cut, l_idx, r_idx = v.get_freq_range(freq, low_freq, high_freq)
    peak_idx, peak_val = get_peaks(norm_am, [l_idx, r_idx])
    mean = 0
    for i in range(0, peak_num):
        mean += freq[peak_idx[i]] * peak_val[i]
    return mean


# 【粒子滤波】暂时没写完
def get_nearest_phi_2(data, am, freq, fs=44100, show_signal=False):
    # # 采样就从头采
    # x = range(0, len(data))
    # # 一个周期分 8 份，进行粗粒度计算
    # gl = 1000
    # phases = np.arange(-np.pi, np.pi, np.pi / gl)
    # app_variance = np.empty((len(phases),))
    # for i in range(0, len(phases)):
    #     y = sin(x, am, freq, phases[i], fs)
    #     app_variance[i] = np.sum((data - y) ** 2)
    # init_idx = np.argmin(app_variance)
    # phi = phases[init_idx]
    # return phi
    pass


# 【获取最接近的 sin 相位值】
# 通过最小二乘法迭代获得
def get_nearest_phi(data, am, freq, fs=44100, show_signal=False):
    # 采样就从头采
    x = range(0, len(data))
    # 一个周期分 16 份，进行粗粒度计算
    gl = 16
    phases = np.arange(-np.pi, np.pi, np.pi / gl)
    app_variance = np.empty((len(phases),))
    for i in range(0, len(phases)):
        y = sin(x, am, freq, phases[i], fs)
        app_variance[i] = np.sum((data - y) ** 2)
    init_idx = np.argmin(app_variance)
    phi = phases[init_idx]
    y = sin(x, am, freq, phi, fs)
    variance = app_variance[init_idx]
    # 乘 2 是为了保证能进入循环
    last_variance = variance * 2
    variance_div = last_variance / variance
    # 从 0.5 逐渐接近 1 的过程
    while not 0.9999 < np.abs(variance_div) < 1.0001:
        # 如果方差商变小说明走错方向了，要倒回去，arctan 获得的正负值没有意义
        if variance_div < 1:
            phi = rev_cal(phi, np.arctan(variance_div - 1) / 32)
        else:
            phi = rev_cal(phi, np.arctan(variance_div - 1) / 32, False)
        """
            注意：
                不论输入的 sin 周期是多少，phi 始终是以 2pi 为一个周期！
            这里有个 bug：
                偏移的相位小时，会导致两次方差的商很小
                而方差的商很小，反过来又会导致相位小
            正常的迭代算法应该是：
                从 -pi 到 pi 粗粒度依次计算方差值
                然后再从中进行精细调整
            方差差值太大，在 arctan 中会导致左右来回摆动，只能用商
            
            上述的 arctan 算法没有问题，但是为什么会出现反相：
                因为反相时，左右偏移的方差变化同样很小
                那么就必须限制其变化范围在 pi/8 以内
        """
        # 重新计算方差
        y = sin(x, am, freq, phi, fs)
        last_variance = variance
        variance = np.sum((data - y) ** 2)
        variance_div = last_variance / variance
    if show_signal:
        v.show_am_time(data)
        v.show_am_time(y)
        v.show_am_time(data - y)
    return phi


# 【两点均分算法】
# 返回两点之间的均分数组
# 值包含 p_1 不包含 p_2
def divide_2points(p_1=(0, 0), p_2=(0, 0), sub_dg=100):
    x_step = (p_2[0] - p_1[0]) / sub_dg
    x = np.arange(p_1[0], p_2[0], x_step)
    """
    注意：
        有时会出现 x，y 不等长的情况？
    """
    y_step = (p_2[1] - p_1[1]) / sub_dg
    # 若两个 y 相同，即两点之间是一条水平线
    if y_step == 0:
        y = np.array([p_1[1]] * sub_dg)
    else:
        y = np.arange(p_1[1], p_2[1], y_step)
    if len(x) > len(y):
        x = x[0:-1]
    elif len(x) < len(y):
        y = y[0:-1]
    return x, y


# 【获取两点连线函数】（直线两点式）
def get_line(p_1=(0, 0), p_2=(0, 0), reverse=False):
    x1 = p_1[0]
    x2 = p_2[0]
    y1 = p_1[1]
    y2 = p_2[1]

    if reverse:
        # 直线的反函数
        def reverse_line(y):
            return (x2 - x1) / (y2 - y1) * y - y1 * (x2 - x1) / (y2 - y1) + x2

        return reverse_line
    else:
        def line(x):
            return (y2 - y1) / (x2 - x1) * x - x1 * (y2 - y1) / (x2 - x1) + y2

        return line


# 获得一个序列的过零点（用于 EarSense 算法）
def get_zero_array(data):
    z_array = []
    for i in range(len(data) - 1):
        reverse_line = get_line((i, data[i]), (i + 1, data[i + 1]), True)
        z_array.append(reverse_line(0))
    return z_array


# 【获取两个值的比例均值】
def get_prop_mean(a, b, prop):
    if not is_num(a) or not is_num(b):
        print("get_prop_mean 错误！输入参数必须为数字！")
        return a
    return (b - a) * prop + a


# 判断数组的正值个数
def get_positive_num(data):
    num = 0
    for x in data:
        if x > 0:
            num += 1
    return num


"""
参考网址：
https://www.cnpython.com/qa/24222

    计算导数使用：np.diff()
    
    计算差分有现成的函数：
    np.gradient(y, dx)
    其中 y 是表达式
"""
