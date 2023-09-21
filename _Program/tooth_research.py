import time

import numpy as np
from scipy.optimize import curve_fit

import utility_functions as uf
import view as v
import random_tools as rt
import power as p
import filter as f
import matplotlib.pyplot as plt
from scipy.io import wavfile
import re
import os
import _thread
import models as m
import multi_thread as mt


# Teeth 类，用于定义数据结构
class Teeth:
    # 牙区标号
    UL = 'UL'
    UR = 'UR'
    DL = 'DL'
    DR = 'DR'

    def __init__(self, uname, area, depth, num, data_1, data_2, fs):
        self.uname = uname
        self.area = area
        self.depth = int(depth)
        self.num = int(num)
        self.data_1 = np.array(data_1, dtype=float)
        self.data_2 = np.array(data_2, dtype=float)
        self.fs = fs


fitting_x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)

"""
三分法只分频带，不分文件，所有文件都要遍历一遍。
需要有个 file_list 或 signal_list 用于多线程切分。

"""


def power_list_by_spectrum(signal_list, freq_range):
    power_list = []
    for s in signal_list:
        power_list.append(
            p.get_freq_power_by_spectrum(s.data_1, freq_range[0], freq_range[1], s.fs) +
            p.get_freq_power_by_spectrum(s.data_2, freq_range[0], freq_range[1], s.fs))
    return power_list


# 计算所有文件在一个频率区间的区分度
# proc_list 表示是否是单线程，如果传入为列表则为多线程
def freq_range_discr(upper_signal_list, lower_signal_list, freq_range, proc_list=0):
    if uf.is_list(proc_list):
        # 准备多线程参数
        upper_args = [upper_signal_list, freq_range]
        upper_power_list = mt.mt_list2list(power_list_by_spectrum, upper_args, 0)
        lower_args = [lower_signal_list, freq_range]
        lower_power_list = mt.mt_list2list(power_list_by_spectrum, lower_args, 0)
    else:
        # 单线程计算
        upper_power_list = power_list_by_spectrum(upper_signal_list, freq_range)
        lower_power_list = power_list_by_spectrum(lower_signal_list, freq_range)

    # 通过 SVM 计算阈值
    th = uf.get_svm_th(upper_power_list, lower_power_list)
    accu = uf.get_th_accu(upper_power_list, lower_power_list, th)

    return accu


# 遍历不同带宽（同起始位置）
def traversal_bandwidth_discr(upper_signal_list, lower_signal_list, freq_pos, bandwidth_range=(100, 1600), proc_num=-1,
                              proc_list=[0]):
    discr_list = []
    # 单线程情况
    if proc_num < 0:
        pass
    else:
        # 多线程情况
        for bandwidth in range(bandwidth_range[1] - bandwidth_range[0]):
            proc_list[proc_num] = bandwidth - bandwidth_range[0]
            v.show_progress(np.sum(proc_num[:-1]), proc_list[-1])
            discr_list.append(
                freq_range_discr(upper_signal_list, lower_signal_list, (freq_pos, freq_pos + bandwidth), 0))
            proc_list[proc_num] += 1
        v.show_progress(np.sum(proc_num[:-1]), proc_list[-1])
    return discr_list


# 遍历频率区间的区分度计算
# 多线程是通过不同的区间来计算的，将 freq 划分为不同区间
def traversal_freq_range_discrimination(upper_path, lower_path, freq_pos_range=(900, 2400),
                                        bandwidth_range=(100, 1600)):
    # for freq_pos in range(freq_pos_range[0], freq_pos_range[1]):
    #     for bandwidth in range(bandwidth_range[0], bandwidth_range[1]):
    #         # 进度算法：总的是长乘宽的二维数组。
    #         v.show_progress((freq_pos - freq_pos_range[0]) * (bandwidth_range[1] - bandwidth_range[0]) + bandwidth -
    #                         bandwidth_range[0],
    #                         (bandwidth_range[1] - bandwidth_range[0]) * (freq_pos_range[1] - freq_pos_range[0]))

    upper_signal_list = uf.dir2_signal_list(upper_path)
    lower_signal_list = uf.dir2_signal_list(lower_path)

    # 构造频率位置序列
    freq_pos_list = np.arange(freq_pos_range[0], freq_pos_range[1])

    # 最后一个参数 proc_list 不写，由多线程调度函数填入
    args = [upper_signal_list, lower_signal_list, freq_pos_list]
    mt.mt_list2list(traversal_bandwidth_discr, args, 2)

    return


# 通过 cd 的值得到 8 颗牙的预测值
def get_real_am(c, d):
    x = np.arange(1, 9)
    y = uf.spherical_attenuation(x, c, d)
    print("预测值为：", uf.list2str(y))
    return y


# # 通过功率预测深度
# def pred_depth(power):
#     return uf.get_nearest_idx(real_am, power) + 1


# 【计算某频率范围内的振幅均值】
# 传入的数据 norm_am 和 freq 需要使用
def freq_mean(norm_am, freq, low_freq, high_freq):
    norm_am = np.array(norm_am, dtype=float)
    low_idx = uf.get_nearest_idx(freq, low_freq)
    high_idx = uf.get_nearest_idx(freq, high_freq)
    # 计算公式
    return np.sum(norm_am[low_idx:high_idx]) / (high_idx - low_idx)


# 【获取某频率范围内前 n 个峰值的均值】
def get_peak_mean(data, low_freq, high_freq, n=5, fs=44100):
    data = np.array(data, dtype=float)
    am, freq, phi = v.get_fft_result(data, fs)
    low_idx = uf.get_nearest_idx(freq, low_freq)
    high_idx = uf.get_nearest_idx(freq, high_freq)
    if type(low_idx) == tuple or type(high_idx) == tuple:
        print('返回的索引是 tuple 类型')
        print(low_idx)
        print(high_idx)
        print('数据维度是：')
        print(data.shape)
        """
            返回的数据是 (20, -1) (45, -1) 这种方式怎么找到的？
        """
    peak_idx, peak_val = uf.get_peaks(am, [low_idx, high_idx])
    if n > len(peak_val):
        n = len(peak_val)
    return np.sum(peak_val[0:n]) / n


# 【校验是否加入 sin 后是否出现同频叠加的情况】
# True 表示出现了这种情况，应当使用原始值
def co_freq(data, mean, fs):
    low_freq = 20
    high_freq = 45
    am, freq, phi = v.get_fft_result(data, fs)
    low_idx = uf.get_nearest_idx(freq, low_freq)
    high_idx = uf.get_nearest_idx(freq, high_freq)
    peak_idx, peak_val = uf.get_peaks(am, [low_idx, high_idx])
    if peak_val[0] >= 2 * mean:
        return True
    else:
        return False


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


"""
    文件命名规则：
        [志愿者名]-[牙区]-[深度]-[编号(3位数)].wav
        例如：WY-UL-2-001.wav
"""


# 读入 wav 文件，并转换为 teeth 对象
# 需要传入完整的带路径的文件名
def file2teeth(file_path=r"F:\python生成的数据\WY-UL-1-000.wav"):
    # dir_reg = r'([A-Za-z]:|\.)\\.+'
    # # 问号表示懒惰匹配
    # file_reg = r'\\\w+?-(UL|UR|DL|DR)-[1-8]-[0-9]{3}\.[Ww][Aa][Vv]'
    # dir_reg_result = re.match(dir_reg, file_path)
    # file_reg_result = re.search(file_reg, file_path)
    # if not dir_reg_result:
    #     print('路径名不正确！')
    #     return None
    # if not file_reg_result:
    #     print('文件名不正确，已跳过...')
    #     return None
    # if not os.path.exists(file_path):
    #     print('文件不存在，已跳过...')
    #     return None

    # UNIX
    file_reg = r'/w+?-(UL|UR|DL|DR)-[1-8]-[0-9]{3}/.[Ww][Aa][Vv]'
    # Windows
    file_reg = r'\w+?-(UL|UR|DL|DR)-[1-8]-[0-9]{3}\.[Ww][Aa][Vv]'
    file_reg_result = re.search(file_reg, file_path)

    fs, signal = wavfile.read(file_path)
    # 其属性来自于文件名
    file_name = file_reg_result.group()[0:].split('.')[0]
    attrs = file_name.split('-')
    data_1 = np.array(signal[..., 0], dtype=float)
    data_2 = np.array(signal[..., 1], dtype=float)
    teeth = Teeth(attrs[0], attrs[1], attrs[2], attrs[3], data_1, data_2, fs)
    # print(teeth)
    return teeth


# 将文件夹中的所有数据读入，并存入 Teeth 对象中，再存入 list 中
# 需要读取或遍历文件名
def dir2_tooth_list(dir_path=r'F:\python生成的数据'):
    tooth_list = []
    dir_reg = r'([A-Za-z]:|\.)\\.*'
    dir_reg_unix = r'/.*'
    dir_reg_result = re.match(dir_reg, dir_path) or re.match(dir_reg_unix, dir_path)
    if not dir_reg_result:
        print('路径名不正确！')
        return None
    if not os.path.exists(dir_path):
        print('路径不存在！')
        return None
    file_list = os.listdir(dir_path)
    print('正在执行数据读入（共找到', len(file_list), '个文件）')
    i = 0
    for f_path in file_list:
        v.show_progress(i, len(file_list))
        tooth_list.append(file2teeth(dir_path + '/' + f_path))
        i += 1
    v.show_progress(1, 1)
    return tooth_list


def add_AC(data_1, ac_multiple=1):
    # 加入 50Hz 正弦
    am_1 = rt.gauss_rand(8, float_range=1)
    freq_1 = np.arange(50, 2000, 50)
    phi_1 = rt.mean_rand(0, -np.pi, np.pi)
    for j in range(len(freq_1)):
        data_1 = uf.add_sin(data_1, am_1 * 50 / freq_1[j], freq_1[j], phi_1)

    # 加入直流分量
    am_1 = rt.gauss_rand(3, float_range=0.5)
    freq_1 = np.arange(0.1, 50, 0.5)
    phi_1 = rt.mean_rand(0, -np.pi, np.pi)
    for j in range(len(freq_1)):
        data_1 = uf.add_sin(data_1, am_1 * 0.15 / freq_1[j] * ac_multiple, freq_1[j], phi_1)
    return data_1


# 现已改称【固有频率调整算法】
# 原【共振峰调整算法】
def nature_freq_adjust(data_1, data_2, area, depth):
    """
    :param data_1:声道 1 数据
    :param data_2:声道 2 数据
    :param area:所属牙区
    :param fs:采样率
    :param adj_range:调整幅度，注意 adj_range 越大，调整幅度越小！
    :return:调整后的左右声道值 data_1, data_2
    """

    dir_path = r"C:\Users\xenon\Desktop\evp"
    csv_num = 19

    # 【三】加入碗形函数幅值分布
    """
    下 hz_offset, hz_scale, vrt_scale = [455, 2, 5]，范围从 420-490
    上 hz_offset, hz_scale, vrt_scale = [690, 2.5, 4]，范围从 650-730
    """

    if area == Teeth.UL or area == Teeth.UR:
        # 后缀号表示声道
        freq_center = rt.gauss_rand(1200, float_range=10)
        freq_range = 300
        # 每间隔5Hz添加一个，中心区左右 200Hz 范围。
        # height 表示纵向缩放系数
        height = 0.7 * np.log(depth + 8) / 3
        # freq 1 是高频部分
        # freq_1 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 30), float_range=10)
        # am_1 = rt.gauss_rand(uf.bowl(freq_1, freq_center, 2.5, 3), float_range=0.05) * height - 0.1

        """
        生成信号待解决问题：
        1. 加入随机频带，可以设置几个大频带和小频带混合（而且频带还是太大，要尽可能小）
        2. 频带边缘过渡生硬
        3. 低频也要加入频带？还是高频附近频带？（实验验证）
        4. 音色问题（不能简单地只滤波，收集一个牙刷的空气信号改一下）
        
        先拉伸（乘法）multi，后偏移（加法）offset
        分为一个主 evp 和副 evp，两者随机叠加。
        
        """

        x_1, y_1 = uf.get_evp(dir_path + '\\' + str(rt.get_randint(0, csv_num)) + ".csv",
                              rt.gauss_rand(900, float_range=100), 0, rt.gauss_rand(1200, float_range=300),
                              rt.gauss_rand(10, float_range=5), pooling=2)
        x_2, y_2 = uf.get_evp(dir_path + '\\' + str(rt.get_randint(0, csv_num)) + ".csv",
                              rt.gauss_rand(1100, float_range=100), 0, rt.gauss_rand(800, float_range=200),
                              rt.gauss_rand(15, float_range=1), pooling=2)

        freq_1 = np.append(x_1, x_2)
        am_1 = np.append(y_1, y_2)
        min_val = -3
        max_val = -min_val
        float_range = max_val
        am_1 = rt.mixed_rand(am_1, min_val=min_val, max_val=max_val, float_range=float_range)

        print("freq_1 最值：", np.min(freq_1), np.max(freq_1))
        print("am_1 最值：", np.min(am_1), np.max(am_1))

        # plt.figure()
        # plt.plot(freq_1, am_1)
        # plt.show()
        # freq 2 是低频部分
        # freq_2 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 30), float_range=10)
        # am_2 = rt.gauss_rand(uf.bowl(freq_2, freq_center, 2.5, 3), float_range=0.05) * height - 0.1

        x_1, y_1 = uf.get_evp(dir_path + '\\' + str(rt.get_randint(0, csv_num)) + ".csv",
                              rt.gauss_rand(300, float_range=10), 0, rt.gauss_rand(500, float_range=100),
                              rt.gauss_rand(20, float_range=5), pooling=5)
        x_2, y_2 = uf.get_evp(dir_path + '\\' + str(rt.get_randint(0, csv_num)) + ".csv",
                              rt.gauss_rand(300, float_range=10), 0, rt.gauss_rand(450, float_range=50),
                              rt.gauss_rand(10, float_range=1), pooling=5)

        freq_2 = np.append(x_1, x_2)
        am_2 = np.append(y_1, y_2)
        min_val = -2
        max_val = -min_val
        float_range = max_val
        am_2 = rt.mixed_rand(am_2, min_val=min_val, max_val=max_val, float_range=float_range)

    elif area == Teeth.DL or area == Teeth.DR:
        # 下颌骨固频给个 501 Hz
        # height 表示纵向缩放系数
        height = 0.6 * np.log(depth + 8) / 3
        freq_center = rt.gauss_rand(501, float_range=3)
        freq_range = 260
        freq_1 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 5), float_range=5)
        am_1 = uf.bowl(freq_1, freq_center, 2, 5) * height - 0.1
        # plt.figure()
        # plt.plot(freq_1, am_1)
        # plt.show()
        freq_2 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 5), float_range=5)
        am_2 = uf.bowl(freq_2, freq_center, 2, 5) * height - 0.1
    else:
        print('tooth_area 参数不合法！')
        print('程序已退出。')
        return [], []
    # 随机找几个位置，取消掉

    idx = rt.get_rand_pos(freq_1, int(len(freq_1) / 3))
    freq_1 = np.delete(freq_1, idx)
    am_1 = np.delete(am_1, idx)
    idx = rt.get_rand_pos(freq_2, int(len(freq_2) / 3))
    freq_2 = np.delete(freq_2, idx)
    am_2 = np.delete(am_2, idx)

    phi_1 = [rt.mean_rand(0, -np.pi, np.pi)] * len(freq_1)
    phi_2 = [rt.mean_rand(0, -np.pi, np.pi)] * len(freq_2)

    # 随机找几个位置，改为随机相位
    for j in range(0, int(len(freq_1) / 4)):
        pos = rt.get_randint(0, len(phi_1))
        phi_1[pos] = rt.mean_rand(0, -np.pi, np.pi)
        pos = rt.get_randint(0, len(phi_2))
        phi_2[pos] = rt.mean_rand(0, -np.pi, np.pi)

    """
    怎么使其产生频带或产生峰？
    相位相同则为频带，相位不同则为峰。
    那么就要随机掺入几个不同的相位
    """

    # 两个 freq 不等长
    for i in range(len(freq_1)):
        data_1 = uf.add_sin(data_1, am_1[i], freq_1[i], phi_1[i])
    for i in range(len(freq_2)):
        data_1 = uf.add_sin(data_1, am_2[i], freq_2[i], phi_2[i])

    # 2月1日修改，下牙区在1200加一些频带。
    # if area == Teeth.DL or area == Teeth.DR:
    #     freq_center = rt.gauss_rand(1200, float_range=5)
    #     freq_range = 200
    #     height = 0.2 * np.log(depth + 8) / 3
    #     # 每间隔5Hz添加一个，中心区左右 200Hz 范围。
    #     freq_1 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 20), float_range=10)
    #     am_1 = rt.gauss_rand(uf.bowl(freq_1, freq_center, 2.5, 3), float_range=0.5) * height - 0.03
    #     freq_2 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 20), float_range=10)
    #     am_2 = rt.gauss_rand(uf.bowl(freq_2, freq_center, 2.5, 3), float_range=0.5) * height - 0.03
    #     phi_1 = rt.mean_rand([0] * len(freq_1), -np.pi, np.pi)
    #     phi_2 = rt.mean_rand([0] * len(freq_2), -np.pi, np.pi)
    #     for j in range(len(freq_1)):
    #         data_1 = uf.add_sin(data_1, am_1[j], freq_1[j], phi_1[j])
    #     for j in range(len(freq_2)):
    #         data_2 = uf.add_sin(data_2, am_2[j], freq_2[j], phi_2[j])
    # elif area == Teeth.UL or area == Teeth.UR:
    #     # 在上牙区加一些低频频带
    #     height = 0.1 * np.log(depth + 8) / 3
    #     freq_center = rt.gauss_rand(501, float_range=3)
    #     freq_range = 220
    #     freq_1 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 5), float_range=5)
    #     am_1 = uf.bowl(freq_1, freq_center, 2, 5) * height
    #     freq_2 = rt.gauss_rand(np.arange(freq_center - freq_range, freq_center + freq_range, 5), float_range=5)
    #     am_2 = uf.bowl(freq_2, freq_center, 2, 5) * height
    #     phi_1 = rt.mean_rand([0] * len(freq_1), -np.pi, np.pi)
    #     phi_2 = rt.mean_rand([0] * len(freq_2), -np.pi, np.pi)
    #     for j in range(len(freq_1)):
    #         data_1 = uf.add_sin(data_1, am_1[j], freq_1[j], phi_1[j])
    #     for j in range(len(freq_2)):
    #         data_2 = uf.add_sin(data_2, am_2[j], freq_2[j], phi_2[j])

    # 【四】加入全局噪声
    # data_1 = rt.gauss_rand(data_1, float_range=1)

    return data_1, data_2


"""
随机幅度大的原因：
50Hz 的振幅随机性太大，现假设统一认为 50Hz 的振幅为 5?

Discussion and Limitation:
1. 只能在外侧面水平方向刷牙，在刷牙面上有局限性
2. 只有在特定压力区间内，功率才能稳定输出，在实际使用场景中，按压力度可能并不稳定
3. 通过键盘敲击声获取用户身份信息

"""


# 【生成 500Hz 的随机信号】
# 现在用的是这个版本
def gen_random_signal_2(uname, area, depth, num, src_path, gen_path, conf_dg=0.2, rand=rt.GAUSS):
    # 步骤（一）：随机截取时域信号
    fs, signal = rt.get_rand_len_signal(src_path)

    data_1 = signal[..., 0]
    # 去掉前后 1/10 部分
    start_idx = int(len(data_1) / 10)
    end_idx = -int(len(data_1) / 10)
    data_1 = data_1[start_idx:end_idx]
    data_2 = signal[..., 1]
    data_2 = data_2[start_idx:end_idx]

    # 先寻找主峰
    peak_freq = uf.get_main_peak_freq(data_1, fs)

    # 变频
    # data_1 = f.get_mean_pooling(data_1, 1.735)
    data_1 = f.get_mean_pooling(data_1, 517 / peak_freq)
    # data_2 = f.get_mean_pooling(data_2, 1.735)
    data_2 = f.get_mean_pooling(data_2, 517 / peak_freq)

    # print("变频后显示")
    # v.show_am_time(data_1)
    # v.show_am_freq(data_1)

    # v.show_am_freq(data_1)

    # 步骤（三）：加入固有频率
    # data_1, data_2 = uf.exchange_val(data_1, data_2)
    # data_1, data_2 = nature_freq_adjust(data_1, data_2, area, depth)

    # 固频调整后，裁掉头 1%
    # data_1 = data_1[int(len(data_1) / 100):]
    # data_2 = data_2[int(len(data_2) / 100):]

    if len(data_1) <= 0 or len(data_2) <= 0:
        return False

    # 加入固频导致出现开头的一个峰

    # print("固频调整后显示")
    # v.show_am_time(data_1)
    # v.show_am_freq(data_1)

    # 步骤（三）：主峰功率调整
    # 这里直接全功率调整了
    power_1 = p.get_power(data_1)
    power_2 = p.get_power(data_2)

    # 加入正弦
    dir_path = r"C:\Users\xenon\Desktop\evp"
    csv_num = 19

    # 左声道是否为主声道
    left_is_main = True

    # 根据深度自适应(最好是带个log之类的东西)
    # conf_dg += depth / 200

    # variance = 20 * conf_dg
    # 浮动范围与真实值有关
    x, real_am = m.get_model(uname, area)
    # real_am = np.array([rt.mean_rand(20000, -5000, 5000)] * 8)
    # 控制波动中心
    real_am = rt.gauss_rand(real_am, float_range=np.mean(real_am) / 50)
    float_range = real_am[-1] * conf_dg

    # data_1 = p.power_adjust(data_1, rt.gauss_rand(
    #     real_am[depth - 1],
    #     float_range=float_range))
    # data_2 = p.power_adjust(data_2, rt.gauss_rand(21000,
    #                                               float_range=float_range))

    # 最终写入的是整体功率调整，而非局部功率调整
    """
        局部功率调整的过程是带通和带阻相加
    """
    # 执行调整过程
    if power_1 > power_2:
        # 原始信号，左声道为主声道
        left_is_main = True
        # 调整时，还要计算功率比
        # ratio = p.get_freq_power_by_spectrum(data_1, 497, 537, fs) / p.get_power(data_1) / 2
        # ratio = p.get_freq_power_by_spectrum(data_1, 497, 537, fs) / p.get_power(data_1) / 2
        ratio = p.get_power(data_1) / p.get_freq_power_by_spectrum(data_1, 497, 537, fs)
        obj_power = 0
        if rand == rt.GAUSS:
            # data_1 = p.freq_power_adjust(data_1, rt.gauss_rand(
            #     real_am[depth - 1], variance=np.log(depth + 2) * 2 * conf_dg,
            #     float_range=float_range), 57.5, -1)
            # 解释一下为什么 log 里面要 +1，因为 ln1 = 0
            obj_power = ratio * rt.gauss_rand(real_am[depth - 1], float_range=float_range)
        elif rand == rt.MEAN:
            obj_power = ratio * rt.mean_rand(real_am[depth - 1], min_val=-float_range, max_val=float_range)
        data_1 = p.power_adjust(data_1, obj_power)
        data_2 = p.power_adjust(data_2, ratio * rt.gauss_rand(
            real_am[7] * 0.9,
            float_range=float_range))

    else:
        # 右声道为主声道
        left_is_main = False
        ratio = p.get_power(data_2) / p.get_freq_power_by_spectrum(data_2, 497, 537, fs)
        obj_power = 0
        if rand == rt.GAUSS:
            # data_2 = p.freq_power_adjust(data_2, rt.gauss_rand(
            #     real_am[depth - 1], variance=np.log(depth + 2) * 2 * conf_dg,
            #     float_range=float_range), 57.5, -1)
            obj_power = ratio * rt.gauss_rand(real_am[depth - 1], float_range=float_range)
        elif rand == rt.MEAN:
            # data_2 = p.freq_power_adjust(data_2, rt.mean_rand(
            #     real_am[depth - 1],
            #     min_val=-float_range,
            #     max_val=float_range), 57.5, -1)
            obj_power = ratio * rt.mean_rand(real_am[depth - 1], min_val=-float_range, max_val=float_range)
        data_2 = p.power_adjust(data_2, obj_power)
        data_1 = p.power_adjust(data_1, ratio * rt.mean_rand(
            real_am[7] * 0.9,
            min_val=-float_range,
            max_val=float_range))

    # print("功率调整后显示")
    # v.show_am_time(data_1)
    # v.show_am_freq(data_1)
    # v.show_am_time(data_2)
    # v.show_am_freq(data_2)

    # 【二】加入 50Hz 正弦
    am_1 = rt.gauss_rand(2, float_range=0.5)
    freq_1 = np.arange(50, 2000, 50)
    phi_1 = rt.mean_rand(0, -np.pi, np.pi)

    am_2 = rt.gauss_rand(2, float_range=0.5)
    freq_2 = np.arange(50, 20000, 50)
    phi_2 = rt.mean_rand(0, -np.pi, np.pi)

    for i in range(len(freq_1)):
        data_1 = uf.add_sin(data_1, am_1 * 5 / np.exp(freq_1[i] / 50), freq_1[i], phi_1)
        data_2 = uf.add_sin(data_2, am_2 * 5 / np.exp(freq_2[i] / 50), freq_2[i], phi_2)

        # 直流分量
        # freq_1, am_1 = uf.get_evp(dir_path + '\\' + str(rt.get_randint(0, csv_num)) + ".csv",
        #                           rt.gauss_rand(3, float_range=0.5), 0,
        #                           rt.gauss_rand(60, float_range=2),
        #                           rt.gauss_rand(1, float_range=0.5), pooling=10)
        # freq_2, am_2 = uf.get_evp(dir_path + '\\' + str(rt.get_randint(0, csv_num)) + ".csv",
        #                           rt.gauss_rand(3, float_range=0.5), 0,
        #                           rt.gauss_rand(60, float_range=2),
        #                           rt.gauss_rand(1, float_range=0.5), pooling=10)
        #
        # phi_1 = rt.mean_rand([0] * len(freq_1), -np.pi, np.pi)
        # phi_2 = rt.mean_rand([0] * len(freq_2), -np.pi, np.pi)
        #
        # for j in range(np.min([len(freq_1), len(freq_2)])):
        #     # 两者索引数不同
        #     # data_1 = uf.add_sin(data_1, am_1 * 0.1 / freq_1[j], freq_1[j], phi_1[j])
        #     data_1 = uf.add_sin(data_1, am_1[j], freq_1[j], phi_1[j])
        #     data_2 = uf.add_sin(data_2, am_2[j], freq_2[j], phi_2[j])

    """
        文件命名规则：
            [志愿者名]-[牙区]-[深度]-[编号(3位数)].wav
            例如：WY-UL-2-001.wav
    """
    # 写入文件
    file_name = uname + '-' + area + '-' + str(depth) + '-' + str(num).zfill(3) + '.wav'
    if left_is_main and (area == Teeth.UL or area == Teeth.DL) \
            or (not left_is_main and (area == Teeth.UR or area == Teeth.DR)):
        # 不需要交换声道的情况：左主且左区，右主且右区
        signal_write(data_1, data_2, fs, gen_path + '\\' + file_name)
        # signal_write(data_1, data_2, fs, gen_path + '/' + file_name)
    else:
        # 否则需要交换声道：左主且右区，右主且左区
        signal_write(data_2, data_1, fs, gen_path + '\\' + file_name)
        # signal_write(data_2, data_1, fs, gen_path + '/' + file_name)
    return True


# 累加频带
def add_nature_freq(path=r"C:\Users\xenon\Desktop\gen\HST-UL\HST-UL-1-001.wav"):
    teeth = file2teeth(path)
    return


# 滑动信号能量
def gen_slide_signal():
    # uf.csv_zeroing(r"C:\Users\xenon\Desktop\slide.csv")
    # x, y = uf.get_evp(r"C:\Users\xenon\Desktop\slide.csv", 0, 0, 2.2, 9258)
    x, y = uf.get_evp(r"C:\Users\xenon\Desktop\evp\2.csv", 0, 0, 2.2, 9258)
    y = np.array(y)
    y = y + 29816

    path = r"C:\Users\xenon\Desktop\清洁3~1.wav"
    fs, signal = wavfile.read(path)
    data_1 = signal[..., 0]
    data_1 = data_1[int(10 * fs):int(17.5 * fs)]
    wnd = 2000
    # total_len = 22
    total_len = 50
    evp_wnd = int(len(y) / total_len)
    y_mean = []
    for j in range(0, total_len):
        y_mean.append(np.mean(y[int(j * evp_wnd):int((j + 1) * evp_wnd)]))
        data_1[int(j * wnd):int((j + 1) * wnd)] = p.power_adjust(data_1[int(j * wnd):int((j + 1) * wnd)],
                                                                 np.mean(y[int(j * evp_wnd):int((j + 1) * evp_wnd)]))
    v.show_am_time(data_1[0:(total_len - 1) * wnd])

    plt.figure(0)
    x_ticks = np.arange(0, len(y_mean))
    plt.plot(x_ticks, y_mean)
    plt.xticks(x_ticks, np.around(np.arange(0.1, 2.3, 0.1), 1))
    plt.show()

    return


# 获取当前牙位的，用于模型判别的功率
def get_model_power(teeth):
    # 517 左右各 20Hz 范围，左右牙区各自计算不同的声道
    if teeth.area == Teeth.DL or teeth.area == Teeth.UL:
        power = p.get_freq_power_by_spectrum(teeth.data_1, 497, 537)
        # power = p.get_freq_power_by_spectrum(teeth.data_1, 260, 300)
        # power = p.get_power(teeth.data_1)
    elif teeth.area == Teeth.DR or teeth.area == Teeth.UR:
        power = p.get_freq_power_by_spectrum(teeth.data_2, 497, 537)
        # power = p.get_freq_power_by_spectrum(teeth.data_2, 260, 300)
        # power = p.get_power(teeth.data_2)
    else:
        power = -1
    return power


# 判断当前牙位（区内）
def recognize_model_position(teeth):
    x, y = m.get_model(teeth.uname, teeth.area)
    # 517 左右各 20Hz 范围，左右牙区各自计算不同的声道
    power = get_model_power(teeth)
    if power == -1:
        print("错误！牙区输入不正确！")
        return -1
    idx = np.argmin(np.abs(y - power))
    # 若报错 'int' 说明模型未录入
    return x[idx]


# 显示数据的拟合处理结果（区内判别算法）
# 传入参数为一整个 tooth_list
# _filter 表示数据处理是否滤波
def show_power_fit(tooth_list=[], c=86.39644761, d=4794.87116189, _filter=True):
    # [86.39644761 4794.87116189]
    """
        先对每个牙齿数据进行滤波（声道排除）
        然后计算功率，存入一个数组 y
        通过 x 和 y 绘制散点图和拟合曲线
        再计算准确率，准确率通过最近点查找算法找到：
            1. 计算 8 个离散点的期望值
            2. 通过循环找到最近点的值，并判断是否正确
            3. 根据结果绘制混淆矩阵，并计算综合准确率
    """
    depth_list = []
    # power_list 用于散点图
    power_list = []
    # 各深度的数据个数
    depth_num_list = [0] * 8
    # 最大行的索引值
    max_row = 1
    # 创建指定维度空数组的方法
    pred_mat = np.array([[0] * 8], dtype=int)
    # 正确预测计数
    right_num = 0
    print('正在处理数据：')
    for teeth in tooth_list:
        """
            pred_mat 一开始为空 [[0]*8]
            共享一个当前的最大行数 max_row，存储当前的最大行数，这一行是 “不可以” 写入的
                也就是说，max_row-1 才是当前能写入的最大行数
            存储一个各向量单独使用的当前 pred_mat_idx 数组
            若 max_row == pred_mat_idx 则数组扩充一行 0，max_row += 1
                具体操作形如：
                a = np.append(a, [0] * 8)
                max_row += 1
                a.shape = (max_row, 8)
            0 表示该列的终结
            新的数据
        """

        # 根据牙区计算主声道功率
        # 主要的性能瓶颈在滤波这一步
        # 9月更新方案：直接计算总功率，不滤波处理。
        if _filter:
            if teeth.area == Teeth.UL or teeth.area == Teeth.DL:
                # power_list = np.append(power_list, p.get_freq_power(teeth.data_1, 57.5, -1))
                power_list = np.append(power_list, p.get_power(teeth.data_1))
            else:
                # power_list = np.append(power_list, p.get_freq_power(teeth.data_2, 57.5, -1))
                power_list = np.append(power_list, p.get_power(teeth.data_2))
        else:
            power_list = np.append(power_list, p.get_power(teeth.data_1) + p.get_power(teeth.data_2))
            # if teeth.area == Teeth.UL or teeth.area == Teeth.DL:
            #     power_list = np.append(power_list, p.get_power(teeth.data_1))
            # else:
            #     power_list = np.append(power_list, p.get_power(teeth.data_2))

        """
            线程同步的主要问题：
                power_list 和 depth_list 是同步索引值
            这个 depth_list 除了用于 scatter 绘图以外没别的用
        """
        depth_list.append(teeth.depth)

        if depth_num_list[teeth.depth - 1] >= max_row:
            pred_mat = np.append(pred_mat, [[0] * 8])
            max_row += 1
            pred_mat.shape = (max_row, 8)
        # 将预测值写入预测矩阵中

        # pred_mat[depth_num_list[teeth.depth - 1]][teeth.depth - 1] = pred_depth(power_list[-1])
        if pred_mat[depth_num_list[teeth.depth - 1]][teeth.depth - 1] == teeth.depth:
            right_num += 1
        depth_num_list[teeth.depth - 1] += 1
        v.show_progress(len(power_list), len(tooth_list))
    print('深度预测准确率为：')
    print(right_num / len(tooth_list))

    # 显示拟合的结果图
    print("当前显示的窗口为：原合成曲线")
    x = np.arange(1, 9)
    step = (np.max(x) - np.min(x)) / 100
    start = np.min(x) - step * 10
    end = np.max(x) + step * 10
    x_ = np.arange(start, end, step)
    # 在这里修改判别公式
    y_ = uf.spherical_attenuation(x_, c=c, d=d)
    plt.figure('curve-fit-result')
    plt.plot(x_, y_, c='black', linestyle='--')
    plt.scatter(depth_list, power_list, c='blue', marker='+')
    plt.title('fitting comparison')
    plt.show()

    print("当前显示的窗口为：拟合曲线")
    v.show_curve_fit(depth_list, power_list, uf.spherical_attenuation_4param)

    # 显示混淆矩阵
    v.show_conf_mat(range(1, 9), pred_mat[0:np.min(depth_num_list)])
    # 准确率的混淆矩阵怎么传入参数？
    """
        那就意味着输入的数据必须是每个值都判断的一个完整的数组
    """
    return


def show_power_fit_7tooth(tooth_list=[], c=86.39644761, d=4794.87116189, _filter=True):
    depth_list = []
    # power_list 用于散点图
    power_list = []
    # 各深度的数据个数
    depth_num_list = [0] * 7
    # 最大行的索引值
    max_row = 1
    # 创建指定维度空数组的方法
    pred_mat = np.array([[0] * 7], dtype=int)
    # 正确预测计数
    right_num = 0
    print('正在处理数据：')
    for teeth in tooth_list:
        depth_list.append(teeth.depth)
        # 将 1 牙区归 2
        if teeth.depth == 1:
            teeth.depth = 2
        # 根据牙区计算主声道功率
        if _filter:
            # 主要的性能瓶颈在滤波这一步
            if teeth.area == Teeth.UL or teeth.area == Teeth.DL:
                # power_list = np.append(power_list, p.get_freq_power(teeth.data_1, 57.5, -1))
                power_list = np.append(power_list, p.get_power(teeth.data_1))
            else:
                # power_list = np.append(power_list, p.get_freq_power(teeth.data_2, 57.5, -1))
                power_list = np.append(power_list, p.get_power(teeth.data_2))
        else:
            power_list = np.append(power_list, p.get_power(teeth.data_1) + p.get_power(teeth.data_2))

        # 列表长度改为 7 后，索引改为 -2
        if depth_num_list[teeth.depth - 2] >= max_row:
            pred_mat = np.append(pred_mat, [[0] * 7])
            max_row += 1
            pred_mat.shape = (max_row, 7)
        # 计算预测值，并将 1 深度改为 2
        # p_depth = pred_depth(power_list[-1])
        if p_depth == 1:
            p_depth = 2
        # 将预测值写入预测矩阵中
        pred_mat[depth_num_list[teeth.depth - 2]][teeth.depth - 2] = p_depth
        if pred_mat[depth_num_list[teeth.depth - 2]][teeth.depth - 2] == teeth.depth:
            right_num += 1
        depth_num_list[teeth.depth - 2] += 1
        v.show_progress(len(power_list), len(tooth_list))

    print('深度预测准确率为：')
    print(right_num / len(tooth_list))

    # 显示拟合的结果图
    print("当前显示的窗口为：原曲线")
    x = np.arange(1, 9)
    step = (np.max(x) - np.min(x)) / 100
    start = np.min(x) - step * 10
    end = np.max(x) + step * 10
    x_ = np.arange(start, end, step)
    y_ = uf.spherical_attenuation(x_, c=c, d=d)
    plt.figure('curve-fit-result')
    plt.plot(x_, y_, c='black', linestyle='--')
    # 画横向参考线
    # for i in range(0, len(real_am)):
    #     plt.plot([0, 8], [real_am[i], real_am[i]], color='gray', linestyle=':')
    plt.scatter(depth_list, power_list, c='blue', marker='+')
    plt.title('fitting comparison')
    plt.show()

    print("当前显示的窗口为：拟合曲线")
    v.show_curve_fit(depth_list, power_list, uf.spherical_attenuation_4param)

    # 显示混淆矩阵
    v.show_conf_mat(range(2, 9), pred_mat[0:np.min(depth_num_list)])

    return


# 【显示固频及拟合效果】
def show_nature_freq(path=r"D:\python生成的数据\WY-UL-8-030.wav"):
    fs, signal = wavfile.read(path)
    data = signal[..., 0]
    am, freq, phi = v.get_fft_result(data, fs)
    idx_max = np.argmax(am)

    # 左右取数组
    start_l = uf.get_nearest_idx(freq, 250)
    end_l = idx_max - 4
    start_r = idx_max + 8
    end_r = uf.get_nearest_idx(freq, 900)
    freq_l = freq[start_l:end_l]
    am_l = am[start_l:end_l]
    freq_r = freq[start_r:end_r]
    am_r = am[start_r:end_r]

    # 拟合
    n = 9
    p_param_l = np.polyfit(freq_l, am_l, n)
    p_param_r = np.polyfit(freq_r, am_r, n)
    p_fun_l = np.poly1d(p_param_l)
    p_fun_r = np.poly1d(p_param_r)

    # 采样（画图用）
    step_l = (freq_l[-1] - freq_l[0]) / 1000
    step_r = (freq_r[-1] - freq_r[0]) / 1000
    freq_l_curve = np.arange(freq_l[0] - 5 * step_l, freq_l[-1] + 5 * step_l, step_l)
    am_l_curve = p_fun_l(freq_l_curve)
    freq_r_curve = np.arange(freq_r[0] - 5 * step_r, freq_r[-1] + 5 * step_r, step_r)
    am_r_curve = p_fun_r(freq_r_curve)

    # 画图
    plt.figure('curve_fit', (6, 4))
    plt.grid(axis='y')
    # plt.xticks(range(1, 9), range(1, 9))
    plt.plot(freq_l_curve, am_l_curve, c='black', linestyle='--', alpha=0.6, linewidth=0.9, label='left fitting curve')
    plt.plot(freq_r_curve, am_r_curve, c='black', linestyle=':', alpha=0.6, linewidth=0.9, label='right fitting curve')
    plt.plot(freq, am, c='blue', label='raw data', alpha=0.8, linewidth=0.8)
    plt.xlim((250, 900))
    plt.ylim((0, 8))
    plt.legend()
    plt.show()

    # 求左右方差之差。方差越大表示波动越大
    l_mse = get_mse(freq_l, am_l, p_fun_l)
    r_mse = get_mse(freq_r, am_r, p_fun_r)
    print(r_mse - l_mse)

    # idx_l = idx_max - 50
    # idx_r = idx_max - 2
    # freq_local = freq[idx_l:idx_r]
    # am_local = am[idx_l:idx_r]
    # v.show_poly_fit(freq_local, am_local, 9)

    # idx_l = idx_max + 2
    # idx_r = idx_max + 50
    # freq_local = freq[idx_l:idx_r]
    # am_local = am[idx_l:idx_r]
    # v.show_poly_fit(freq_local, am_local, 9)

    return


# 【固频判别法】
def nature_freq_distinguish():
    start_time = time.time()
    # 批量读入
    path = r"F:\python生成的数据"
    t_list = dir2_tooth_list(path)
    upper_depth = []
    upper_mse = []
    lower_depth = []
    lower_mse = []
    """
    原来是按照牙齿标号作为横轴来的
    """
    print("正在计算MSE：")
    i = 0
    for t in t_list:
        # print(t.area, t.depth, t.num)
        v.show_progress(i, len(t_list))
        mse = get_teeth_mse(t)
        if not mse == []:
            if t.area == Teeth.UL or t.area == Teeth.UR:
                upper_mse.append(mse)
                upper_depth.append(t.depth)
            elif t.area == Teeth.DL or t.area == Teeth.DR:
                lower_mse.append(mse)
                lower_depth.append(t.depth)
        else:
            # 无效数据的情况
            pass
        i += 1
    v.show_progress(1, 1)
    mean_th = uf.get_mean_th(upper_mse, lower_mse)

    right_num = 0

    # 计算准确率
    for mse in upper_mse:
        if mse < mean_th:
            right_num += 1
    for mse in lower_mse:
        if mse > mean_th:
            right_num += 1
    print("准确率为：", right_num / (len(upper_mse) + len(lower_mse)))

    stop_time = time.time()
    print('用时：', round(stop_time - start_time), 'sec')

    plt.figure("nature frequency mse")
    plt.scatter(upper_depth, upper_mse, alpha=0.6, marker="+", color="blue", label="upper area")
    plt.scatter(lower_depth, lower_mse, alpha=0.6, marker="x", color="black", label="lower area")
    plt.plot((1, 8), (mean_th, mean_th), linestyle='--', color='red')
    plt.xlabel("depth")
    plt.ylabel("mse")
    plt.legend()
    plt.show()
    return


# 【固频判别法2】使用二阶差分方案
def nature_freq_distinguish_2():
    start_time = time.time()
    # 批量读入
    # path = r"F:\python的数据"
    path = r"/Volumes/UGREEN/517Hz牙刷-ZR"
    t_list = dir2_tooth_list(path)
    print(t_list)
    upper_depth = []
    upper_diff = []
    lower_depth = []
    lower_diff = []
    """
    原来是按照牙齿标号作为横轴来的
    """
    i = 0
    for t in t_list:
        # print(t.area, t.depth, t.num)
        v.show_progress(i, len(t_list))
        diff = get_diff_sum(t)
        if diff:
            if t.area == Teeth.UL or t.area == Teeth.UR:
                upper_diff.append(diff)
                upper_depth.append(t.depth)
            elif t.area == Teeth.DL or t.area == Teeth.DR:
                lower_diff.append(diff)
                lower_depth.append(t.depth)
        else:
            # 无效数据的情况
            pass
        i += 1
    v.show_progress(1, 1)
    # print(upper_diff)
    # print(upper_depth)
    # print(lower_diff)
    # print(lower_depth)
    mean_th = uf.get_mean_th(upper_diff, lower_diff)

    right_num = 0

    # 计算准确率
    for diff in upper_diff:
        if diff < mean_th:
            right_num += 1
    for diff in lower_diff:
        if diff > mean_th:
            right_num += 1
    print("准确率为：", right_num / (len(upper_diff) + len(lower_diff)))

    stop_time = time.time()
    print('用时：', round(stop_time - start_time), 'sec')

    plt.figure("nature frequency diff")
    # ValueError: setting an array element with a sequence. 准确率为： 0.0
    plt.scatter(upper_depth, upper_diff, alpha=0.6, marker="+", color="blue", label="upper area")
    plt.scatter(lower_depth, lower_diff, alpha=0.6, marker="x", color="black", label="lower area")
    plt.plot((1, 8), (mean_th, mean_th), linestyle='--', color='red')
    plt.xlabel("depth")
    plt.ylabel("diff")
    plt.legend()
    plt.show()
    return


# 【固频判别法3】使用1200Hz功率
def nature_freq_distinguish_3():
    start_time = time.time()
    # 批量读入
    # path = r"F:\python的数据"
    path = r"C:\Users\xenon\Desktop\部分数据"
    t_list = dir2_tooth_list(path)
    print(t_list)
    upper_depth = []
    upper_diff = []
    lower_depth = []
    lower_diff = []
    """
    原来是按照牙齿标号作为横轴来的
    """
    i = 0
    for t in t_list:
        # print(t.area, t.depth, t.num)
        v.show_progress(i, len(t_list))
        diff = get_square_sum(t)
        if diff:
            if t.area == Teeth.UL or t.area == Teeth.UR:
                upper_diff.append(diff)
                upper_depth.append(t.depth)
            elif t.area == Teeth.DL or t.area == Teeth.DR:
                lower_diff.append(diff)
                lower_depth.append(t.depth)
        else:
            # 无效数据的情况
            pass
        i += 1
    v.show_progress(1, 1)
    mean_th = uf.get_mean_th(upper_diff, lower_diff)

    right_num = 0

    # 计算准确率
    for diff in upper_diff:
        if diff < mean_th:
            right_num += 1
    for diff in lower_diff:
        if diff > mean_th:
            right_num += 1
    print("准确率为：", right_num / (len(upper_diff) + len(lower_diff)))

    stop_time = time.time()
    print('用时：', round(stop_time - start_time), 'sec')

    plt.figure("nature frequency diff")
    # ValueError: setting an array element with a sequence. 准确率为： 0.0
    plt.scatter(upper_depth, upper_diff, alpha=0.6, marker="+", color="blue", label="upper area")
    plt.scatter(lower_depth, lower_diff, alpha=0.6, marker="x", color="black", label="lower area")
    plt.plot((1, 8), (mean_th, mean_th), linestyle='--', color='red')
    plt.xlabel("depth")
    plt.ylabel("diff")
    plt.legend()
    plt.show()
    return


# 【计算平滑度】
# 输入参数为：原始点、拟合函数
def get_mse(x, y, fun):
    if len(x) != len(y):
        print("错误！输入的 x、y 不等长！")
        return 0
    # 拟合值采样
    y_ = fun(x)
    return np.mean((y_ - y) ** 2)


# 【获取一颗牙齿的 MSE】
def get_teeth_mse(teeth, fs=44100):
    mse = np.array([], dtype=float)
    # 循环两个声道
    for data in (teeth.data_1, teeth.data_2):
        # AttributeError: 'NoneType' object has no attribute 'data_1'
        am, freq, phi = v.get_fft_result(data, fs)
        """
            这个是找到峰值了，但是出现了其他问题。
            提示说，polyfit 的 x 位置必须是非空向量
        """
        idx_max = np.argmax(am)
        # 左右取数组
        start_l = uf.get_nearest_idx(freq, 250)
        end_l = idx_max - 4
        start_r = idx_max + 8
        end_r = uf.get_nearest_idx(freq, 900)

        # start_l = uf.get_nearest_idx(freq, 317)
        # end_l = uf.get_nearest_idx(freq, 507)
        # start_r = uf.get_nearest_idx(freq, 527)
        # end_r = uf.get_nearest_idx(freq, 717)

        # 数据校验：无峰值或峰值不在区间内
        if uf.get_peaks(am) == [[], []] or start_l >= end_l or start_r >= end_r:
            print("无效数据", teeth.uname, teeth.area, teeth.depth, teeth.num)
            print("请及时删除。")
            return []
        freq_l = freq[start_l:end_l]
        am_l = am[start_l:end_l]
        freq_r = freq[start_r:end_r]
        am_r = am[start_r:end_r]
        # 拟合
        n = 9
        p_param_l = np.polyfit(freq_l, am_l, n)
        p_param_r = np.polyfit(freq_r, am_r, n)
        p_fun_l = np.poly1d(p_param_l)
        p_fun_r = np.poly1d(p_param_r)
        mse = np.append(mse, get_mse(freq_r, am_r, p_fun_r) + get_mse(freq_l, am_l, p_fun_l))
    return np.sum(mse)


# 【获取牙齿二阶差分平方和】或者说是获取 Ra 值
def get_diff_sum(teeth, fs=44100):
    sum_ = 0
    for data in (teeth.data_1, teeth.data_2):
        # AttributeError: 'NoneType' object has no attribute 'data_1'
        am, freq, phi = v.get_fft_result(data, fs)
        """
            这个是找到峰值了，但是出现了其他问题。
            提示说，polyfit 的 x 位置必须是非空向量
        """
        # idx_max = np.argmax(am)

        # 左右取数组
        # start_l = uf.get_nearest_idx(freq, 250)
        # end_l = idx_max - 4
        # start_r = idx_max + 8
        # end_r = uf.get_nearest_idx(freq, 900)

        # 主峰左侧，偏移 10-300，实测偏 200、300 均不行，判别率只有 66%
        # start_l = uf.get_nearest_idx(freq, 401)
        # end_l = uf.get_nearest_idx(freq, 497)

        # 主峰右侧
        # start_r = uf.get_nearest_idx(freq, 537)
        # end_r = uf.get_nearest_idx(freq, 601)

        start_l = uf.get_nearest_idx(freq, 450)
        end_l = uf.get_nearest_idx(freq, 550)
        start_r = uf.get_nearest_idx(freq, 1000)
        end_r = uf.get_nearest_idx(freq, 1200)

        start_h = uf.get_nearest_idx(freq, 1100)
        end_h = uf.get_nearest_idx(freq, 1300)

        # 数据校验：无峰值或峰值不在区间内
        # if uf.get_peaks(am) == [[], []] or start_l >= end_l or start_r >= end_r:
        #     print("无效数据", teeth.uname, teeth.area, teeth.depth, teeth.num)
        #     print("请及时删除。")
        #     return []

        am_l = am[start_l:end_l]
        am_r = am[start_r:end_r]

        am_h = am[start_h:end_h]

        # 二阶差分平方和
        data_l = np.sum(np.diff(am_l, 2) ** 2)
        data_r = np.sum(np.diff(am_r, 2) ** 2)
        # data_l = 0
        # data_r = 0
        data_h = np.sum(np.diff(am_h, 2) ** 2)
        # print("声道功率是：", p.get_power(am_h) * len(am_h))
        # data_h = 0
        # 低减高
        # data_l, data_r = 0, 0
        data_h = 0
        sum_ += data_r + data_l - data_h
    # print(sum_)
    return sum_


# 获取牙齿 1000-1400Hz 平方和
def get_square_sum(teeth, fs=44100):
    sum_ = 0
    for data in (teeth.data_1, teeth.data_2):
        # AttributeError: 'NoneType' object has no attribute 'data_1'
        sum_ += p.get_freq_power_by_spectrum(data, 1000, 1400, fs)
    # print(sum_)
    return sum_


"""
EarSense 方法的核心思想：
一个滑动窗口计算一个序列两个声道之间的延迟，并创建一个 profile。
这个延迟 profile 就是 DeltaT slide，是一个数组
该数组的第 i 个元素计算方法为：

现在的问题是该公式怎么写？argmax 只有一个参数，就是找到当前数组的索引。

相关系数的计算方法：使用 pandas 封装序列

#两个变量计算#
import pandas as pd

# 定义两个序列
A=[1,3,6,9,0,3]
B=[3,5,1,4,11,3]

# 封装为 pd 格式
A1=pd.Series(A)
B1=pd.Series(B)

# 计算 pearson 或 spearman
corr=B1.corr(A1,method=‘pearson’)
print(corr)

# DataFrame 计算 #
import pandas as pd
# 直接定义封装序列
data=pd.DataFrame({‘a’:[1,3,6,9,0,3],‘b’:[3,5,1,4,11,3]})
# 计算
corr=data.corr(method=‘pearson’)
print(corr)

DeltaT_slide[i]=np.argmax()

公式中的 S'li 和 S'ri 分别为 (i-1)窗口长度/2:(i+1)窗口长度/2
窗口长度是一个常量 win=441

"""


# 频带功率变化
def get_band_power_change(data, freq=(250, 300), wnd=4410, fs=44100):
    wnd_time = np.around(wnd / fs, 2)
    power = []
    moment = []
    for j in range(0, int(len(data) / wnd) - 1):
        power.append(p.get_freq_power_by_spectrum(data[j * wnd:(j + 1) * wnd], freq[0], freq[1], fs))
        moment.append((j + 1) * wnd_time)
    return moment, power


def ear_sense_method():
    path = r"/Users/wangyang/Library/CloudStorage/OneDrive-个人/_真实数据/【右下】/L2.wav"
    fs, signal = wavfile.read(path)

    # 两个声道
    Sl = signal[..., 0]
    Sr = signal[..., 1]

    # 定义 win 和 profile
    win = 4410
    profile = []

    # 337920 且数量级也不同
    print(len(Sl), "   ", Sl)
    print(len(Sr), "   ", Sr)

    # 改 0-20
    for i in range(20):
        # 这里 i+1 的意思是，论文公式中的 i 下标从 1 开始，程序中下标从 0 开始。
        left_idx = int((i + 1 - 1) * win / 2)
        right_idx = int((i + 1 + 1) * win / 2)

        # 切割序列
        Sl_ = Sl[left_idx:right_idx]
        Sr_ = Sr[left_idx:right_idx + win]

        corr_sequence = []
        # 先将 corr 的序列计算出来，再从中找到 argmax
        for k in range(len(Sr_)):
            # 计算 Sl[0:end] 及 Sr[k:end] 之间的相关系数
            # pearson 相关
            # corr = Sl_.corr(Sr_[k:], "pearson")
            # corrcoef 的参数可以是一维数组，也可以是二维的
            # print(len(Sl_))
            # print(len(Sr_[k:k + win]))
            if len(Sl_) != len(Sr_[k:k + win]):
                # print("结束", i, "   ", k)
                break
            corr = np.corrcoef(Sl_, Sr_[k:k + win])[0, 1]
            # 这样 k 值，就等于对应计算结果的索引值（corr_sequence 中的索引值）
            corr_sequence.append(corr)

        DeltaT_slide = np.argmax(corr_sequence)
        profile.append(DeltaT_slide)

    print(profile)
    return


"""
ZcD 方法：
Z 序列表示零点序列，也就是过零数
左右声道的 Z 序列相减表示 ZcD
看 ZcD 的正负元素比例，来判断相对位置。
"""


def ZcD():
    path = r"/Users/wangyang/Library/CloudStorage/OneDrive-个人/_数据/【左上】/L6.wav"
    fs, signal = wavfile.read(path)

    # 还要排除 nan 数据

    # 两个声道
    Sl = np.array(signal[..., 0])
    Sr = np.array(signal[..., 1])

    # Sl

    Zl = np.array(uf.get_zero_array(Sl))
    Zr = np.array(uf.get_zero_array(Sr))

    ZcD = Zl - Zr
    p_num = uf.get_positive_num(ZcD)

    print("ZcD 的值：", ZcD)
    print("正值数：", p_num)
    print("负值数：", len(ZcD) - p_num)

    return


"""
颌骨的几何约束为牙滑块的分类提供了机会。
在垂直滑动时，接触的牙齿变化不大。
因此ΔTs i d e几乎是常数，并且接近于0。
由于某些颌骨解剖结构的潜在变形，偏差是可能的，但这将是一个常数。
对于水平滑动，牙齿的接触部分会发生很大的变化，导致一个非恒定的延迟剖面.
因此，通过观察延迟剖面的方差，如果稳定(即接近恒定)，我们将报告一个垂直滑动，否则报告一个水平滑动。

从图中也可以看出来，这个序列应当是逐渐变化至归零的。

"""

# 部分混淆矩阵 32 颗牙
conf_mat = [
    # UL8
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UL7
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UL6
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UL5
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UL4
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UL3
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UL2
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UL1
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR1
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR2
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR3
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR4
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR5
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR6
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR7
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # UR8
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL8
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL7
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL6
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL5
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL4
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL3
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL2
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LL1
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR1
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR2
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR3
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR4
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR5
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR6
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR7
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # LR8
    # UL8,  UL7,  UL6,  UL5,  UL4,  UL3,  UL2,  UL1,  UR1,  UR2,  UR3,  UR4,  UR5,  UR6,  UR7,  UR8
    # LL8,  LL7,  LL6,  LL5,  LL4,  LL3,  LL2,  LL1,  LR1,  LR2,  LR3,  LR4,  LR5,  LR6,  LR7,  LR8
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
     1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
]


# 【计算归一化幅频积】 20-40 Hz
# 输入一个 data 找到其对应的
def get_am_freq_prod(data, low_freq=20, high_freq=40, fs=44100):
    data = np.array(data, dtype=float)
    norm_am, freq, phi = v.get_fft_result(data, fs)
    low_idx = uf.get_nearest_idx(freq, low_freq)
    high_idx = uf.get_nearest_idx(freq, high_freq)
    # mean = np.mean(norm_am[low_idx:high_idx])
    return np.sum(norm_am[low_idx:high_idx] * freq[low_idx:high_idx]) / (high_idx - low_idx)


# 【计算峰值幅频积】
# 鉴于算法可能不对，因此改用计算峰值，而非计算均值。
# n 表示获取前 n 个峰值的幅频积
def get_peak_prod(data, low_freq=20, high_freq=40, n=5, fs=44100):
    data = np.array(data, dtype=float)
    norm_am, freq, phi = v.get_fft_result(data, fs)
    low_idx = uf.get_nearest_idx(freq, low_freq)
    high_idx = uf.get_nearest_idx(freq, high_freq)
    # mean = np.mean(norm_am[low_idx:high_idx])
    peak_idx, peak_val = uf.get_peaks(norm_am, [low_idx, high_idx])
    prod = 0
    if len(peak_idx) <= n:
        uf.error_data(data, "20-40Hz 峰太少不足以计算幅频积", fs=fs)
        return -1
    for i in range(0, n):
        prod += peak_val[i] * freq[peak_idx[i]]
    return prod


"""
这个算法不能除以 mean，会导致计算的结果极为相似
问题的关键在于这个除以 mean，这个 mean 是整体信号的 mean！
重写幅频积算法
"""


# 【显示幅频积】
def show_am_freq_prod(path=r"F:\python生成的数据"):
    t_list = dir2_tooth_list(path)
    prod_u = []
    depth_u = []
    prod_d = []
    depth_d = []
    i = 0
    print("正在处理数据：")
    for t in t_list:
        v.show_progress(i, len(t_list))
        if t.area == Teeth.UL or t.area == Teeth.UR:
            prod_u.append(get_am_freq_prod(t.data_1) + get_am_freq_prod(t.data_2))
            depth_u.append(t.depth)
        elif t.area == Teeth.DL or t.area == Teeth.DR:
            prod_d.append(get_am_freq_prod(t.data_1) + get_am_freq_prod(t.data_2))
            depth_d.append(t.depth)
        else:
            print("无效的牙区值")
        i += 1
    v.show_progress(1, 1)
    # 同时还要把判别阈值计算出来。这个问题不能绘制混淆矩阵，只能计算综合识别准确率。
    th = uf.get_mean_th(prod_u, prod_d)
    print("判别阈值为：", th)
    true_num = 0
    false_num = 0
    for i in range(0, len(prod_u)):
        if prod_u[i] <= th:
            # 上牙区应当小于阈值
            true_num += 1
        else:
            false_num += 1
    for i in range(0, len(prod_d)):
        if prod_d[i] > th:
            # 下牙区应当大于阈值
            true_num += 1
        else:
            false_num += 1
    acr_rate = true_num / (true_num + false_num)
    print("综合准确率为：", acr_rate)
    plt.figure("Mean Value of Amplitude Frequency Product Distribution")
    plt.title("Mean Value of Amplitude Frequency Product Distribution")
    plt.plot([1, 8], [th, th], color='red', linestyle='--', label="threshold")
    plt.scatter(depth_u, prod_u, color='blue', marker='+', alpha=0.6, label="upper area")
    plt.scatter(depth_d, prod_d, color='black', marker='x', alpha=0.6, label="lower area")
    plt.xlabel("Depth")
    plt.ylabel("M")
    plt.legend()
    plt.show()
    return


# 【绘制共振峰分布散点图】
# 需要输入两组数据的总集合
def show_formant_scatter(tooth_list):
    depth_u = []
    depth_d = []
    # plt.figure('scatter')
    # scatter = plt.scatter(depth, u_data, c='blue', marker='+')
    # 添加图例
    # plt.legend(*scatter.legend_elements(), title="classes")
    # plt.show()
    return
