"""
该 module 用于各种信号的查看、获取
"""
import numpy as np
# from brokenaxes import brokenaxes
from scipy.fftpack import fft
import matplotlib.pyplot as plt
# 安装 PyWavelets，要在解释器中安装，不能用 pip 命令
import pywt
from scipy.optimize import curve_fit
from sklearn.metrics import confusion_matrix
import pandas as pd

import utility_functions as uf

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 18}
plt.rcParams['figure.figsize'] = (8.0, 6.0)


# 注意：pywt 的安装为 pip install PyWavelets

# 【显示并更新进度条】每调用一次都会更新一次，因此需要在程序中循环调用。
# prog 表示进度，默认是百分数，例如 prog = 20 表示百分之 20
# 注意：prog 从 0 到 100 共 101 个数，scale 的取值必须等于 prog 的最大取值。
# scale 表示进度条的满值是多少，默认是 100
# view_scale 表示显示的进度条长度，默认是 50
def show_progress(prog=100, scale=100, view_scale=40):
    view_compl_num = int(prog / scale * view_scale)
    compl = '#' * view_compl_num
    imcompl = '.' * (view_scale - view_compl_num)
    percent = (prog / scale) * 100
    if percent >= 100:
        # 如果进度完成
        percent = 100
        print("\r{:.2f}%[{}->{}]".format(percent, compl, imcompl))
        return
    # 这里不能换行，因为重写只能是光标移到行首。
    print("\r{:.2f}%[{}->{}]".format(percent, compl, imcompl), end="")
    return


# 【获取频域步进】保留 6 位小数
# 频域步进与数据长度、采样率都有关
def get_freq_step(length, fs=44100):
    return np.around(fs / (length - 1), 6)


def show_plot(x, y, title="plot line"):
    plt.figure("plot")
    plt.title(title)
    plt.plot(x, y)
    plt.show()
    return


# 【通过 FFT 获取幅频相信息】返回值为三个长度一致的数组，分别为幅、频、相。
# norm_am 表示每个频率上的归一化振幅
# freq 表示对应的频谱分布
# phi 表示相位信息
def get_fft_result(data, fs=44100):
    data = np.array(data)
    # 复系数，结果取半
    complex_c = fft(data)[0: int((len(data) - 1) / 2) + 1]
    # 获得归一化振幅，除以长度，再×2
    norm_am = np.abs(complex_c) / len(data) * 2
    # 获得相位数组
    phi = np.angle(complex_c)

    # phi = np.array([], dtype=float)
    # for i in range(0, len(complex_c)):
    #     if norm_am[i] < 1e-10:
    #         phi = np.append(phi, 0)
    #     else:
    #         phi = np.append(phi, np.angle(complex_c[i]) * 180 / np.pi)
    # 获得每个位置对应的频率数组，freq_step 表示频率粒度（步进），保留 6 位小数。

    freq_step = get_freq_step(len(data), fs)
    freq = np.around(np.arange(0, len(norm_am)), 6) * freq_step
    return norm_am, freq, phi


# 【获取峰值频率】
def get_peak_freq(am, freq):
    max_idx = np.argmax(am)
    return freq[max_idx]


"""
关于归一化：
fft 的结果是求得原数据中，某个频率下的振幅总和
归一化就是除以数据长度，再乘 2.

当幅度为零时，则相位由数值不精确度给出。
如果显示由fft计算的值，则会看到您期望为0的值实际上是1e-16或类似的值。 这是浮点计算中舍入导致的数值不精确性。
解决方案是同时计算幅度和相位，如果幅度太小，则忽略相位分量。


实测发现：
在对应频率的位置，相位有跳变。这个跳变是怎么回事？



FFT 复指数相位问题解释：
https://www.bilibili.com/video/BV1j64y1a7NV/
复指数为负，则正相位加 Pi
欧拉公式：
1. e^ix = cos x + i·sin x
2. sin x = [e^ix - e^(-ix)] / 2i
3. cos x = [e^ix + e^(-ix)] / 2
那么 sin 和复指数的幅频相对应的换算关系是什么呢？

那么实际上用复指数分解的结果需要同时含有正弦和余弦？
我们实际发现，fft 的结果正是复数，其数组索引对应了频率，但相位却是 cos 和 sin 的总和？
那么可否使用复指数进行信号的转换？


合成复信号的文章：https://www.jianshu.com/p/e3d7e74094f7
Fs =100;%采样率
N=512;%序列长度
T = 1/Fs;%采样间隔
t = 0:T:(N-1)*T;%时间序列
s1 = cos(2*pi*5.*t+pi/4)+cos(2*pi*10.*t+pi/8)+cos(2*pi*15.*t+pi/4);%信号实部
s2 = sin(2*pi*5.*t+pi/4)+sin(2*pi*10.*t+pi/8)+sin(2*pi*15.*t+pi/4);%信号虚部
ss = complex(s1,s2);%合成复信号
也就是说在这个复信号的合成过程中，上下两部分幅频相完全相同，唯有函数不同。

实际使用发现，使用复信号进行傅里叶变换的结果是，振幅变成两倍，相位不变。

书上说：https://wenku.baidu.com/view/b353c8f0c2c708a1284ac850ad02de80d4d8060d.html
采样率应为信号频率的整数倍，采样点应包含整周期，否则会产生频谱泄露的问题。
就是出现平滑的曲线。

明天要解决的问题：
1. 网上说，虽然两侧有跳变，但是中间的实际值是接近的。需要查看实际值 n * 0.441 的位置。
2. 搞明白相位到底是加还是减，为什么 sin(XXX+phi) 得到的相位是 0？

实验证明：
1. FFT 使用的是 cos 来计算相位
2. 无法使用 sin 或 cos 为原始信号滤波，因为相位无法判断，取平均值也不好使



"""


# 显示 CSV
def show_csv(path=r"C:\Users\xenon\Desktop\slide.csv"):
    x, y = uf.get_evp(r"C:\Users\xenon\Desktop\slide.csv", 0, 0, 1, 1)
    plt.figure(0)
    plt.plot(x, y)
    plt.show()
    return


# 【获取模拟信号的采样值数组】
# func 为模拟信号函数
# proc_range 表示要获取数据的范围
# step 表示步长，即精度
# show_prog 表示是否显示进度（适用于数据量较大的情况）
# 返回值有两个：x、y
def get_an_samp(func, proc_range=(-25, 25), step=0., params=(0, 0), show_prog=False):
    if show_prog:
        print("正在处理曲线采样...")
    if step <= 0.:
        step = (proc_range[1] - proc_range[0]) / 100
    x = np.arange(proc_range[0], proc_range[1] + step, step)
    y = np.array([], dtype=float)
    if params == (0, 0):
        for i in range(0, len(x)):
            y = np.append(y, func(x))
            if show_prog:
                show_progress(i, len(x) - 1)
    else:
        for i in range(0, len(x)):
            param_list = np.append(x[i], params)
            y = np.append(y, func(*param_list))
            if show_prog:
                show_progress(i, len(x) - 1)
    print("曲线采样完成")
    return x, y


# 【时域查看器】
# data1、data2 分别表示左右声道，若省略 data2 表示单声道。
def show_am_time(data1, data2=0, fs=44100, time_lim=(0, 0), am_lim=(0, 0)):
    x1 = np.arange(0, len(data1)) / fs
    plt.figure("time domain", (4, 2))
    plt.plot(x1, data1, c='blue', alpha=1, label="left")
    print("声道 1 数据：", end="")
    print(data1)

    # 以下为绘制另一声道的数据
    if uf.is_ary(data2):
        x2 = np.arange(0, len(data2)) / fs
        plt.plot(x2, data2, c='blue', alpha=1, label="right")
        print("声道 2 数据：", end="")
        print(data2)
    plt.xlabel("time(s)")
    plt.ylabel("amplitude")
    if time_lim != (0, 0):
        plt.xlim(time_lim)
    if am_lim != (0, 0):
        plt.ylim(am_lim)

    # plt.legend()
    plt.show()
    return '绘图完成。'


# 【从频率数组中获取指定频率范围的子数组】
# 输入数据为从 sv.get_freq_dm 中获取的频率数组
# 返回值为指定频率范围的子数组、数组左右索引
def get_freq_range(freq, low_freq=0., high_freq=0.):
    if low_freq <= 0 and high_freq <= 0:
        print("截止频率输入不正确，已返回原始数据。")
        return freq
    if low_freq <= 0:
        low_freq = -1
    elif high_freq <= 0:
        high_freq = -1
    cut_l_idx = 0
    cut_r_idx = 2147483647
    for i in range(0, len(freq)):
        if freq[i] > low_freq:
            cut_l_idx = i - 1
            break
    for i in range(len(freq) - 1, -1, -1):
        if freq[i] < high_freq:
            cut_r_idx = i + 1
            break
    if cut_l_idx < 0:
        cut_l_idx = 0
    if cut_r_idx > len(freq):
        cut_r_idx = len(freq)
    return freq[cut_l_idx:cut_r_idx], cut_l_idx, cut_r_idx


# 【频域查看器】绘制幅频图
# 输入数据必须是一维的
# freq 表示要显示的频率范围
def show_am_freq(data, fs=44100, freq_range=[0., 0.], am_lim=(0., 0.)):
    if freq_range[0] >= freq_range[1]:
        freq_range[0] = 0
        freq_range[1] = fs * 2
    norm_am, freq, phi = get_fft_result(data, fs)
    freq, cut_l_idx, cut_r_idx = get_freq_range(freq, freq_range[0], freq_range[1])
    norm_am = norm_am[cut_l_idx:cut_r_idx]
    plt.figure("frequency domain")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.plot(freq, norm_am, c='blue')
    if am_lim != (0., 0.):
        plt.ylim(am_lim)
    plt.show()
    return


# 【局部显示示例代码】
# def show_am_freq_local(data, freq_range=(0., 0.), fs=44100):
#     path = r"D:\OneDrive\_真实数据\【左上】\L1.wav"
#     fs, signal = wavfile.read(path)
#     data = signal[..., 0]
#     am, freq, phi = v.get_fft_result(data, fs)
#     low_idx = uf.get_nearest_idx(freq, 80)
#     high_idx = uf.get_nearest_idx(freq, 81.5)
#     freq_, am_ = f.get_up_sampling_smooth(freq[low_idx:high_idx], am[low_idx:high_idx])
#     plt.figure(1)
#     # plt.plot(freq[low_idx:high_idx], am[low_idx:high_idx])
#     plt.plot(freq_, am_)
#     plt.show()
#     return


# 【带断裂轴的频域查看器】
# def show_am_freq_broken(data, freq_bk_range, am_bk_range, fs=44100):
#     norm_am, freq, phi = get_fft_result(data, fs)
#     max_am = np.max(norm_am)
#     max_freq = freq[-1]
#     fig = plt.figure("断裂轴频谱图", (4, 3))
#     bax = brokenaxes(
#         xlims=((0, freq_bk_range[0]), (freq_bk_range[1], max_freq)),
#         ylims=((0, am_bk_range[0]), (am_bk_range[1], max_am)),
#         wspace=0.2,
#         hspace=0.2,
#         diag_color='r'
#     )
#     bax.plot(freq, norm_am, label="Frequency Domain")
#     # bax.grid()
#     # plt.xlabel("frequency(Hz)", font)
#     # plt.ylabel("normalized amplitude", font)
#     # plt.xticks(freq, freq, fontproperties='Time New Roman', size=16, weight='bold')
#     # plt.yticks(norm_am, norm_am, fontproperties='Time New Roman', size=16, weight='bold')
#     bax.legend()
#     plt.show()
#     return


"""
因为断裂轴缝隙的原因，label 偏移
"""


# 【相位查看器1】绘制幅相曲线
def show_am_phi(data, fs=44100, title="amplitude_phase_chart"):
    norm_am, freq, phi = get_fft_result(data, fs)
    plt.figure("amplitude_phase_chart")
    plt.title(title)
    plt.xlabel("phase")
    plt.ylabel("normalized amplitude")
    plt.scatter(phi, norm_am, marker="+")
    plt.show()
    pass


# 【相位查看器2】绘制相频曲线
def show_phi_freq(data, fs=44100, title="phase_frequency_chart"):
    norm_am, freq, phi = get_fft_result(data, fs)
    plt.figure("phase_frequency_chart")
    plt.title(title)
    plt.xlabel("frequency")
    plt.ylabel("phase")
    plt.plot(freq, phi)
    plt.show()
    pass


# 【模拟信号绘制】
# 传入参数为信号函数，该函数必须是一个参数、一个返回值，即自变量为 x，因变量为 y。
# proc_range 表示显示范围
# step 表示步进，即 x 的取值粒度
def show_an_signal(func, proc_range=[-25., 25.], step=0., title="analog_signal"):
    if step <= 0.:
        step = (proc_range[1] - proc_range[0]) / 500
    x, y = get_an_samp(func, proc_range, step, show_prog=True)
    plt.figure("analog_signal")
    plt.title(title)
    plt.grid(True)
    plt.plot(x, y, c='blue')
    plt.show()


# 【查看小波变换】
def show_wavelet(data, fs=44100, title='wavelet', color_lv=15):
    t = np.arange(0, len(data) / fs, 1 / fs)
    # 小波名称
    wt_name = 'cgau8'
    # 计算中心频率
    c_freq = pywt.central_frequency(wt_name)
    # 总尺度大小，最终输出的频率分辨率与该值有关
    total_scale = int(fs / 4)
    # 尺度数组
    scales = 2 * c_freq * total_scale / np.arange(total_scale, 1, -1)
    print('正在执行小波变换...')
    cwt_mat, freq = pywt.cwt(data, scales, wt_name, 1 / fs)
    print('小波变换执行完毕，正在绘制图形...')
    plt.figure('wavelet')
    plt.title(title)
    # 第 4 个参数表示颜色分级密度，若分级密度过高会导致内存不足。
    plt.contourf(t, freq, abs(cwt_mat), color_lv)
    plt.colorbar()
    plt.show()
    return


"""
混淆矩阵性质：
混淆矩阵不一定是对称的！只有两两混淆才是对称的！
若纵轴表示实际值，则每一列之和为 100%（每一行之和不一定）

"""


# 【显示混淆矩阵】
# 输入 true_val 务必为整数，且为 ”一维“ 数组
# 输入 pred_val 是一个 list，包含若干个 ndarray，其中的取值务必为与 true_val 的取值相同
# show_annotate 表示是否显示标签
# reverse 表示是否将矩阵转置
def show_conf_mat(true_val, pred_mat, ticks=[], show_annotate=True, reverse=False):
    """
    显示混淆矩阵
    :param show_annotate:
    :param true_val:输入 true_val 务必为整数，且为 ”一维“ 数组
    :param pred_mat:输入 pred_val 是一个 list，包含若干个 ndarray，其中的取值务必为与 true_val 的取值相同
    :return:无返回值
    """
    # 初始化混淆矩阵
    cm = np.zeros(shape=(len(true_val), len(true_val)), dtype=float)
    for pred_val in pred_mat:
        # cm 就是一步求出来的，不能通过列表遍历的方式求准确率。
        # 注意，真值和预测值必须是连续的，即 [0, 1, 2, 3] 不能是 [0, 1, 3, 4]，否则会出现不对称的问题。
        # print("当前混淆矩阵：")
        # print(confusion_matrix(true_val, pred_val))
        cm += confusion_matrix(true_val, pred_val)
    # 通过计算多组混淆矩阵求平均值的方式，可以看准确率，保留 2 位小数
    cm = np.around(cm / len(pred_mat), 2)
    if reverse:
        cm = np.flip(cm)

    # print("当前混淆矩阵：")
    # print(cm)

    # 初始化窗口
    plt.figure(0, (6, 4))
    plt.rc("font", family="Times New Roman", size=14)
    plt.rcParams['figure.figsize'] = (6.0, 1.0)

    # plt.imshow(cm, cmap=plt.cm.Blues)  # 根据最下面的图按自己需求更改颜色
    plt.matshow(cm, cmap=plt.cm.Blues)  # 根据最下面的图按自己需求更改颜色
    # plt.clim(0.8, 1)                    # 设置颜色条范围
    # 为什么不能设置颜色条范围？因为旁边的浅色格子会被截断。
    plt.colorbar()

    accu_rate = 0

    for i in range(len(cm)):
        # 求对角线平均值准确率
        accu_rate += cm[i][i]
        # 标记热度值
        if show_annotate:
            for j in range(len(cm)):
                plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    accu_rate /= len(cm)
    print("平均准确率：", accu_rate)

    # 设置字体大小
    font_size = 15
    plt.tick_params(labelsize=font_size)
    plt.ylabel('True Depth', fontdict={'family': 'Times New Roman', 'size': font_size})
    plt.xlabel('Predicted Depth', fontdict={'family': 'Times New Roman', 'size': font_size})
    if not ticks or len(true_val) != len(ticks):
        plt.xticks(range(0, len(true_val)), true_val)
        plt.yticks(range(0, len(true_val)), true_val)
    else:
        plt.xticks(range(0, len(true_val)), ticks, rotation=50, font="Times New Roman")
        plt.yticks(range(0, len(true_val)), ticks, rotation=50, font="Times New Roman")
    # plt.grid()
    plt.show()
    return


# 【显示散点图】
def show_scatter(x, y):
    plt.figure('scatter')
    plt.scatter(x, y, c='blue', marker='+')
    plt.show()
    return


# 【条形图绘制示例代码】
def bar_chart_example():
    ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    x_indexes = np.arange(len(ages_x))
    width = 0.33

    dev_y = [38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752]
    py_dev_y = [45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640]

    # 绘制条形图
    plt.bar(x_indexes, dev_y, width=width, label="全部开发者")
    plt.bar(x_indexes + width, py_dev_y, width=width, label="Python开发者")

    # 横纵坐标
    plt.xlabel("年龄")
    plt.ylabel("年薪")

    # 标题
    plt.title("年龄和薪水的关系")

    # 图例
    plt.legend()

    # 将横坐标转换成指定数据
    plt.xticks(ticks=x_indexes, labels=ages_x)

    plt.show()
    return


# 【将散点和函数画在一幅图里】
def show_plot_scatter(x, y, f, title='plot and scatter'):
    if len(x) != len(y):
        print('输入 x，y 长度不一致！')
        return []
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    step = (np.max(x) - np.min(x)) / 100
    start = np.min(x) - step * 10
    end = np.max(x) + step * 10
    x_ = np.arange(start, end, step)
    y_ = f(x_)
    plt.figure(title)
    plt.title(title)
    plt.grid(axis='y')
    plt.plot(x_, y_, c='black', linestyle='--')
    plt.scatter(x, y, c='blue', marker='+')

    # 自定义横纵坐标
    plt.xticks(range(1, 9), range(1, 9))
    plt.ylim((0, 50000))

    plt.show()
    return


# 【显示函数拟合的效果】同时可以显示参数，返回值即为参数
# 并返回一个拟合的参数结果
# 现在仅支持 3 参数函数
def show_curve_fit(x, y, f, title='curve_fit'):
    if len(x) != len(y):
        print('输入 x，y 长度不一致！')
        return []
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    p_opt, p_cov = curve_fit(f, x, y)
    print('拟合的参数为：')
    print(uf.list2str(p_opt))
    """
        重新设计功能：
            需要有能传入参数的函数
        因为这个函数不能传入参数，所以不能使用采样函数     
    """
    step = (np.max(x) - np.min(x)) / 100
    start = np.min(x) - step * 10
    end = np.max(x) + step * 10
    x_ = np.arange(start, end, step)
    y_ = f(x_, p_opt[0])
    # y_ = f(x_, p_opt[0], p_opt[1])
    plt.figure('curve_fit')
    plt.title(title)
    plt.grid(axis='y')
    plt.xticks(range(1, 9), range(1, 9))
    plt.plot(x_, y_, c='black', linestyle='--')
    plt.scatter(x, y, c='blue', marker='+')
    plt.ylim((0, 50000))
    plt.show()
    return p_opt


# 【给定阶数的多项式拟合】
def show_poly_fit(x, y, n, title="poly_fit"):
    if len(x) != len(y):
        print('输入 x，y 长度不一致！')
        return []
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    p_param = np.polyfit(x, y, n)
    print('拟合的参数为：')
    print(uf.list2str(p_param))
    p_fun = np.poly1d(p_param)
    # 采样
    step = (np.max(x) - np.min(x)) / 100
    start = np.min(x) - step * 10
    end = np.max(x) + step * 10
    x_ = np.arange(start, end, step)
    y_ = p_fun(x_)
    # 画图
    plt.figure('curve_fit')
    plt.title(title)
    plt.grid(axis='y')
    plt.xticks(range(1, 9), range(1, 9))
    plt.plot(x_, y_, c='black', linestyle='--', alpha=0.6, label='fitting curve')
    # plt.scatter(x, y, c='blue', marker='+')
    # 以折线的形式画出来
    plt.plot(x, y, c='blue', label='raw data')
    plt.show()
    return p_param


def show_csv(path=r"C:\Users\xenon\Desktop\包络图.csv"):
    csv_data = pd.read_csv(path)
    x = np.array(csv_data["x"])
    y = np.array(csv_data["Curve1"])
    plt.figure()
    plt.plot(x, y)
    plt.grid()
    plt.show()
    return


# def show_dwt(data):
#     wt_name = 'db8'
#     ca, cd = pywt.dwt(data, wt_name)
#     print(type(ca))
#     print(type(cd))
#     plt.figure('low_frequency')
#     plt.plot(range(len(ca)), ca)
#     plt.show()
#     plt.figure('high_frequency')
#     plt.plot(range(len(cd)), cd)
#     plt.show()
#     """
#         将这两部分信号加起来就可以还原原始信号，不过是半数的关系
#     """
#     c = ca + cd
#     plt.figure('back')
#     plt.plot(range(len(c)), c)
#     plt.show()
#     return


"""
pywt 支持的小波函数
'dmey', 'fbsp', 'shan', 'haar', 'mexh', 'morl', 'cmor'
'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8'
'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8'
'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'
'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'
'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17'
'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20'
'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22'
'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38'
"""

"""
DWT 离散小波变换的分解与重构
该方法可以对信号的特定频域进行调整？
https://www.jianshu.com/p/56733f6c0a10
执行一次 dwt 就是执行一级分解，再执行一次就是二级分解
分解的 level 数为不超过信号长度的2的n次幂
其迭代过程为：
https://blog.csdn.net/qq_42874547/article/details/106942497
怎么将信号复原：
https://blog.csdn.net/wsp_1138886114/article/details/116780542
其实就是直接使用 idwt 将两个信号输入进去，即 idwt(cA,cD,'db8')
网上说，不论 cA 还是 cD 都能执行高次分解，那么就能找到某个确定的频率。
但问题是怎么找到某个离散信号的最高频率？？？

离散小波变换步骤：
原文链接：https://blog.csdn.net/abc1234abcdefg/article/details/123517320
将信号x(n)通过具有脉冲响应h(n)的高通滤波器，过滤掉频率低于P/2的部分（信号最高频率为P），即为半带高通滤波。
根据奈奎斯特定理进行下采样，间隔一个剔除样本点，信号留下一半样本点，尺度翻倍，将这一半进行高通滤波。
进一步分解，就把高通滤波器的结果再次一分为二，进行高通滤波和低通滤波。
不断反复进行上述操作，根据自己要求调整。

猜想：DWT计算过程中不计算最高频，需要自己通过采样率计算
例如采样率 44100Hz，那么计算出来的频率分布是一种情况
而如果 48000Hz，则计算出来的是另一种情况




"""

"""
    两个长度都是 15047
    信号长度是 30079
    也就是不到一半，但近似一半
    为什么不到一半？因为用的是 db8，窗口大小是 8？

    网上说法，ca 表示两个采样点之间的均值 averaging / approximation
    也就是低频分量，这与原始信号近似
    cd 表示之间的差值 differencing / detail coefficient
    网上说法是，这是高频细节？
"""

"""

cwt 共有 5 个参数：
1. data : array_like， 信号数组
2. scales : 要使用的小波尺度（s）。
    可以用 *f = scale2frequency(wavelet, scale)/sampling_period 来确定物理频率大小。f的单位是赫兹，采样周期的单位为秒。
3. wavelet : Wavelet 对象或名字
4. sampling_period : float
    频率输出的采样周期。coefs的计算值与sampling_period的选择无关。scales不按抽样周期进行缩放。
5. axis: int, optional
    计算CWT的轴。如果不给出，则使用最后一个轴。


关于小波变换的一些理解：

小波变换有个尺度数组，这个是干嘛的？
小波基是通过将一个基本的小波函数，分别平移 k，2k，3k 个距离
然后再分别缩放 2 倍，4 倍 等
由于，缩放 a 除了会导致中心频率改变，也会导致小波长度改变！！！
a 的值越大，相当于傅里叶变换中的 Ω 值越小


小波变换的基函数是由父函数和母函数组成，一组小波基，就是对父母函数的缩放平移的集合。
父母函数都是标准正交的，不仅正交还归一化。

本来只有母小波没有父小波的，但是后来引入父小波是为了实现多解析度分析

https://blog.csdn.net/qq_41990294/article/details/114238515


我们的重点不是执行小波变换，而是获取小波基来实现滤波。


cwt_mat 中是个二维数组，每个元素是一个复数。

小波变换的缺点：分辨率比较低，而且计算量大，需要的内存也大。
如果是 2 Hz 一个分辨率的

"""
