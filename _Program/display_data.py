import random_tools as rt
import power as p
import numpy as np
import tooth_research as tr
import utility_functions as uf
import multi_thread as mt
import time

# 哪个是启动脚本，哪个模块名称就是 __main__，这与文件名无关
if __name__ == '__main__':

    UL1, UL2, UL3, UL4, UL5, UL6, UL7, UL8 = 0, 1, 2, 3, 4, 5, 6, 7
    UR8, UR7, UR6, UR5, UR4, UR3, UR2, UR1 = 8, 9, 10, 11, 12, 13, 14, 15
    LL1, LL2, LL3, LL4, LL5, LL6, LL7, LL8 = 16, 17, 18, 19, 20, 21, 22, 23
    LR8, LR7, LR6, LR5, LR4, LR3, LR2, LR1 = 24, 25, 26, 27, 28, 29, 30, 31
    # 无刷牙动作
    NON = 32

    # 实验室
    src_path = r"C:\Users\xenon\OneDrive\_真实数据\【左上】\L5.wav"
    write_path = r"F:\python生成的数据\real_time.wav"
    obj_curve_path = r"F:\python生成的数据\curve.wav"

    # Mac 笔记本，错误原因：文件目录复制的时候出现了一个反斜线
    # src_path = r"/Users/wangyang/Library/CloudStorage/OneDrive-个人/_真实数据/【左上】/L5.wav"
    # write_path = r"/Volumes/UGREEN/real_time.wav"
    # obj_curve_path = r"/Volumes/UGREEN/curve.wav"

    # python生成的数据
    # write_path = r"/Volumes/UGREEN/python生成的数据/real_time.wav"
    # obj_curve_path = r"/Volumes/UGREEN/python生成的数据/curve.wav"

    # 先进行文件写入测试
    tr.signal_write([], [], 44100, write_path)
    tr.signal_write([], [], 44100, obj_curve_path)

    # 位置表，不用全刷
    # 时长要大于 transtime，否则函数拼接部分报错。
    position = [NON, UL8, NON, UL8, UL7, UL6, UL5, UL4, UL3, LL3, LL4, LL5, LL6, NON, LR6, LR5, LR4]
    duration = [2, 0.2, 5, 3, 2.2, 4, 3.5, 5.3, 6.1, 5, 4.9, 5.8, 5.9, 4, 3.2, 6.8, 4.5]
    # 各位置的时间序列为： [(0, 2), (2, 2.2), (2.2, 7.2), (7.2, 10.2), (10.2, 12.399999999999999), (12.399999999999999, 16.4),
    #             (16.4, 19.9), (19.9, 25.2), (25.2, 31.299999999999997), (31.299999999999997, 36.3),
    #             (36.3, 41.199999999999996), (41.199999999999996, 46.99999999999999),
    #             (46.99999999999999, 52.89999999999999), (52.89999999999999, 56.89999999999999),
    #             (56.89999999999999, 60.099999999999994), (60.099999999999994, 66.9), (66.9, 71.4)]

    # 牙位功率分布，从 U1 到 U8
    upper_main = [36000, 26000, 19500, 15000, 13000, 10000, 9000, 8000]
    # 转换为平均振幅
    upper_main = np.sqrt(upper_main)
    upper_vice = [3300, 3700, 4200, 4800, 5500, 6300, 7200, 7800]
    upper_vice = np.sqrt(upper_vice)
    # 牙位功率分布，从 L1 到 L8
    lower_main = [48000, 43500, 41000, 38000, 35000, 33000, 31000, 28000]
    lower_main = np.sqrt(lower_main)
    lower_vice = [18000, 19000, 20000, 21000, 22000, 23500, 25500, 27000]
    lower_vice = np.sqrt(lower_vice)

    trans_time = 0.2
    steep_dg = 3


    def gen_data():
        if len(position) != len(duration):
            print("错误！数组长度不同。")
        else:
            start_time = time.time()
            total_time = np.sum(duration)
            print("理论总时长：", total_time, "秒")

            # 时长区间，应当是个左右值对。
            time_position = []

            left_signal = np.array([])
            right_signal = np.array([])
            left_obj_curve = np.array([])
            right_obj_curve = np.array([])
            # 功率只大体调整一下即可，不需要滤波等操作.
            # 要在边缘部分有过度感，也就是说要检测数组的下一个位置。
            # 过度时间
            for i in range(len(duration)):
                # 左右位置标识，True 表示左，False 表示右。
                # 上下位置标识 True 表示上，False 表示下。

                if i == 0:
                    time_position.append((0, duration[i]))
                else:
                    time_position.append((np.sum(duration[:i]), np.sum(duration[:i + 1])))

                left_obj, right_obj = get_obj_power(position[i])

                # 每一段 i 是一个整体，从零计数，总体相加就是整个。
                pre_trans_range = (0, trans_time)
                mid_steady_range = (trans_time, duration[i] - trans_time)
                next_trans_range = (duration[i] - trans_time, duration[i])

                # 以下两种情况，必属于其中的一种，或者两种都占。
                # 但是计算得到的 end 和 start 不一定使用。
                if i > 0:
                    # 前半部分过度，自身是后半部分。
                    # 先找到前一个元素索引信息。
                    pre_left_obj, pre_right_obj = get_obj_power(position[i - 1])
                    pre_left_start = (pre_left_obj + left_obj) / 2
                    pre_left_end = left_obj
                    if pre_left_start > pre_left_end:
                        left_trans_type = uf.TRANS_CONCAVE
                    else:
                        left_trans_type = uf.TRANS_CONVEXITY
                    pre_right_start = (pre_right_obj + right_obj) / 2
                    pre_right_end = right_obj
                    if pre_right_start > pre_right_end:
                        right_trans_type = uf.TRANS_CONCAVE
                    else:
                        right_trans_type = uf.TRANS_CONVEXITY
                    pre_left_curve = uf.get_trans_func((pre_trans_range[0], pre_left_start),
                                                       (pre_trans_range[1], pre_left_end), left_trans_type,
                                                       steep_dg=steep_dg)
                    pre_right_curve = uf.get_trans_func((pre_trans_range[0], pre_right_start),
                                                        (pre_trans_range[1], pre_right_end), right_trans_type,
                                                        steep_dg=steep_dg)
                else:
                    # 如果不存在前过渡区间，则为常函数
                    pre_left_curve = lambda x: left_obj if uf.is_num(x) else np.array([left_obj] * len(x))
                    pre_right_curve = lambda x: right_obj if uf.is_num(x) else np.array([right_obj] * len(x))
                if i < len(duration) - 1:
                    # 后半部分过度，自身是前半部分。
                    next_left_obj, next_right_obj = get_obj_power(position[i + 1])
                    next_left_start = left_obj
                    next_left_end = (next_left_obj + left_obj) / 2
                    if next_left_start > next_left_end:
                        left_trans_type = uf.TRANS_CONVEXITY
                    else:
                        left_trans_type = uf.TRANS_CONCAVE
                    next_right_start = right_obj
                    next_right_end = (next_right_obj + right_obj) / 2
                    if next_right_start > next_right_end:
                        right_trans_type = uf.TRANS_CONVEXITY
                    else:
                        right_trans_type = uf.TRANS_CONCAVE

                    next_left_curve = uf.get_trans_func((next_trans_range[0], next_left_start),
                                                        (next_trans_range[1], next_left_end), left_trans_type,
                                                        steep_dg=steep_dg)
                    next_right_curve = uf.get_trans_func((next_trans_range[0], next_right_start),
                                                         (next_trans_range[1], next_right_end), right_trans_type,
                                                         steep_dg=steep_dg)
                else:
                    next_left_curve = lambda x: left_obj if uf.is_num(x) else np.array([left_obj] * len(x))
                    next_right_curve = lambda x: right_obj if uf.is_num(x) else np.array([right_obj] * len(x))

                # 匿名函数，三元表达式可以写成: lambda 参数 : 返回值 1 if 条件 else 返回值 2
                mid_left_curve = lambda x: left_obj if uf.is_num(x) else np.array([left_obj] * len(x))
                mid_right_curve = lambda x: right_obj if uf.is_num(x) else np.array([right_obj] * len(x))

                # 随机波动处理
                pre_left_curve = uf.wave_curve(pre_left_curve, x_lim=pre_trans_range, wave_order=7)
                pre_right_curve = uf.wave_curve(pre_right_curve, x_lim=pre_trans_range, wave_order=7)
                mid_left_curve = uf.wave_curve(mid_left_curve, x_lim=mid_steady_range, wave_order=9)
                mid_right_curve = uf.wave_curve(mid_right_curve, x_lim=mid_steady_range, wave_order=9)
                # 回回都是这两个报错：RankWarning: Polyfit may be poorly conditioned
                next_left_curve = uf.wave_curve(next_left_curve, x_lim=next_trans_range, wave_order=7)
                next_right_curve = uf.wave_curve(next_right_curve, x_lim=next_trans_range, wave_order=7)

                # 三个曲线拼接
                left_curve, x_lim = uf.multi_func_splicing((pre_left_curve, mid_left_curve, next_left_curve),
                                                           (pre_trans_range, mid_steady_range, next_trans_range))
                right_curve, x_lim = uf.multi_func_splicing((pre_right_curve, mid_right_curve, next_right_curve),
                                                            (pre_trans_range, mid_steady_range, next_trans_range))

                # 采样，按照 44100 的采样率来。
                samp_t = np.arange(x_lim[0] * 44100, x_lim[1] * 44100)

                # 采样多线程版
                # left_args = [samp_t, left_curve, 517, 0, 44100]
                # left_signal = np.append(left_signal, mt.multi_thread_proc(uf.sin, left_args, 0))
                left_signal = np.append(left_signal, uf.sin(samp_t, left_curve, 517, 0, 44100))

                # right_args = [samp_t, right_curve, 517, 0, 44100]
                # right_signal = np.append(right_signal, mt.multi_thread_proc(uf.sin, right_args, 0))
                right_signal = np.append(right_signal, uf.sin(samp_t, right_curve, 517, 0, 44100))

                print("已完成：", time_position[-1][1], "/", total_time)

                # 振幅曲线存下来以供调试查看，这个曲线采样时间超级长 ？？？
                # left_obj_curve = np.append(left_obj_curve, left_curve(samp_t))
                # right_obj_curve = np.append(right_obj_curve, right_curve(samp_t))
                # print("完成 obj 曲线")

                """
                1. 对每个位置，获取前后区间。第一个位置的前区间为 0。最后一个位置的后区间为 0. 共有 3 区间 4 限。
                2. 得出每个区间的函数，注意要区分左右声道。过渡区间使用凹凸函数，平稳区间使用常函数。
                3. 得到函数后，进行随机波动处理，得到新的函数。
                4. 将三个区间的函数进行拼接，得到一个函数。
                5. 采样，并添加到总数据中。
                
                未完成：
                
                6. 加入随机 50Hz 信号。
                7. 多线程。
                8. 通过功率找振幅，因为 sin 最总输入的是振幅曲线。
                9. 过渡区的曲率可调。
                """
            print("采样时长[", time.time() - start_time, "]秒")

            # 全部生成后再加入 50Hz 随机信号
            # 信号组数
            group_num = 5
            am = rt.mixed_rand([0.5] * group_num, -0.5, 0.5, 0.6, float_range=0.1)
            freq = rt.mixed_rand([50] * group_num, -2, 2, 0.6, float_range=2)
            phi = rt.mixed_rand([0] * group_num, -np.pi / 2, np.pi / 2, 0.6, float_range=np.pi / 8)
            for i in range(group_num):
                # 多线程采样
                left_args = [left_signal, am[i], freq[i], phi[i]]
                # left_signal = mt.multi_thread_proc(uf.add_sin, left_args, 0)
                left_signal = uf.add_sin(left_signal, am[i], freq[i], phi[i], 44100)
                # left_signal = uf.add_sin(left_args)
                """
                ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape
                after 1 dimensions. The detected shape was (4,) + inhomogeneous part.
                报错信息出在 data = np.array(data, dtype=float)
                # 输入前和输入后，长度和维度都没问题。
                问题出在这，参数应该为 5 个，实际传入 4 个参数。
                """
                right_args = [right_signal, am[i], freq[i], phi[i], 44100]
                # right_signal = mt.multi_thread_proc(uf.add_sin, right_args, 0)
                # 这里忘了加星号了！！！！！列表传参要加星号！
                right_signal = uf.add_sin(*right_args)

            # 加入 50 Hz 谐波
            group_num = 20
            am = rt.mixed_rand([0.5] * group_num, -0.5, 0.5, 0.6, float_range=0.1)
            freq = rt.mixed_rand([50] * group_num, -1, 1, 0.6, float_range=0.5)
            phi = rt.mixed_rand([0] * group_num, -np.pi / 2, np.pi / 2, 0.6, float_range=np.pi / 8)
            for i in range(group_num):
                # 多线程采样
                left_args = [left_signal, am[i] / np.log(i + 2), freq[i] * (i + 2), phi[i], 44100]
                # left_signal = mt.multi_thread_proc(uf.add_sin, left_args, 0)
                # left_signal = uf.add_sin(left_signal, am[i], freq[i], phi[i], 44100)
                left_signal = uf.add_sin(*left_args)
                """
                ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape
                after 1 dimensions. The detected shape was (4,) + inhomogeneous part.
                报错信息出在 data = np.array(data, dtype=float)
                # 输入前和输入后，长度和维度都没问题。
                问题出在这，参数应该为 5 个，实际传入 4 个参数。
                """
                right_args = [right_signal, am[i] / np.log(i + 2), freq[i] * (i + 2), phi[i], 44100]
                # right_signal = mt.multi_thread_proc(uf.add_sin, right_args, 0)
                # 这里忘了加星号了！！！！！列表传参要加星号！
                right_signal = uf.add_sin(*right_args)

            # 再加入直流分量
            group_num = 10
            am = rt.mixed_rand([0.5] * group_num, -0.5, 1, 0.6, float_range=0.1)
            freq = rt.mixed_rand([3] * group_num, -3, 3, 0.6, float_range=2)
            phi = rt.mixed_rand([0] * group_num, -np.pi / 2, np.pi / 2, 0.6, float_range=np.pi / 8)
            for i in range(group_num):
                # 多线程采样
                left_args = [left_signal, am[i], freq[i], phi[i], 44100]
                # left_signal = mt.multi_thread_proc(uf.add_sin, left_args, 0)
                left_signal = uf.add_sin(*left_args)
                right_args = [right_signal, am[i], freq[i], phi[i], 44100]
                # right_signal = mt.multi_thread_proc(uf.add_sin, right_args, 0)
                right_signal = uf.add_sin(*right_args)

            # 直流附近的过渡分量
            group_num = 20
            am = rt.mixed_rand([0.1] * group_num, -0.1, 1, 0.6, float_range=0.02)
            freq = rt.mixed_rand([10] * group_num, -10, 10, 0.6, float_range=5)
            phi = rt.mixed_rand([0] * group_num, -np.pi / 2, np.pi / 2, 0.6, float_range=np.pi / 8)
            for i in range(group_num):
                # 多线程采样
                left_args = [left_signal, am[i], freq[i], phi[i], 44100]
                # left_signal = mt.multi_thread_proc(uf.add_sin, left_args, 0)
                left_signal = uf.add_sin(*left_args)
                right_args = [right_signal, am[i], freq[i], phi[i], 44100]
                # right_signal = mt.multi_thread_proc(uf.add_sin, right_args, 0)
                right_signal = uf.add_sin(*right_args)

            # 再加入主峰附近噪声
            group_num = 5
            am = rt.mixed_rand([1] * group_num, -0.5, 2, 0.6, float_range=0.2)
            freq = rt.mixed_rand([517] * group_num, -5, 5, 0.6, float_range=2)
            phi = rt.mixed_rand([0] * group_num, -np.pi / 2, np.pi / 2, 0.6, float_range=np.pi / 8)
            for i in range(group_num):
                # 多线程采样
                left_args = [left_signal, am[i], freq[i], phi[i], 44100]
                # left_signal = mt.multi_thread_proc(uf.add_sin, left_args, 0)
                left_signal = uf.add_sin(*left_args)
                right_args = [right_signal, am[i], freq[i], phi[i], 44100]
                # right_signal = mt.multi_thread_proc(uf.add_sin, right_args, 0)
                right_signal = uf.add_sin(*right_args)

            print("各位置的时间序列为：", time_position)
            tr.signal_write(left_signal, right_signal, 44100, write_path)
            # tr.signal_write(left_obj_curve, right_obj_curve, 44100, obj_curve_path)

            print("总处理时长[", time.time() - start_time, "]秒")


    """
    现存问题：
    出来的信号是 500Hz 整，这是由于 441 的窗口大小导致其频率分辨率最高只能达到 220.5
    这就需要先升采样，再降采样。
    """


    # 根据牙位，找到左右声道对应的目标功率
    def get_obj_power(pos):
        if 0 <= pos <= 7:
            # 上左牙区
            # position 作为查找下标
            obj_idx = pos
            lr_flag = True
            ul_flag = True
        elif 8 <= pos <= 15:
            # 上右牙区
            obj_idx = 15 - pos
            lr_flag = False
            ul_flag = True
        elif 16 <= pos <= 23:
            # 下左牙区
            obj_idx = pos - 16
            lr_flag = True
            ul_flag = False
        elif 24 <= pos <= 31:
            # 下右牙区
            obj_idx = 31 - pos
            lr_flag = False
            ul_flag = False
        else:
            # 无刷牙动作
            return 10, 10
        if ul_flag:
            left_obj = upper_main[obj_idx]
            right_obj = upper_vice[obj_idx]
        else:
            left_obj = lower_main[obj_idx]
            right_obj = lower_vice[obj_idx]
        # 如果是右侧，则左右声道值互换
        if not lr_flag:
            left_obj, right_obj = uf.exchange_val(left_obj, right_obj)
        return left_obj, right_obj


    gen_data()
