"""
本程序文件用于实现数据分割
"""
import os
import random_tools as rt
import utility_functions as uf


def cut(dir_path, name, number):
    # 文件夹，搜索整体文件夹内文件，切割并改名
    # dir_path = r"C:\Users\xenon\Desktop\230627刷牙实验数据\1朱昊"
    file_list = os.listdir(dir_path)
    for f_path in file_list:
        # 遍历文件夹里的文件
        if f_path.split('.')[1] != "wav" and f_path.split('.')[1] != "WAV":
            # 校验是否为 wav 文件
            continue
        file_name = f_path.split('.')[0]
        path = dir_path + '/' + f_path

        # 随机长度切分
        for j in range(number):
            fs, signal = rt.get_rand_len_signal(path, length=44100 * 1, float_range=8820)
            # HST-DL-1-001
            # uf.signal_write(signal[:, 0], signal[:, 1], fs,
            #                 dir_path + '/' + name + '-' + file_name[0:2] + '-' + file_name[2] + '-' + str(j).zfill(
            #                     3) + ".wav")
            uf.signal_write(signal[:, 1], signal[:, 0], fs,
                            dir_path + '/' + name + '-' + file_name[0:2] + '-' + file_name[2] + '-' + str(j).zfill(
                                3) + ".wav")

    return
