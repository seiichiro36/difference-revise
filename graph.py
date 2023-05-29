from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from graph_phase import phase_constant_linear, linear_transform
from _measurements_transform import Open, Short
from _propagation_constant import Propagation_constant


# 終端、解放時と短絡時の送電端インピーダンスの波形データ
open = Open("OPEN.CSV")
open_gamma_data = np.array(open.open_parameter()[2])
open_gamma_data_real = open.open_parameter()[0].flatten()

short = Short("SHORT.CSV")
short_gamma_data = np.array(short.short_parameter()[2])
short_gamma_data_real = short.short_parameter()[0].flatten()

# ケーブルの長さ
length = 10.8

Frequency = np.arange(60000, 30000001, 59880)

C = 102e-12
Z = 50
L = 50**2 * C

omega = Frequency * 2 * np.pi

# 位相定数(無損失時)
beta = omega * np.sqrt(C * L)

# 伝送法による減衰比(10.8mのケーブルを使用)
# 60kHz~30MHzの減衰比
actual_data = pd.read_csv("TRANSMISSION.CSV", usecols=[0], header=None)
actual_data = np.log(10 ** (abs(actual_data)/20))/length
actual_data = actual_data.values

# 伝搬定数
gamma = (actual_data).flatten() + np.array(beta) * 1j

z = 50 / open_gamma_data
x = 1 / 2 * np.log((1 + z) / (1 - z)) / 10.8

difference_propagation = (actual_data).flatten() - x.real 
difference_phase = phase_constant_linear - beta


if __name__ == "__main__":
    fig = plt.figure(figsize=(7, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel("Frequency [Hz]")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel("Frequency [Hz]")

    # # グラフをプロット
    ax1.plot(Frequency, x.real, color="red")
    ax1.plot(Frequency, (actual_data).flatten(), color="blue")
    ax1.plot(Frequency, (actual_data).flatten() - x.real, color="green")
    ax2.plot(Frequency, linear_transform(x.imag), color="red")
    ax2.plot(Frequency, np.array(beta), color="blue")
    ax2.plot(Frequency, linear_transform(x.imag) - np.array(beta), color="green")

    ax1.ticklabel_format(style='plain', axis='x')
    ax2.ticklabel_format(style='plain', axis='x')

    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="sci",  axis="x", scilimits=(6, 6))
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci",  axis="x", scilimits=(6, 6))

    plt.show()


