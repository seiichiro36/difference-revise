from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from _measurements_transform import Open, Short

fig = plt.figure(figsize=(7, 5))

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Frequency [Hz]")

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

# 教科書より、終端解放時と短絡時の送電端インピーダンスをそれぞれ導出
tanh_of_Zino = 1 / np.tanh(gamma * length)
tanh_of_Zins = np.tanh(gamma * length)

gamma_tanh_of_Zino = 1 / 2 * np.log((1 + tanh_of_Zino) / (1 - tanh_of_Zino))
gamma_tanh_of_Zins = 1 / 2 * np.log((1 + tanh_of_Zins) / (1 - tanh_of_Zins))

gamma_tanh_of_Zino_real = gamma_tanh_of_Zino.real
gamma_tanh_of_Zins_real = gamma_tanh_of_Zins.real

tanh_open_gamma_actual_data = open_gamma_data / 50
tanh_short_gamma_actual_data = short_gamma_data / 50

gamma_tanh_open_gamma_actual_data = 1 / 2 * np.log((1 + tanh_open_gamma_actual_data)/(1 - tanh_open_gamma_actual_data))
gamma_tanh_short_gamma_actual_data = 1 / 2 * np.log((1 + tanh_short_gamma_actual_data)/(1 - tanh_short_gamma_actual_data))

gamma_tanh_open_gamma_data_real = gamma_tanh_open_gamma_actual_data.real
gamma_tanh_short_gamma_data_real = gamma_tanh_short_gamma_actual_data.real

peaks_gamma_tanh_of_Zino, _ = find_peaks(gamma_tanh_of_Zino_real, height=1)
peaks_gamma_tanh_of_Zins, _ = find_peaks(gamma_tanh_of_Zins_real, height=1)
peaks_gamma_tanh_Zino_actual, _ = find_peaks(gamma_tanh_open_gamma_data_real, height=1)
peaks_gamma_tanh_Zins_actual, _ = find_peaks(gamma_tanh_short_gamma_data_real, height=1)

# ピークの横軸と縦軸の値を取得
tanh_peak_Frequency_of_Zino = Frequency[peaks_gamma_tanh_of_Zino]
tanh_peak_Zino_real = gamma_tanh_of_Zino_real[peaks_gamma_tanh_of_Zino]
# tanh_peak_Frequency_of_Zins = Frequency[peaks_gamma_tanh_of_Zins]
# tanh_peak_Zins_real = gamma_tanh_of_Zins_real[peaks_gamma_tanh_of_Zins]
# tanh_peak_Frequency_of_Zino_actual = Frequency[peaks_gamma_tanh_Zino_actual]
# tanh_peak_Zino_real_actual = gamma_tanh_open_gamma_data_real[peaks_gamma_tanh_Zino_actual]
# tanh_peak_Frequency_of_Zins_actual = Frequency[peaks_gamma_tanh_Zins_actual]
# tanh_peak_Zins_real_actual = gamma_tanh_short_gamma_data_real[peaks_gamma_tanh_Zins_actual]


# グラフをプロット
ax.plot(Frequency, gamma_tanh_of_Zino_real, color="red")  
ax.plot(Frequency, gamma_tanh_of_Zins_real, color="blue")  
ax.plot(Frequency, gamma_tanh_open_gamma_data_real, color="red", linestyle="dotted")  
ax.plot(Frequency, gamma_tanh_short_gamma_data_real, color="blue", linestyle="dotted")  
ax.plot(tanh_peak_Frequency_of_Zino, tanh_peak_Zino_real, 'x', color="red")
fig.tight_layout()

ax.ticklabel_format(style='plain', axis='x')

ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="x", scilimits=(6, 6))

plt.show()