from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from _measurements_transform import Open, Short
from compare_propagation_constant import difference_damping
from compare_propagation_constant import difference_phase

fig = plt.figure(figsize=(7, 5))

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Transmission end impedance[Ω]")

# 終端、解放時と短絡時の送電端インピーダンスの波形データ
open = Open("./data/OPEN.CSV")
open_gamma_data = np.array(open.open_parameter()[2])
open_gamma_data_real = open.open_parameter()[0].flatten()

short = Short("./data/SHORT.CSV")
short_gamma_data = np.array(short.short_parameter()[2])
short_gamma_data_real = short.short_parameter()[0].flatten()


# ケーブルの長さ
length = 10.8

Frequency = np.arange(60000, 30000001, 59880)
# 位相定数(無損失時)

C = 102e-12
Z = 50
L = 50**2 * C

omega = Frequency * 2 * np.pi

beta = omega * np.sqrt(C * L)

# 伝送法による減衰比(10.8mのケーブルを使用)
actual_data = pd.read_csv("./data/TRANSMISSION.CSV", usecols=[0], header=None)
actual_data = np.log(10 ** (abs(actual_data)/20))/length
actual_data = actual_data.values


# # 伝搬定数
gamma_original = actual_data.flatten() + (np.array(beta)) * 1j
gamma = actual_data.flatten() + difference_damping + (np.array(beta) + difference_phase) * 1j

average_gamma = (gamma_original + gamma) / 2
# 教科書より、終端解放時と短絡時の送電端インピーダンスをそれぞれ出力
Zino_original = 50 / np.tanh(gamma_original * length)
Zino = 50 / np.tanh(gamma * length)
Zino_real_original = Zino_original.real
Zino_real = Zino.real
average_Zino = 50 / np.tanh(average_gamma * length)
average_Zino_real = average_Zino.real

peaks_Zino, _ = find_peaks(Zino_real, height=50)
peaks_Zino_original, _ = find_peaks(Zino_real_original, height=50)
peaks_Zino_actual, _ = find_peaks(open_gamma_data_real, height=50)
peaks_Zino_average, _ = find_peaks(average_Zino_real, height=50)

peak_Frequency_of_Zino = Frequency[peaks_Zino]
peak_Zino_real = Zino_real[peaks_Zino]
peak_Frequency_of_Zino_original = Frequency[peaks_Zino_original]
peak_Zino_real_original = Zino_real_original[peaks_Zino_original]
peak_Frequency_of_Zino_actual = Frequency[peaks_Zino_actual]
peak_Zino_real_actual = open_gamma_data_real[peaks_Zino_actual]
peak_Frequency_of_Zino_average = Frequency[peaks_Zino_average]
peak_Zino_real_average = average_Zino_real[peaks_Zino_average]


# グラフをプロット
ax.plot(Frequency, Zino_real, color="red", label="after revise")
ax.plot(Frequency, Zino_original, color="blue", label="original")
ax.plot(Frequency, average_Zino_real, color="green", label="average both before and after")
ax.plot(Frequency, open_gamma_data_real, color="red", label="actual wave", linestyle="dotted")

ax.plot(peak_Frequency_of_Zino, peak_Zino_real, 'x', color="red")
ax.plot(peak_Frequency_of_Zino_original, peak_Zino_real_original, 'x', color="blue")
ax.plot(peak_Frequency_of_Zino_average, peak_Zino_real_average, 'x', color="green")
ax.plot(peak_Frequency_of_Zino_actual, peak_Zino_real_actual, 'o', color="red")

ax.tick_params(labelsize=20)
ax.legend()

fig.tight_layout()

ax.ticklabel_format(style='plain', axis='x')

ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="x", scilimits=(6, 6))

plt.show()
