import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

from _propagation_constant import Propagation_constant
from _measurements_transform import Open, Short
from theoretical import beta
from graph_phase import linear_transform

open = Open("./data/OPEN.CSV")
open_complex_data = open.open_parameter()

short = Short("./data/SHORT.CSV")
short_complex_data = short.short_parameter()
length = 10.8

# 伝送法による減衰比(10.8mのケーブルを使用)
actual_data = pd.read_csv("./data/TRANSMISSION.CSV", usecols=[0], header=None)

actual_data = np.log(10 ** (abs(actual_data)/20))/length

actual_data = actual_data.values

Frequency = np.arange(60000, 30000001, 59880)

propagation_constant_data = Propagation_constant(open_complex_data[2], short_complex_data[2], length)

damping_constant_actual_value = propagation_constant_data.propagation_constant()[0]
phase_constant_actual_value = propagation_constant_data.propagation_constant()[1]

difference_damping = damping_constant_actual_value - (actual_data).flatten()
difference_phase = linear_transform(phase_constant_actual_value) - beta

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 8))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("compare damping constant")
    ax1.set_xlabel("(a)  Frequency [Hz]")
    ax1.set_ylabel("[Np/m]")
    ax1.set_title("aaa")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("compare damping constant")
    ax2.set_xlabel("(b)  Frequency [Hz]")
    ax2.set_ylabel("[rad/m]")
    
    ax1.plot(Frequency, damping_constant_actual_value, color="red", label="actual data")
    ax1.plot(Frequency, actual_data, color="blue", label="calculation value")
    ax2.plot(Frequency, linear_transform(phase_constant_actual_value), color="red", label="actual data")
    ax2.plot(Frequency, beta, color="blue", label="calculation value")
    
    ax1.tick_params(labelsize=20)
    ax1.legend()
    
    ax2.tick_params(labelsize=20)
    ax2.legend()

    fig.tight_layout()

    ax1.ticklabel_format(style='plain', axis='x')
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="sci",  axis="x", scilimits=(6, 6))
    ax1.set_xlim([60000, 30000000])

    ax2.ticklabel_format(style='plain', axis='x')
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci",  axis="x", scilimits=(6, 6))
    ax2.set_xlim([60000, 30000000])
    
    print(difference_damping)
    print(difference_phase)
    
    
    # plt.show()