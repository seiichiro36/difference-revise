import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from _measurements_transform import Open, Short
from _propagation_constant import Propagation_constant
from theoretical import beta

# Frequency1 = np.linspace(60000, 30000000, 501)
# np.linespaceからnp.arangeに変更
Frequency = np.arange(60000, 30000001, 59880)

open = Open("./data/OPEN.CSV")
open_complex_data = open.open_parameter()

short = Short("./data/SHORT.CSV")
short_complex_data = short.short_parameter()

def linear_transform(phase_constant):
    phase = []
    for i in range(0, len(phase_constant)):
        phase.append(phase_constant[i])

    initial_point = phase[0]
    start_point = []
    end_point = []
    final_point = phase[-1]

    total_break_point = []

    for j in range(0, len(phase_constant)):
        # out of range対策
        if j+1 == len(phase_constant):
            break
        # 位相の一番下のところ
        if phase[j] > phase[j+1]:
            start_point.append(phase[j+1])
            end_point.append(phase[j])

    between_phase = phase[phase.index(initial_point): phase.index(start_point[0])]

    break_point = phase.index(start_point[0])
    total_break_point.append(break_point)

    for m in range(len(end_point) - 1):
        amount_of_change = start_point[m]
        between_start_end = phase[phase.index(start_point[m]): phase.index(end_point[m+1])+1]
        between_start_end = between_start_end + between_phase[-1] + amount_of_change * -1
        between_phase.extend(between_start_end)

        break_point = phase.index(start_point[m+1])
        total_break_point.append(break_point)

    amount_of_change = start_point[-1]
    between_start_end_final = phase[phase.index(start_point[-1]): phase.index(final_point)+1]
    between_start_end_final = between_start_end_final + between_phase[-1] + amount_of_change*-1
    between_phase.extend(between_start_end_final)

    phase = between_phase

    return phase

data = Propagation_constant(open_complex_data[2], short_complex_data[2], 10.8)

propagation_constant = data.propagation_constant()

phase_constant = propagation_constant[1]

phase_constant_linear = linear_transform(phase_constant)

if __name__== "__main__":
    fig = plt.figure(figsize=(12, 8))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel("(b)   Frequency [Hz]")
    ax1.set_ylabel("[rad/m]")
    
    ax1.plot(Frequency, phase_constant_linear, color="red", label="phase constant from actual data")

    fig.tight_layout()

    ax1.ticklabel_format(style='plain', axis='x')

    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="sci",  axis="x", scilimits=(6, 6))
    
    ax1.set_xlim([60000, 30000000])
    
    x_formatter = ScalarFormatter(useOffset=False)
    
    ax1.legend(loc='lower left')  # 凡例

    plt.show()
