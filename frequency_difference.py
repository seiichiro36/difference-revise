from identify_peak_value import Zino_peaks, Zins_peaks, peaks_Zino_actual, peaks_Zins_actual
import numpy as np
import matplotlib.pyplot as plt


Frequency = np.arange(60000, 30000001, 59880)

Zino_Frequency_difference = Frequency[peaks_Zino_actual] - Frequency[Zino_peaks]
Zins_Frequency_difference = Frequency[peaks_Zins_actual] - Frequency[Zins_peaks]


print(Zino_Frequency_difference[1] - Zino_Frequency_difference[0])
print(Zino_Frequency_difference[2] - Zino_Frequency_difference[1])