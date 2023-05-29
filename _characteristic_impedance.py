from _measurements_transform import Open, Short
import numpy as np


open = Open("OPEN.CSV")
open_complex_data = open.open_parameter()

short = Short("SHORT.CSV")
short_complex_data = short.short_parameter()


class Characteristic_impedance:
    def __init__(self, open_complex_data, short_complex_data) -> None:
        self.open_complex_data = open_complex_data
        self.short_complex_data = short_complex_data

    def characteristic_impedance(self):
        characteristic_impedance = []

        def calc(a, b):
            return np.sqrt(a*b)

        for k in range(len(self.open_complex_data)):
            characteristic_impedance.append(
                calc(self.open_complex_data[k], self.short_complex_data[k]))

        characteristic_impedance_real = np.array(characteristic_impedance).real
        characteristic_impedance_imag = np.array(characteristic_impedance).imag

        return characteristic_impedance_real, characteristic_impedance_imag

