# 測定値から伝搬定数を求める。
# Pdpagation_constantクラスの第1戻り値が減衰定数、第2戻り値が位相定数である。

import numpy as np

class Propagation_constant:
    def __init__(self, open_complex_data, short_complex_data, length) -> None:
        self.open_complex_data = open_complex_data
        self.short_complex_data = short_complex_data
        self.length = length

    def propagation_constant(self):
        propagation_constant = []

        def calc(x, y):
            x =  np.sqrt(x/y)
            return 1/(2*self.length)*np.log((1 + x)/(1 - x))

        for k in range(len(self.open_complex_data)):
            propagation_constant.append(
                calc(self.short_complex_data[k], self.open_complex_data[k]))

        propagation_constant_real = np.array(
            propagation_constant).real
        propagation_constant_imag = np.array(
            propagation_constant).imag
              
        return propagation_constant_real, propagation_constant_imag