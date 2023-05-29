import numpy as np
import pandas as pd


class Open:
    def __init__(self, filename) -> None:
        self.filename = filename

    def open_parameter(self):

        mea_open = pd.read_csv(self.filename, usecols=[0], header=None)
        IMPEDANCE_open = 10 ** (mea_open / 10) / 1000  # 取説より
        IMPEDANCE_open = IMPEDANCE_open.values

        arg_open = pd.read_csv(self.filename, usecols=[1], header=None)
        ARG_open = arg_open.values.tolist()
        ARG_open_real = np.cos(np.radians(ARG_open))
        ARG_open_imag = np.sin(np.radians(ARG_open))

        def impedance_vector(a, b):
            return a * b

        open_real = impedance_vector(IMPEDANCE_open, ARG_open_real)
        open_imag = impedance_vector(IMPEDANCE_open, ARG_open_imag)

        open_complex = []
        for i in range(len(mea_open)):
            open_complex.append(complex(open_real[i], open_imag[i]))
            
        return open_real, open_imag, open_complex
        
        
class Short:
    def __init__(self, filename) -> None:
        self.filename = filename
        
    def short_parameter(self):
         # 実測波形データ
        mea_short = pd.read_csv(self.filename, usecols=[0], header=None)
        IMPEDANCE_short = 10 ** (mea_short / 10) / 1000
        IMPEDANCE_short = IMPEDANCE_short.values

        # 実測位相データ
        arg_short = pd.read_csv(self.filename, usecols=[1], header=None)
        ARG_short = arg_short.values.tolist()
        ARG_short_real = np.cos(np.radians(ARG_short))
        ARG_short_imag = np.sin(np.radians(ARG_short))
        def impedance_vector(a, b):
            return a * b

        short_real = impedance_vector(IMPEDANCE_short, ARG_short_real)
        short_imag = impedance_vector(IMPEDANCE_short, ARG_short_imag)

        short_complex = []
        for j in range(len(mea_short)):
            short_complex.append(complex(short_real[j], short_imag[j]))
        
        return short_real, short_imag, short_complex
    