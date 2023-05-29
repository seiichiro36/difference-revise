import numpy as np 

Frequency = np.arange(60000, 30000001, 59880)

C = 102e-12
Z = 50
L = 50**2 * C

#無損失の場合

omega = Frequency * 2 * np.pi

beta = omega * np.sqrt(C * L)