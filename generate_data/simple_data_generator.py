import numpy as np
np.std(a, axis=0)
noise = np.random.normal(0, 1, a.shape) * np.std(a, axis=0)
b= a+noise