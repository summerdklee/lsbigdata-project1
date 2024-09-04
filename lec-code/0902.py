import numpy as np
import pandas as pd

# 1
X = np.arange(2, 13, 1)
p = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

E = sum(X * p)
V = sum((X ** 2) * p) - (E ** 2)

# 2
X2 = (2 * X) + 3

E2 = sum(X2 * p)
V2 = sum((X2 ** 2) * p) - (E2 ** 2)
s = np.sqrt(V2)

