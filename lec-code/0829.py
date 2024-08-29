import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error

k = 2
x = np.linspace(-4, 8, 100)
y = ((x - 2) ** 2) + 1
line_y = (4 * x) - 11

plt.plot(x, y, color='black')
plt.plot(x, line_y, color='red')
plt.xlim(-4, 8)
plt.ylim(0, 15)

# f'(x) = 2x - 4
l_slope = (2 * k) - 4
f_k = ((k - 2) ** 2) + 1
l_intercept = f_k - (l_slope * k)

# y = (slope * x) + intercept 그래프
line_y = (l_slope * x) + l_intercept
plt.plot(x, line_y, color='blue')

# y = x^2 경사하강법
# 초기값 : 10, 델타 : 0.9
x = 10
lstep = np.arange(100, 0, -1) * 0.01

for i in range(100):
    x -= lstep[i] * (2 * x)

print(x)

# ====================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# 등고선 그래프
import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)
x = np.linspace(-10, 10, 400)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

x, y = 9, 2
lstep = 0.1

for i in range(100):
    x, y = np.array([x, y]) - lstep * np.array([(2 * x) - 6, (2 * y) - 8])
    plt.scatter(float(x), float(y), color='red', s=25)

print(x, y)

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# =====================================

# x, y의 값을 정의
x = np.linspace(-100, 100, 800)
y = np.linspace(-100, 100, 800)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = ((1 - (x+y)) ** 2) + ((4 - (x+(2*y))) ** 2) + ((1.5 + (x+3*y)) ** 2) + ((5 - (x+4*y)) ** 2)

x, y = 10, 10
lstep = 0.0001

for i in range(500):
    x, y = np.array([x, y]) - lstep * np.array([(8 * x) + 177, (60 * y) + 133])
    plt.scatter(float(x), float(y), color='red', s=10)

print(x, y)

