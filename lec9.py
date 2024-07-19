# E[x]: 기대값
## 계산 방법: sum(가질 수 있는 값 * 대응하는 확률)

import numpy as np

np.arange(4) * np.array([1, 2, 2, 1]) / 6 # 확률
sum(np.arange(4) * np.array([1, 2, 2, 1]) / 6) # 기대값


# 예제 넘파이 배열 생성 >>>>> 여기서부터 놓침
import matplotlib.pyplot as plt

# 히스토그램 그리기
plt.hist(mean, bins = 10, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel("Value")
plt.ylabel('Frequency')
plt.grid(True)
plt.show(False)
plt.clf()

x = np.random.rand(50000).reshape(-1, 5).mean(axis = 1) # np.random.rand(10000, 5).mean(axis = 1)

plt.hist(x, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel("Value")
plt.ylabel('Frequency')
plt.grid(True)
plt.show(False)
