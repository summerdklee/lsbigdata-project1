from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# python은 점을 직선으로 이어서 표현
## y = 2x 그래프 그리기
x = np.linspace(0, 8, 2)
y = 2 * x
plt.scatter(x, y, s = 30)
plt.plot(x, y, color = 'black')
plt.show()
plt.clf()

## y = x^2 그래프 그리기
x = np.linspace(-8, 8, 100)
y = x ** 2

plt.scatter(x, y, s = 30, color = 'green')

plt.plot(x, y, color = 'black')
plt.xlim(-10, 10) # x축 범위 설정
plt.ylim(0, 40) # y축 범위 설정
plt.gca().set_aspect('equal', adjustable = 'box')
# plt.axis('equal') # x축, y축 비율 맞추기 / xlim, ylim과 함께 사용 불가

plt.show()
plt.clf()

# 신뢰구간 구하기 연습문제
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
mu = x.mean() # 표본평균

z_005 = norm.ppf(0.95, loc = 0, scale = 1)

mu + (z_005 * (6 / np.sqrt(16)))
mu - (z_005 * (6 / np.sqrt(16)))

# 데이터로부터 E[X^2] 구하기
sum(x ** 2) / (len(x) - 1)
np.mean((x - x ** 2) / (2 * x))


np.random.seed(20240729)
x = norm.rvs(loc = 3, scale = 5, size = 100000)
x_bar = x.mean()
s_2 = sum((x - x_bar) ** 2) / (100000 - 1)

np.var(x) # n으로 나눈 값
np.var(x, ddof = 1) # n-1로 나눈 값 : 표본분산 > 이걸로만 계산
