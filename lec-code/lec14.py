from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 균일분포(Uniform)
## X ~ U(a, b) > a, b: 범위 (a~b 까지)
## uniform.OOO(k, loc, scale) > k: 구하고자 하는 값 / loc: 구간 시작점(a) / scale: 구간 길이(b - a)

## X ~ U(2, 6)
uniform.rvs(loc = 2, scale = 4, size = 1)

  = np.linspace(0, 8, 100)
x = uniform.pdf(k, loc = 2, scale = 4) # 균일분포 그래프의 높이

plt.plot(k, x, color = 'black')
plt.show()
plt.clf()

### P(X < 3.25) = ?
uniform.cdf(3.25, loc = 2, scale = 4) # 왼쪽에서부터 3.25까지 넓이

### P(5 < X < 8.39) = ?
uniform.cdf(8.39, loc = 2, scale = 4) - uniform.cdf(5, loc = 2, scale = 4)

### 상위 7% 값?
uniform.ppf(0.93, loc = 2, scale = 4)
#### y * 1/4 = 0.07
#### y = 0.28
#### 6 - 0.28 = 5.72

### 시뮬레이터
#### 표본을 뽑아서 표본평균 계산
x = uniform.rvs(loc = 2, scale = 4, size = 20*1000, random_state = 42) # random_state: 랜덤값을 고정
x = x.reshape(1000, 20) # x.reshape((-1, 20))와 같은 결과
blue_x = x.mean(axis = 1) # 표본평균

#### 표본평균의 그래프: 정규분포
sns.histplot(blue_x, stat = 'density')
plt.show()

##### X bar ~ N(mu, sig ** 2 / n)
##### X bar ~ N(4, 1.333333/20)
uniform.var(loc = 2, scale = 4) # 모집단 분산(검은 벽돌)
uniform.expect(loc = 2, scale = 4) # 모집단 기대값(검은 벽돌)

xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_vsalues, color = 'red')
plt.show()
plt.clf()


#### 신뢰구간
4 - norm.ppf(0.025, loc = 4, scale = np.sqrt(1.333333/20)) # 95%
4 - norm.ppf(0.975, loc = 4, scale = np.sqrt(1.333333/20))

a = 4 - norm.ppf(0.005, loc = 4, scale = np.sqrt(1.333333/20)) # 99%
b = 4 - norm.ppf(0.995, loc = 4, scale = np.sqrt(1.333333/20))

# 정규분포 표현
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(1.333333/20))
plt.plot(x_values, pdf_values, color = 'red')

# 기대값을 라인으로 표현
plt.axvline(x = 4, color = 'green', linewidth = 2)
plt.show()

# 표본평균(파란벽돌) 점찍기
x = uniform.rvs(loc = 2, scale = 4, size = 20)
plt.show()
blue_x = x.mean() # 표본평균
## norm.ppf(0.975, loc = 0, scale = 1) == 1.96

plt.scatter(blue_x, 0.002, color = 'blue', zorder = 100, s = 10)
plt.show()

# 신뢰구간 표현
a = blue_x + 0.665
b = blue_x - 0.665

plt.axvline(x = a, color = 'blue', linestyle = 'dotted')
plt.axvline(x = b, color = 'blue', linestyle = 'dotted')
plt.show()
plt.clf()
