# 확률질량함수(pmf)
# 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수

# bernoulli.pmf(k, p)
from scipy.stats import bernoulli

## P(X = 1)
bernoulli.pmf(1, 0.3)

## P(X = 0)
bernoulli.pmf(0, 0.3)


# P(X = k | n, p)
# n: 베르누이 확률변수 몇개를 더한 개수
# p: 1이 나올 확률

# binom.pmf(k, n, p)
from scipy.stats import binom

binom.pmf(0, n = 2, p = 0.3)
binom.pmf(1, n = 2, p = 0.3)
binom.pmf(2, n = 2, p = 0.3) 

a = [binom.pmf(x, n = 30, p = 0.3) for x in range(31)]

import numpy as np
binom.pmf(np.arange(31), n = 30, p = 0.3)

# 경우의 수
import math
math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54, 26)
# fact_54 = np.cumprod(np.arange(1, 55))[-1] / (np.cumprod(np.arange(1, 27)[-1] / np.cumprod(np.arange(1, 29)[-1])))

# ln (로그)
## log(a * b) = log(a) + log(b)
log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

np.log(24)
sum(np.log(np.arange(1, 5)))

math.log(mat.factorial(54))
sum(np.log(np.arange(1, 55)))

math.comb(2, 0) * (0.3 ** 0) * ((1 - 0.3) ** 2)
math.comb(2, 1) * (0.3 ** 1) * ((1 - 0.3) ** 1)
math.comb(2, 2) * (0.3 ** 2) * ((1 - 0.3) ** 0)

# pmf: probability math function (확률질량함수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

binom.pmf(np.arange(31), n = 30, p = 0.3)

# X ~ B(n = 10, p = 0.36)
# P(X = 4) = ?
binom.pmf(4, n = 10, p = 0.36)

# P(X =< 4) = ?
binom.pmf(np.arange(5), n = 10, p = 0.36).sum()

# P(2 < X <= 8) = ?
binom.pmf(np.arange(9), n = 10, p = 0.36)[3:9].sum() # binom.pmf(np.arange(3, 9), n = 10, p = 0.36).sum()

# X ~ B(30, 0.2)
# P(X < 4 or X >= 25) = ?
a = binom.pmf(np.arange(4), n = 30, p = 0.2).sum()
b = binom.pmf(np.arange(25, 31), n = 30, p = 0.2).sum()
a + b

1 - binom.pmf(np.arange(4, 25), n = 30, p = 0.2).sum() # 위 a + b와 같은 결과


# rvs 함수: random variates sample (표본 추출 함수)
# X1 ~ Bernoulli(p = 0.3)
bernoulli.rvs(p = 0.3)
# X2 ~ Bernoulli(p = 0.3)
bernoulli.rvs(p = 0.3)
# X ~ B(n = 2, p = 0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n = 2, p = 0.3, size = 1)

# X ~ B(n = 30, p = 0.26)
# 표본 30개 추출
a = binom.rvs(n = 30, p = 0.26, size = 30) # 표본 추출
b = binom.pmf(np.arange(31), n = 30, p = 0.26) # 확률변수가 갖는 값에 해당하는 확률

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.barplot(data = b)
plt.show()
plt.clf()

df = pd.DataFrame({'num' : np.arange(31), 'prob' : b})
sns.barplot(data = df, x = 'num', y = 'prob')
plt.show()
plt.clf()


# cdf 함수: cumulative dist. function (누적확률분포 함수)
#$F_X(x) = P(X <= x)
binom.cdf(4, n = 30, p = 0.26)

a = binom.cdf(18, n = 30, p = 0.26)
b = binom.cdf(4, n = 30, p = 0.26)
a - b

a = binom.cdf(19, n = 30, p = 0.26)
b = binom.cdf(13, n = 30, p = 0.26)
a - b

# 시뮬레이터
x_1 = binom.rvs(n = 30, p = 0.26, size = 10)
x = np.arange(31)
prob_x = binom.pmf(x, n = 30, p = 0.26)
sns.barplot(prob_x, color = 'pink')
plt.show()

plt.scatter(x_1, np.repeat(0.005, 10), color = 'purple', zorder = 100, s = 50)
plt.show()

plt.axvline(x = 7.8, color = 'green', linestyle = 'dotted')
plt.show()
plt.clf()


# ppf()
# binom.ppf(0~1 사이 확률값, n, p)
## P(X < ?) = 0.5 >>> ?의 값을 구하는 함수
binom.ppf(0.5, n = 30, p = 0.26)
binom.cdf(8, n = 30, p = 0.26)
binom.cdf(7, n = 30, p = 0.26)

## P(X < ?) = 0.7
binom.ppf(0.7, n = 30, p = 0.26)
binom.cdf(8, n = 30, p = 0.26)


# 정규분포(종 모양)
1 / np.sqrt(2 * math.pi)

from scipy.stats import norm

norm.pdf(0, loc = 0, scale = 1)
norm.pdf(5, loc = 3, scale = 4)

# 정규분포 pdf 그리기
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = 0, scale = 1)

plt.plot(k, y, color = 'black')
plt.show()
plt.clf()

## mu(loc): 분포의 중심을 결정하는 모수 > 평균
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = 0, scale = 1)

plt.plot(k, y, color = 'black')
plt.show()
plt.clf()

## sigma(scale): 분포의 퍼짐을 결정하는 모수 > 표준편차
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = 0, scale = 1)
y2 = norm.pdf(k, loc = 0, scale =2)
y3 = norm.pdf(k, loc = 0, scale =0.5)

plt.plot(k, y, color = 'black')
plt.plot(k, y2, color = 'red')
plt.plot(k, y3, color = 'blue')
plt.show()
plt.clf()

norm.cdf(0, loc = 0, scale = 1)
norm.cdf(100, loc = 0, scale = 1)
norm.cdf(-100, loc = 0, scale = 1)

a = norm.cdf(-2, loc = 0, scale = 1)
b = norm.cdf(0.54, loc = 0, scale = 1)
b - a

a = norm.cdf(1, loc = 0, scale = 1)
b = 1 - norm.cdf(-3, loc = 0, scale = 1)
b - a

# 정규분포: Normal distribution
# X ~ N(3, 5^2)
# P(3 < X < 5) =? 15.54%
a = norm.cdf(3, loc = 3, scale = 5)
b = norm.cdf(5, loc = 3, scale = 5)
b - a

# 평균 : 3, 표준편차 : 5
x = norm.rvs(loc = 3, scale = 5, size = 1000)
sum((x > 3) & (x < 5)) / 1000

# 평균 : 0, 표준편차 : 1
x = norm.rvs(loc = 0, scale = 1, size = 1000)
sum(x < 0) / 1000 # np.mean(x < 0)
