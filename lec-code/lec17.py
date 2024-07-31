from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# [z = (x - mu) / sig]
# X ~ N(3, 7^2)
## 하위 25%
x = norm.ppf(0.25, loc = 3, scale = 7)
norm.cdf(5, loc = 3, scale = 7)

# Z ~ N(0, 1^2)
## 하위 25%
z = norm.ppf(0.25, loc = 0, scale = 1)
norm.cdf(2/7, loc = 0, scale = 1)

# 95% 신뢰구간
norm.ppf(0.975, loc = 0, scale = 1)
norm.ppf(0.025, loc = 0, scale = 1)

a = x_bar + (1.96 * (sig / root_n))
a = x_bar - (1.96 * (sig / root_n))

# 표본정규분포 / 표준(Z) 1000개 뽑고 히스토그램 그리기 > pdf 겹쳐서 그리기
z = norm.rvs(loc = 0, scale = 1, size = 10000)
x = z * np.sqrt(2) + 3
sns.histplot(z, stat = 'density', color = 'pink')
sns.histplot(x, stat = 'density', color = 'grey')

zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc = 0, scale = 1)
pdf_values2 = norm.pdf(z_values, loc = 3, scale = np.sqrt(2))
plt.plot(z_values, pdf_values, color = 'blue')
plt.plot(z_values, pdf_values2, color = 'red')

plt.show()
plt.clf()

# X ~ N(5, 3^2)이고, z = (x - 5) / 3일때, z가 표준정규분포를 따르나요?
## 표본 추출
x = norm.rvs(loc = 5, scale = 3, size = 10000)

## 표준화
z = (x - 5) / 3

## z의 히스토그램, pdf 겹치기
sns.histplot(z, stat = 'density', color = 'blue')
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc = 0, scale = 1)
plt.plot(z_values, pdf_values, color = 'black')

plt.show()
plt.clf()

# 표본표준편차를 나눠도 표준정규분포가 될까?
## (1) X 표본 10개 추출해서 표본분산값 계산
x = norm.rvs(loc = 5, scale = 3, size = 20)
s_2 = np.var(x, ddof = 1)

## (2) X에서 표본 1000개 추출
x = norm.rvs(loc = 5, scale = 3, size = 1000)

## (3) (1)에서 계산한 표본분산값으로 sig^2를 대체한 표준화 진행
### 표준화: z = (X - mu) / sig
z = (x - 5) / np.sqrt(s_2)

## (4) z의 히스토그램, 표준정규분포 pdf 겹쳐 그리기
sns.histplot(z, stat = 'density', color = 'yellow')
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc = 0, scale = 1)
plt.plot(z_values, pdf_values, color = 'red')

plt.show()
plt.clf()

# 풀이
## (1) X 표본 10개 추출해서 표본분산값 계산
x = norm.rvs(loc = 5, scale = 3, size = 20)
s = np.std(x, ddof = 1)

## (2) X에서 표본 1000개 추출
x = norm.rvs(loc = 5, scale = 3, size = 1000)

## (3) (1)에서 계산한 표본분산값으로 sig^2를 대체한 표준화 진행
### 표준화: z = (X - mu) / sig
z = (x - 5) / s

## (4) z의 히스토그램, 표준정규분포 pdf 겹쳐 그리기
sns.histplot(z, stat = 'density', color = 'yellow')
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc = 0, scale = 1)
plt.plot(z_values, pdf_values, color = 'red')

plt.show()
plt.clf()


# [t 분포] X ~ t(df)
## 모수가 1개인 분포
## t 분포 특징: 종 모양, 대칭 분포, 중심이 0
## 모수 df: '자유도'라고 부르고, 퍼짐을 나타냄
## 따라서, df가 작으면 분산이 커짐
## 자유도가 커질수록 정규분포와 비슷해짐 (자유도가 무한대로 가면 결국 표준정규분포가 된다)
## t.OOO(k, df) - OOO은 pdf/cdf/ppf/rvs

from scipy.stats import t

# 예제: 자유도가 4인 t분포의 pdf를 그려보세요.
x = np.linspace(-4, 4, 100)
t_values = t.pdf(x, df = 30)
pdf_values = norm.pdf(x, loc = 0, scale = 1)

plt.plot(x, t_values, color = 'violet')
plt.plot(x, pdf_values, color = 'black')
plt.show()
plt.clf()

# X ~ ?(mu, sig^2/n)
# X_bar ~ N(mu, sig^2/n)
# X_bar ~= t(x_bar, s^2/n) 자유도가 n-1인 t 분포

# 예제: 모평균에 대한 95% 신뢰구간을 구해보자
x = norm.rvs(loc = 15, scale = 3, size = 16, random_state = 42)
x_bar = x.mean()
n = len(x)

## 모분산을 모를때
x_bar + t.ppf(0.975, df = n - 1) * np.std(x, ddof = 1) / np.sqrt(n)
x_bar - t.ppf(0.975, df = n - 1) * np.std(x, ddof = 1) / np.sqrt(n)

## 모분산(3^2)을 알때
x_bar + norm.ppf(0.975, loc = 0, scale = 1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc = 0, scale = 1) * 3 / np.sqrt(n)
