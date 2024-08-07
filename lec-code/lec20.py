import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import t
from sklearn.linear_model import LinearRegression

x = np.linspace(0, 100, 400)
y = (2 * x) + 3

# np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20)
eps_i = norm.rvs(loc = 0, scale = 100, size = 20)
obs_y = (2 * obs_x) + 3 + eps_i

plt.plot(x, y, color = 'black')
plt.scatter(obs_x, obs_y, color = 'blue')

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
obs_x = obs_x.reshape(-1, 1)
model.fit(obs_x, obs_y)

# 회귀 직선의 기울기와 절편
slope = model.coef_[0] # a_hat
intercept = model.intercept_ # b_hat

y_2 = (slope * x) + intercept
plt.plot(x, y_2, color = 'red')
plt.xlim([0, 100])
plt.ylim([0, 300])
plt.show()
plt.clf()

# ! pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())

1 - norm.cdf(18, loc = 10, scale = 1.96) # P-value; 유의확률

(18-10) / 1.96
1- norm.cdf(4.08, loc = 0, scale = 1)

# ==============================================================================

# 모표준편차를 모르는 경우
# 1. 귀무가설 vs 대립가설 설정
## 귀무가설: 2022년 슬통 자동차 실형 모델의 평균 복합 에너지 소비효율이 16.0 이상이다.
## 대립가설: 2022년 슬통 자동차 실형 모델의 평균 복합 에너지 소비효율이 16.0 이상이 아니다.

# 2. t = (x_bar - mu0) / s/root_n 계산
new_car = np.array([15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,
15.382, 16.709, 16.804])

x_bar = new_car.mean()
mu0 = 16
sig = np.std(new_car, ddof = 1)
n = len(new_car)

t_value = (x_bar - mu0) / std / np.sqrt(n)

# 3. 주어진 t로 유의확률 계산(t분포 n-1 활용)
p_value = t.cdf(t_value, df = 14)

# 4. 유의수준과 비교해서 귀무가설 기각할지 결정

