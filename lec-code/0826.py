import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# [벡터 * 벡터 (내적)]
a = np.arange(1, 4)
b = np.array([3, 6, 9]).reshape(3, 1)

a.dot(b)

# [행렬 * 벡터]
a = np.array([1, 2, 3, 4]).reshape(2, 2, order="F")
b = np.array([5, 6]).reshape(2, 1)

a.dot(b)
a @ b

# [행렬 * 행렬]
a = np.array([1, 2, 3, 4]).reshape(2, 2, order="F")
b = np.array([5, 6, 7, 8]).reshape(2, 2, order="F")

a @ b

## [연습문제 1]
a = np.array([1, 2, 1, 0, 2, 3]).reshape(2, 3)
b = np.array([1, 0, -1, 1, 2, 3]).reshape(3, 2)

a @ b

## [연습문제 2]
a = np.array([3, 5, 7, 2, 4, 9, 3, 1, 0]).reshape(3, 3)
b = np.eye(3) # 단위행렬

a @ b

# [행렬 뒤집기 (전치, Transpose)]
a.transpose()

b = a[:, 0:2]
b.transpose()

# [회귀분석 데이터 행렬]
x = np.array([13, 15, 12, 14, 10, 11, 5, 6]).reshape(4, 2)
vec1 = np.repeat(1, 4).reshape(4, 1)
matX = np.hstack((vec1, x))

beta_vec = np.array([2, 0, 1]).reshape(3, 1)

matX @ beta_vec

y = np.array([20, 19, 20, 12]).reshape(4, 1)

(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)

# [역행렬 (Inverse Matrix)]
A = np.array([1, 5, 3, 4]).reshape(2, 2)
A_inv = (-1/11) * np.array([4, -5, -3, 1]).reshape(2, 2)

A @ A_inv

## [연습문제. (3, 3) 역행렬]
a = np.array([-4, -6, 2, 5, -1, 3, -2, 4, -3]).reshape(3, 3)
a_inv = np.linalg.inv(a) # 역행렬 구하는 함수

np.round(a @ a_inv, 3)
np.linalg.det(a)

# 주의! 역행렬은 항상 존재하는 것이 아니다.
# 행렬의 세로 벡터들이 선형 독립일때만 역행렬을 구할 수 있다.
# '선형 독립'이 아닌 경우: 선형 종속
# 역행렬이 존재하지 않는다. = 특이행렬 (singular matrix) = 행렬식이 0 이다.
b = np.array([1, 2, 3, 2, 4, 5, 3, 6, 7]).reshape(3, 3)
b_inv = np.linalg.inv(b) # singular matrix 에러
np.linalg.det(b) # 행렬식이 0

# [벡터 형태로 베타 구하기]
XtX_inv = np.linalg.inv((matX.transpose() @ matX))
Xty = matX.transpose() @ y
beta_hat = XtX_inv @ Xty

# [model.fit으로 베타 구하기]
model = LinearRegression()
model.fit(matX[:, 1:], y)

model.coef_
model.intercept_

# [minimize로 베타 구하기]
def line_perform(beta):
    beta = np.array(beta).reshape(3, 1)
    a = y - matX @ beta # matX @ beta : y_hat
    return a.transpose() @ a

# line_perform([8.55, 5.96, -4.38])

initial_guess = [0, 0, 0]

result = minimize(line_perform, initial_guess)
result.fun # 최소값
result.x   # 최소값을 갖는 x 값

# [minimize로 Lasso 베타 구하기]
def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3, 1)
    a = y - matX @ beta
    return (a.transpose() @ a) + (30 * np.abs(beta[1:]).sum())

# line_perform_lasso([8.55, 5.96, -4.38])
# line_perform_lasso([8.14, 0.96, 0])

initial_guess = [0, 0, 0]

result = minimize(line_perform_lasso, initial_guess)
result.fun # 최소값
result.x   # 최소값을 갖는 x 값

## [8.55, 5.96, -4.38] # 람다가 0일때
## 예측식 : y_hat = 8.55 + 5.96*X1 + (-4.38)*X2

## [8.14, 0.96, 0] : 람다가 3일때
## 예측식 : y_hat = 8.14 + 0.96*X1 + 0*X2

## [17.74, 0, 0] : 람다가 500일때
## 예측식 : y_hat = 17.74 + 0*X1 + 0*X2

## 람다 값에 따라 변수가 선택된다.
## X 변수가 추가되면, train_X에서는 성능이 항상 좋아진다.
## X 변수가 추가되면, valid_X에서는 성능이 좋아졌다가 나빠진다. (오버피팅)
## 어느순간 X 변수 추가하는 것을 멈춰야 한다.
## 람다 0부터 시작 : 내가 가진 모든 변수를 넣겠다.
## 람다를 증가 : 변수가 하나씩 빠지는 효과
## valid_X에서 가장 성능이 좋은 람다로 선택 : 변수가 선택됨을 의미한다.

## 20차 모델 성능 확인하기
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x})

train_df = df.loc[:19]

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
### 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

model = Lasso(alpha=0.03) # lambda가 alpha로 표현
model.fit(train_x, train_y)

valid_df = df.loc[20:]

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

### 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y = valid_df["y"]

### 모델 성능
y_hat_train = model.predict(train_x)
y_hat_val = model.predict(valid_x)

sum((train_df["y"] - y_hat_train)**2)
sum((valid_df["y"] - y_hat_val)**2)

tr_result = np.repeat(0.0, 100)
val_result = np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model = Lasso(alpha=i*0.1)
    model.fit(train_x, train_y)

    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train = sum((train_df["y"] - y_hat_train)**2)
    perf_val = sum((valid_df["y"] - y_hat_val)**2)

    tr_result[i] = perf_train
    val_result[i] = perf_val

df = pd.DataFrame({
    'l' : np.arange(0, 10, 0.1),
    'tr' : tr_result,
    'val' : val_result
})

sns.scatterplot(data = df, x = 'l', y = 'tr', color = 'blue')
sns.scatterplot(data = df, x = 'l', y = 'val', color = 'red')
plt.xlim(0, 1)

np.min(val_result) # alpha를 0.1로 선택!

# [8/27 그래프 그리기]
model = Lasso(alpha=0.03)
model.fit(train_x, train_y)

x_pred = np.arange(-4, 4, 0.01)
x_pred_poly = np.column_stack([x_pred ** i for i in range(1, 21)])  # 상수항 추가하지 않음
y_pred = model.predict(x_pred_poly)

plt.scatter(valid_df["x"], valid_df["y"], color="blue", label="Validation Data")
plt.plot(x_pred, y_pred, color="red", label="Lasso Prediction")
plt.xlabel("x")
plt.ylabel("y")
