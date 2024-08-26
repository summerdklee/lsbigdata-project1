import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# y = ax^2 +bx + c 그래프 그리기 (2차 곡선의 방정식)
a = 2
b = 3
c = 5

x = np.linspace(-8, 8, 100)
y = (a * (x ** 2)) + (b * x) + c

plt.plot(x, y, color = 'black')

# y = ax^3 +bx^2 + cx + b 그래프 그리기 (3차 곡선의 방정식)
a = 2
b = 3
c = 5
d = -1

x = np.linspace(-8, 8, 100)
y = (a * (x ** 3)) + (b * (x ** 2)) + (c * x) + d

plt.plot(x, y, color = 'blue')

# y = ax^4 +bx^3 + cx^2 + bx + e 그래프 그리기 (4차 곡선의 방정식)
a = 1
b = 0
c = -10
d = 0
e = 10

x = np.linspace(-4, 4, 1000)
y = (a * (x ** 4)) + (b * (x ** 3)) + (c * (x ** 2)) + d * x + e

plt.plot(x, y, color = 'red')

# 데이터 만들기
## 검정 곡선
k = np.linspace(-4, 4, 200) # 이상적인 데이터 (정답)
sin_y = np.sin(k)

## 파란 점
x = uniform.rvs(loc = -4, scale = 8, size = 20) # 노이즈 (엡실론, έ)
y = np.sin(x) + norm.rvs(loc = 0, scale = 0.3, size = 20)

plt.plot(k, sin_y, color = 'black') # 원래 이 부분은 모르는 부분, 파란 점만 보고 검정 곡선을 유추해야 함
plt.scatter(x, y, color = 'blue')

# train, test 데이터 만들기
np.random.seed(42)

x = uniform.rvs(loc = -4, scale = 8, size = 30)
y = np.sin(x) + norm.rvs(loc = 0, scale = 0.3, size = 30)

df = pd.DataFrame({"x": x, "y": y})

train_df = df.loc[:19]
test_df = df.loc[20:]

plt.scatter(train_df["x"], train_df["y"], color = "blue")

model = LinearRegression()

## 1차 직선 회귀
train_x = train_df[["x"]] # 시리즈 타입
train_y = train_df["y"]

model.fit(train_x, train_y)

model.coef_
model.intercept_

reg_line = model.predict(train_x)

plt.plot(train_x, reg_line, color = 'red')

## 2차 곡선 회귀
train_df['x2'] = train_df["x"] ** 2

train_x2 = train_df[["x", "x2"]]
train_y = train_df["y"]

model.fit(train_x2, train_y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({"x": k, "x2": k**2})

reg_line2 = model.predict(df_k)

plt.plot(k, reg_line2, color = 'red')
plt.scatter(train_df["x"], train_df["y"], color = "blue")

## 3차 곡선 회귀
train_df['x3'] = train_df["x"] ** 3

train_x3 = train_df[["x", "x2", "x3"]]
train_y = train_df["y"]

model.fit(train_x3, train_y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k2 = pd.DataFrame({"x": k, "x2": k**2, "x3": k**3})

reg_line3 = model.predict(df_k2)

plt.plot(k, reg_line3, color = 'red')
plt.scatter(train_df["x"], train_df["y"], color = "blue")

## 4차 곡선 회귀
train_df['x4'] = train_df["x"] ** 4

train_x4 = train_df[["x", "x2", "x3", "x4"]]
train_y = train_df["y"]

model.fit(train_x4, train_y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k3 = pd.DataFrame({"x": k, "x2": k**2, "x3": k**3, "x4": k**4})

reg_line4 = model.predict(df_k3)

plt.plot(k, reg_line4, color = 'red')
plt.scatter(train_df["x"], train_df["y"], color = "blue")

## 20차 곡선 회귀
train_df['x5'] = train_df["x"] ** 5
train_df['x6'] = train_df["x"] ** 6
train_df['x7'] = train_df["x"] ** 7
train_df['x8'] = train_df["x"] ** 8
train_df['x9'] = train_df["x"] ** 9
train_df['x10'] = train_df["x"] ** 10
train_df['x11'] = train_df["x"] ** 11
train_df['x12'] = train_df["x"] ** 12
train_df['x13'] = train_df["x"] ** 13
train_df['x14'] = train_df["x"] ** 14
train_df['x15'] = train_df["x"] ** 15
train_df['x16'] = train_df["x"] ** 16
train_df['x17'] = train_df["x"] ** 17
train_df['x18'] = train_df["x"] ** 18
train_df['x19'] = train_df["x"] ** 19
train_df['x20'] = train_df["x"] ** 20

train_x5 = train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                     "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]]
train_y = train_df["y"]

model.fit(train_x5, train_y)

model.coef_
model.intercept_

k = np.linspace(-4, 4, 200)
df_k4 = pd.DataFrame({"x": k, "x2": k**2, "x3": k**3, "x4": k**4, "x5": k**5,
                      "x6": k**6, "x7": k**7, "x8": k**8, "x9": k**9,
                      "x10": k**10, "x11": k**11, "x12": k**12, "x13": k**13, "x14": k**14,
                      "x15": k**15, "x16": k**16, "x17": k**17, "x18": k**18, "x19": k**19, "x20": k**20,})

reg_line5 = model.predict(df_k4)

plt.plot(k, reg_line5, color = 'red')
plt.scatter(train_df["x"], train_df["y"], color = "blue")

# test 데이터 예측
test_df['x2'] = test_df["x"] ** 2
test_df['x3'] = test_df["x"] ** 3
test_df['x4'] = test_df["x"] ** 4
test_df['x5'] = test_df["x"] ** 5
test_df['x6'] = test_df["x"] ** 6
test_df['x7'] = test_df["x"] ** 7
test_df['x8'] = test_df["x"] ** 8
test_df['x9'] = test_df["x"] ** 9
test_df['x10'] = test_df["x"] ** 10
test_df['x11'] = test_df["x"] ** 11
test_df['x12'] = test_df["x"] ** 12
test_df['x13'] = test_df["x"] ** 13
test_df['x14'] = test_df["x"] ** 14
test_df['x15'] = test_df["x"] ** 15
test_df['x16'] = test_df["x"] ** 16
test_df['x17'] = test_df["x"] ** 17
test_df['x18'] = test_df["x"] ** 18
test_df['x19'] = test_df["x"] ** 19
test_df['x20'] = test_df["x"] ** 20

test_x = test_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                  "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]]
y_hat = model.predict(test_x)

### 9차 모델 성능: 0.8949
### 20차 모델 성능: 278823.2788
sum((test_df['y'] - y_hat) ** 2)

# ===================================================

house_train = pd.read_csv('../data/kaggle/houseprice/train.csv')
house_test = pd.read_csv('../data/kaggle/houseprice/test.csv')
sub_df = pd.read_csv('../data/kaggle/houseprice/sample_submission.csv')

## 이상치 탐색 (여기에 넣으면 안 됨)
# house_train = house_train.query("GrLivArea <= 4500")

house_train.shape
house_test.shape
train_n = house_train.shape[0]

df = pd.concat([house_train, house_test], ignore_index=True)

neighborhood_dummies = pd.get_dummies(
    df['Neighborhood'],
    drop_first=True)

x = pd.concat([df[['GrLivArea', 'GarageArea']],
               neighborhood_dummies], axis=1)
y = df['SalePrice']

train_x = x.iloc[:train_n, ]
test_x = x.iloc[train_n:, ]
train_y = y[:train_n]

## Validation 셋 (모의고사 셋) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size=438, replace=False)

valid_x = train_x.loc[val_index]  # 30%
train_x = train_x.drop(val_index) # 70%
valid_y = train_y[val_index]      # 30%
train_y = train_y.drop(val_index) # 70%

## 이상치 탐색 (여기가 맞는 자리)
train_x = train_x.query("GrLivArea <= 4500")

## 선형 회귀 모델 생성
model = LinearRegression()

## 모델 학습
model.fit(train_x, train_y)

## 모델 성능 측정
y_hat = model.predict(valid_x)
np.sqrt(np.mean((valid_y - y_hat) ** 2))