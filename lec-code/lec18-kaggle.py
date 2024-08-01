import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# y = 2x + 3의 그래프를 그려보세요!
# 기울기: 직선의 각도
# 절편: 직선의 위치
a = 1
b = 0

x = np.linspace(-5, 5, 100)
y = (a * x) + b

plt.axvline(0, color = 'gray')
plt.axhline(0, color = 'gray')
plt.plot(x, y, color = 'blue')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.clf()

# ==============================================================================

df = pd.read_csv('data/kaggle/houseprice/train.csv')
house_train = df.copy()

a = 16
b = 117

x = np.linspace(0, 5, 100)
y = (a * x) + b

my_df = house_train[['BedroomAvg', 'SalePrice']]
my_df['SalePrice'] = my_df['SalePrice']
plt.scatter(data = my_df, x = 'BedroomAbvGr', y = 'SalePrice', color = 'pink')
plt.plot(x, y, color = 'brown')
plt.show()
plt.clf()

test =  pd.read_csv('data/kaggle/houseprice/test.csv')
test_df = test[["Id", "BedroomAbvGr"]]
test_df['SalePrice'] = (a * test_df.loc[:, ['BedroomAbvGr']] + b) * 1000
test_df = test_df[["Id", "SalePrice"]]

test_df.to_csv('sample_submission3.csv', index = False)

# 직선 성능 평가
a = 70
b = 10

## 절대값 사용: sum(|y - y_hat|)
y_hat = ((a * house_train['OverallQual']) + b) * 1000
y = house_train['SalePrice']

np.abs(y - y_hat)
np.sum(np.abs(y - y_hat))

## 제곱 사용: sum((y - y_hat) ** 2) = sum((y - (ax + b)) ** 2) > 보편적으로 더 많이 사용함
### 해당 식의 값을 최소로 만들어 주는 a/b를 구하고, 그 직선을 '회귀직선'이라고 부른다.
y_hat = ((a * house_train['OverallQual']) + b) * 1000
y = house_train['SalePrice']

(y - y_hat) ** 2
np.sum(np.abs(y - y_hat))

! pip install scikit-learn

from sklearn.linear_model import LinearRegression

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기(a)
model.intercept_ # 절편(b)

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope: a): {slope}")
print(f"절편 (intercep: b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# house price predict
## 회귀분석 적합(fit)하기
x = np.array(house_train['OverallQual']).reshape(-1, 1)
y = np.array(house_train['SalePrice'])

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기(a)
model.intercept_ # 절편(b)

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope: a): {slope}")
print(f"절편 (intercep: b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# submission
y_pred = pd.DataFrame(y_pred)
test =  pd.read_csv('data/kaggle/houseprice/test.csv')
test_df = test[["Id"]]
test_df['SalePrice'] = y_pred
test_df.to_csv('sample_submission3.csv', index = False)
