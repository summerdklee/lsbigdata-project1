import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

df = pd.read_csv('data/kaggle/houseprice/train.csv')
df_test = pd.read_csv('data/kaggle/houseprice/test.csv')
sub_df = pd.read_csv('data/kaggle/houseprice/sample_submission.csv')

house_train = df.copy()
house_test = df_test.copy()
sub = sub_df.copy()

## 이상치 탐색
# house_train['GrLivArea'].sort_values(ascending = False).head(2)
house_train = house_train.query('GrLivArea <= 4500')

## 회귀분석 적합(fit)하기
x = np.array(house_train['GrLivArea']).reshape(-1, 1)
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
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test['GrLivArea']).reshape(-1, 1)
y_pred = model.predict(test_x)

sub["SalePrice"] = y_pred
sub

sub.to_csv("data/kaggle/houseprice/sample_submission4.csv", index = False)

# ==============================================================================

# 변수 2개 사용하기 (숫자 변수만)
df = pd.read_csv('data/kaggle/houseprice/train.csv')
df_test = pd.read_csv('data/kaggle/houseprice/test.csv')
sub_df = pd.read_csv('data/kaggle/houseprice/sample_submission.csv')

house_train = df.copy()
house_test = df_test.copy()
sub = sub_df.copy()

## 이상치 탐색
house_train['GarageArea']
house_train = house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# x = np.array(house_train[['GrLivArea', 'GarageArea']]).reshape(-1, 2)
x = house_train[['GrLivArea', 'GarageArea']]
y = house_train['SalePrice']

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기(a)
model.intercept_ # 절편(b)

slope = model.coef_
intercept = model.intercept_
print(f"기울기 (slope: a): {slope}")
print(f"절편 (intercep: b): {intercept}")

# 함수 생성
def my_houseprice(x, y):
    return (slope[0] * x) + (slope[1] * y) + intercept

a = house_test['GrLivArea']
b = house_test['GarageArea']

my_houseprice(a, b)

test_x = house_test[['GrLivArea', 'GarageArea']]

# 결측치 확인
test_x['GarageArea'].isna().sum()
test_x = test_x.fillna(house_test['GarageArea'].mean())

y_pred = model.predict(test_x)

sub["SalePrice"] = y_pred
sub

sub.to_csv("data/kaggle/houseprice/sample_submission4.csv", index = False)
