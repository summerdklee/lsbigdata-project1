import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

penguins = load_penguins()

df = penguins.dropna()
df = df[['bill_length_mm', 'bill_depth_mm']]
df = df.rename(columns={'bill_length_mm' : 'y',
                        'bill_depth_mm' : 'x'})

# 원래 MSE는?
np.mean((df['y'] - df['y'].mean()) ** 2)

# x = 15 기준으로 나눴을때, 데이터 포인터가 몇개씩 나뉘는지?
n1 = df.query('x < 15').shape[0] # 1번 그룹
n2 = df.query('x >= 15').shape[0] # 2번 그룹

y_hat1 = df.query('x < 15').mean()[0] # 1번 그룹은 얼마로 예측하나요?
y_hat2 = df.query('x >= 15').mean()[0] # 2번 그룹은 얼마로 예측하나요?

# 각 그룹의 MSE
mse1 = np.mean((df.query('x < 15')['y'] - y_hat1) ** 2) # 1그룹
mse2 = np.mean((df.query('x >= 15')['y'] - y_hat2) ** 2) # 2그룹

# x = 15의 MSE 가중평균은?
(mse1 * n1 + mse2 * n2) / (n1 + n2)

# x = 20의 MSE 가중평균은?
n1 = df.query('x < 20').shape[0]
n2 = df.query('x >= 20').shape[0]

y_hat1 = df.query('x < 20').mean()[0]
y_hat2 = df.query('x >= 20').mean()[0]

mse1 = np.mean((df.query('x < 20')['y'] - y_hat1) ** 2)
mse2 = np.mean((df.query('x >= 20')['y'] - y_hat2) ** 2)

(mse1 * n1 + mse2 * n2) / (n1 + n2)

# MSE 함수 만들기 : 기준값 x를 넣으면 MSE값이 나오는 함수
def my_mse(num):
    n1 = df.query(f'x < {num}').shape[0]
    n2 = df.query(f'x >= {num}').shape[0]

    y_hat1 = df.query(f'x < {num}').mean()[0]
    y_hat2 = df.query(f'x >= {num}').mean()[0]

    mse1 = np.mean((df.query(f'x < {num}')['y'] - y_hat1) ** 2)
    mse2 = np.mean((df.query(f'x >= {num}')['y'] - y_hat2) ** 2)

    return (mse1 * n1 + mse2 * n2) / (n1 + n2)

my_mse(20)

# 13~22 사이 값 중 0.01 간격으로 MSE 계산 후, minimize 사용하여 가장 작은 MSE가 나오는 x 찾기
df['x'].min() # 13.1
df['x'].max() # 21.5

## 내가 끄적인거...
x_values = np.arange(13.2, 21.4, 0.01)
result = []
for i in range(len(x_values)):
    result.append(my_mse(x_values[i]))

x_values[np.argmin(result)]

## 샘 코드
x_values = np.linspace(13.2, 21.4, 100)
result = np.repeat(0.0, 100)
for i in range(100):
    result[i] = my_mse(x_values[i])

x_values[np.argmin(result)]

# 첫번째 기준값 : 16.4, 두번째 기준값은 얼마가 되어야 하는지?
# 깊이 2 트리의 기준값 두 개 구하기
## depth 1
group1 = df.query('x < 16.4')
group2 = df.query('x >= 16.4')

## depth 2
### group1
group1['x'].min() # 13.1
group1['x'].max() # 16.3

x_values = np.arange(13.11, 16.3, 0.01)
result = []
for i in range(len(x_values)):
    result.append(my_mse(x_values[i]))

x_values[np.argmin(result)] # 16.01

### group2
group2['x'].min() # 16.4
group2['x'].max() # 21.5

x_values = np.arange(16.51, 21.5, 0.01)
result = []
for i in range(len(x_values)):
    result.append(my_mse(x_values[i]))

x_values[np.argmin(result)] # 16.41

# ========================================== 아영이 코드

group1 = df.query("x < 16.4")# 1번 그룹
group2 = df.query("x >= 16.4")  # 2번 그룹

# depth2 
def my_mse(data, x):
    n1 = data.query(f"x < {x}").shape[0]  # 1번 그룹
    n2 = data.query(f"x >= {x}").shape[0]  # 2번 그룹 

    y_hat1 = data.query(f"x < {x}")['y'].mean() # 1번 그룹 예측값
    y_hat2 = data.query(f"x >= {x}")['y'].mean() # 2번 그룹 예측값

      # 각 그룹의 MSE는 얼마인가요?
    mse1 = np.mean((data.query(f"x < {x}")['y'] - y_hat1)**2)
    mse2 = np.mean((data.query(f"x >= {x}")['y'] - y_hat2)**2)

    return (mse1*n1 + mse2*n2) / (n1+n2) 

my_mse(group1, 14)

x_values1 = np.arange(group1['x'].min()+0.01, group1['x'].max(), 0.01)
result1 = np.repeat(0.0, len(x_values1))
for i in range(0, len(x_values1)):
    result1[i] = my_mse(group1, x_values1[i])
x_values1[np.argmin(result1)] # 14.01
result1.min()

x_values2 = np.arange(group2['x'].min() + 0.01, group2['x'].max(), 0.01)
result2 = np.repeat(0.0, len(x_values2))
for i in range(0, len(x_values2)):
    result2[i] = my_mse(group2, x_values2[i])
x_values[np.argmin(result2)] # 19.4

# x, y 산점도 그래프 & 평행선 4개
thresholds = [14.01, 16.42, 19.4]
df['group'] = np.digitize(df['x'], thresholds)
y_mean = df.groupby('group').mean()['y']

k1 = np.linspace(13, 14.01, 100)
k2 = np.linspace(14.01, 16.42, 100)
k3 = np.linspace(16.42, 19.4, 100)
k4 = np.linspace(19.4, 22, 100)

df.plot(kind='scatter', x='x', y='y', color = 'black', s=5)
plt.axvline(x=16.4, color='b', linestyle=':')
plt.axvline(x=14.01, color='r', linestyle=':')
plt.axvline(x=19.4, color='r', linestyle=':')

plt.scatter(k1, np.repeat(y_mean[0], 100), color='r', s=5)
plt.scatter(k2, np.repeat(y_mean[1], 100), color='r', s=5)
plt.scatter(k3, np.repeat(y_mean[2], 100), color='r', s=5)
plt.scatter(k4, np.repeat(y_mean[3], 100), color='r', s=5)

# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 랜덤 시드 고정 (재현성을 위해)
np.random.seed(42)

# x 값 생성: -10에서 10 사이의 100개 값
x = np.linspace(-10, 10, 100)

# y 값 생성: y = x^2 + 정규분포 노이즈
y = x ** 2 + np.random.normal(0, 10, size=x.shape)

# 데이터프레임 생성
df = pd.DataFrame({'x': x, 'y': y})

# 데이터 시각화
plt.scatter(df['x'], df['y'], label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Quadratic Data')
plt.legend()
plt.show()

# 입력 변수와 출력 변수 분리
X = df[['x']]  # 독립 변수는 2차원 형태로 제공되어야 함
y = df['y']

# 학습 데이터와 테스트 데이터로 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 디시전 트리 회귀 모델 생성 및 학습
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 테스트 데이터의 실제 값과 예측 값 시각화
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()