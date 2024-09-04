import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

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