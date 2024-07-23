# 자격에 대하여
## 어떤 일에 대한 본인의 자격은 자신의 생각이 크게 중요하지 않다.
## 중요한 것은 타인이 생각하는 그 포지션에 대한 내 자격이다.

# 확률 - 기대값
import numpy as np

x = np.arange(33) # 확률변수 X가 가질 수 있는 값
x.sum() / 33 # X의 기대값

x - 16
(x - 16) ** 2
np.unique((x - 16) ** 2)

sum(np.unique((x - 16) ** 2) * (2/33))

# E[X^2]
sum((x ** 2) * (1 / 33))

# Var(X) = E[X^2] - (E[X]^2)
sum((x ** 2) * (1 / 33)) - 16 ** 2

X = 0, 1, 2, 3

## 예제
y = np.arange(4)
pro_y = np.array([1/6, 2/6, 2/6, 1/6])

Ex = sum(y * pro_y) # (E[X]^2)
Exx = sum((y ** 2) * pro_y) # E[X^2]

Exx - (Ex ** 2) # Var(X)
sum((y - Ex) ** 2 * pro_y) # Var(X)

## 예제2
x = np.arange(99)

# 1~50, 50~1 벡터 생성
x_1_50_1 = np.concatenate((np.arange(1, 51), np.arange(49, 0, -1)))
pro_x = x_1_50_1 / 2500

Ex = sum(x * pro_x) # (E[X]^2)
Exx = sum((x ** 2) * pro_x) # E[X^2]

Exx - (Ex ** 2) # Var(X)

## 예제3
y = np.arange(0, 7, 2) # np.arange(4) * 2
pro_y = np.array([1/6, 2/6, 2/6, 1/6])

Ex = sum(y * pro_y) # (E[Y]^2)
Exx = sum((y ** 2) * pro_y) # E[Y^2]

Exx - (Ex ** 2) # Var(Y)
