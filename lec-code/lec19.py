from scipy.optimize import minimize
import numpy as np

# y = x^2 + 3의 최소값 구하기
def my_f(x) :
    return (x ** 2) + 3

my_f(3)

## 초기 추정값
initial_guess = [10]

## 최소값 찾기 
result = minimize(my_f, initial_guess)

## 결과 출력
result.fun
result.x

# z = x^2 + y^2 + 3의 최소값 구하기
def my_f2(x):
    return (x[0] ** 2) + (x[1] ** 2) + 3

my_f2([1, 3])

## 초기 추정값
initial_guess = [-10, 3]

## 최소값 찾기 
result = minimize(my_f2, initial_guess)

## 결과 출력
result.fun
result.x

# k = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7의 최소값 구하기
def my_f3(x):
    return ((x[0] - 1) ** 2) + ((x[1] - 2) ** 2) + ((x[2] - 4) ** 2) + 7

## 초기 추정값
initial_guess = [1, 2, 4]

## 최소값 찾기 
result = minimize(my_f3, initial_guess)

## 결과 출력
result.fun # 최소값
result.x # 최소값을 갖는 x값
