# soft copy
a = [1, 2, 3]

b = a

a[1] = 4
a
b

id(a)
id(b)


# deep copy
a = [1, 2, 3]
a

b = a[:] # 이후 a의 값을 변경해도, b의 값은 변경되지 않음
b = a.copy() # 이후 a의 값을 변경하면, b의 값도 변경됨

a[1] = 4
a
b


# 숫자와 친해지기

import math # 파이썬에 내장된 모듈이지만, import는 해줘야 함

x = 4
math.sqrt(x) # 제곱근
sqrt_val = math.sqrt(16)
print("16의 제곱근은:", sqrt_val)

math.exp(x) # 지수
exp_val = math.exp(5)
print("e^5의 값은:", exp_val)

math.log(x, [base]) # 로그
log_val = math.log(10, 10)
print("10의 밑 10 로그 값은:", log_val)

math.factorial(x) # 팩토리얼
fact_val = math.factorial(5)
print("5의 팩토리얼은:", fact_val)

math.sin(x) # 삼각 함수 - 사인
sin_val = math.sin(math.radians(90))
print("90도의 사인 함수 값은:", sin_val)

math.cos(x) # 삼각 함수 - 코사인
cos_val = math.cos(math.radians(180))
print("180도의 코사인 함수 값은:", cos_val)

math.tan(x) # 삼각 함수 - 탄젠트
tan_val = math.tan(math.radians(45))
print("45도의 탄젠트 함수 값은:", tan_val)

math.radians(x) # 도(degree) 단위 x를 라디안으로 변환

math.degrees(x) # 라이단 단위 x를 도(degree)로 변환


## 함수 생성 실습

### 정규분포 함률밀도함수(PDF)
def my_normal_pdf(x, mu, sigma):
  part_1 = 1 / (sigma * math.sqrt(2 * math.pi)) # 혹은 (sigma * math.sqrt(2 * math.pi)) ** -1
  part_2 = math.exp((-(x - mu) ** 2) / (2 * sigma) ** 2)
  return part_1 * part_2

my_normal_pdf(3, 3, 1)
  
### 예제
def practice(x, y, z):
  part_1 = (x ** 2) + math.sqrt(y) + math.sin(z)
  part_2 = math.exp(x)
  return part_1 * part_2

practice(2, 9, math.pi / 2)

def my_f(x, y, z):
  return ((x ** 2) + math.sqrt(y) + math.sin(z)) * math.exp(x)

my_f(2, 9, math.pi / 2)

### 예제
def my_g(x):
    return math.cos(x) + math.sin(x) * math.exp(x)

my_g(math.pi)

### 코드 스니펫: 자주 사용하는 함수 템플릿을 저장하고, 자동으로 불러올 수 있도록 하는 기능
def fname(input): # fcn 함수
    contents      # 코드 스니펫을 불러오는 키보드 숏컷: Shift + Space
    return        # ex) fcn 입력 후 > Shift + Space

import pandas as pd # pandas 패키지 import

import numpy as np # numpy 패키지 import


# 벡터와 친해지기기

## Ctrl +SHift + C: 주석 처리

## numpy
import numpy as np

a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성

type(a)
a[3]
a[2:]
a[1:4]

print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

## 벡터 생성

### 빈 array 생성
b = np.empty(3)
b

b[0] = 1
b[1] = 4
b[2] = 10
b
b[2]

### np.arange([start, ]stop, [step, ]dtype=None)
vec1 = np.arange(100) # 0~99
vec1

vec2 = np.arange(1, 100) # 1~99
vec2

vec3 = np.arange(1, 101, 0.5) # 1~100, 0.5의 간격으로
vec3

### numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
l_space1 = np.linspace(0, 1, 5)
l_space1

l_space2 = np.linspace(0, 1, 5, endpoint = False)
l_space2

?np.linspace

np.repeat(1, 5) # 1을 5번 반복하는 벡터 생성

vec1 = np.arange(5)
np.repeat(vec1, 5) # vec1 값을 5번씩 반복하는 벡터 생성

vec2 = np.arange(-100, 1) # 음수에서 양수로 커지는 벡터
vec2

vec3 = np.arange(0, -100, -1) # 양수에서 음수로 작아지는 벡터 1
vec3

vec4 = -np.arange(0, 100) # 양수에서 음수로 작아지는 벡터 2
vec4

### repeat() vs tile()

vec1 = np.arange(5)
np.repeat(vec1, 3)
np.tile(vec1, 3)

vec1 * 2
vec1 / 3
vec1 +vec1 # 벡터의 동일 인덱스끼리 연산

max(vec1)
sum(vec1)

#### 예제:35672 이하 홀수들의 합은?
vec2 = np.arange(1, 35673, 2)
sum(vec2)
vec2.sum()

### len() vs shape()

len(vec2)
vec2.shape # 튜플 형태로 반환

b = np.array([[1, 2, 3], [4, 5, 6]])

length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size


a = np.array([1, 2])
b = np.array([1, 2, 3, 4])
a + b # 길이가 맞지 않으면 연산 불가

np.tile(a, 2) + b # 길이를 맞춰주는 방법
np.repeat(a, 2)+ b

b == 3

#### 예제: 10보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?
vec = np.arange(1, 10)
x = vec % 7
sum(x == 3) # True = 1인 것을 활용 
