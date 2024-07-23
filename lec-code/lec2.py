a = 1
a

# . 현재 폴더
# .. 상위 폴더

# show folder in new window : 해당 위치 탐색기

# cd(change directory) : 폴더 이동
# ls : 폴더 내 세부 정보 확인

a = 7
a

a = "안녕하세요!"
a

a = '안녕합니다.'
a

a = "'안녕하세요!'라고 아빠가 말했다."
a

a = [1, 2, 3]
a

a2 = [4, 5, 6]
a2

a + a2

str1 = '안녕하세요!'
str2 = 'LS 빅데이터 스쿨!'

str1 + str2
str1 + ' ' + str2

print(a)
print(a + a2)

a = 10
b = 3.3

print('a + b =', a + b)
print('a - b =', a - b)
print('a * b =', a * b)
print('a / b =', a / b)
print('a % b =', a % b)
print('a // b =', a // b)
print('a ** b =', a ** b)

(a ** 3) % 7
(a ** 3) % 7
(a ** 3) % 7

a != b
a == b
a > b
a < b
a >= b
a <= b

a = ((2 ** 4) + (12453 // 7)) % 8
b = ((9 ** 7) / 12) * (36452 % 253)
a < b

print(a)
print(b)

user_age = 17
is_adult = user_age >= 18
print("성인입니까?", is_adult)

TRUE = 3
true = 5

a = "True"
b = TRUE 
c = true
d = True

print(b)
print(c)

a = True
b = False

print(a and b)
print(a or b)
print(not a)
print(not b)

# and 연산자
True and False
True and True
False and False
False and True

# or 연산자
True or False
False or False

# True = 1, False = 0
True + True
True + False
False + False

# and는 곱셈(*)
True * False
True * True
False * False
False * True

# or는 덧셈(+)
True  + False
True  + True
False + False
False + True

a = True
b = True
a or b
max(a + b, 1)

a= 3
a += 10
a

a -= 4
a

a %= 3
a

a += 12
a

a **= 2
a

str1 = "hello"
str1 + str1
str1 * 2

# 문자열 반복
repeated_str = str1 * 2
print("Repeated string :", repeated_str)

# 정수 : int (integer)
# 실수 : float (double)

# 단항 연산자
x = 5
x
+x
-x
~x

# bin() : binary의 약자, 2진수 문자열로 변환해주는 함수
bin(5)
bin(-5)
bin(-6)

x = 1
~x

bin(1)
bin(-2)

x = -3
bin(~x)
~x

x = 5
~x
bin(~x)

max(3, 4)

var1 = [1,2,3]
sum(var1)
min(var1)

pip install pydataset
! pip install pydataset

import pydataset
pydataset.data()

df = pydataset.data('cake')
df

! pip install pandas
! pip install numpy

import pandas as pd
import numpy as np

pd.data()
np.data()

# 테스트
