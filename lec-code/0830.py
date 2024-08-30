def g(x=3):
    result = x + 1
    return result

g()
print(g)

# 함수 내용 확인
import inspect
print(inspect.getsource(g))

# if ... else 정식
x = 3
if x > 4:
    y = 1
else:
    y = 2
print(y)

# if ... else 축약형
y = 1 if x > 4 else 2

# 리스트 컴프리헨션
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)

# 조건 3개 이상
x = 0
if x > 0:
 result = "양수"
elif x == 0:
 result = "0"
else:
 result = "음수"
print(result)

# 조건 3개 이상 넘파이 ver.
import numpy as np

x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
choices = ["양수", "0", "음수"]
result = np.select(conditions, choices)
print(result)

# for loop
for i in range(1, 4):
 print(f"Here is {i}")

# 리스트 컴프리헨션
[f"Here is {i}" for i in range(1, 4)]

# f-string
name = "Summer"
age = 28
greeting = f"Hello, my name is {name} and I am {age} years old."
print(greeting)

names = ["John", "Alice"]
ages = np.array([25, 30])

greetings = [f"Hello, my name is {name} and I am {age} years old." for name, age
in zip(names, ages)]
for greeting in greetings:
 print(greeting)

# zip()
names = ["John", "Alice"]
ages = np.array([25, 30])

## zip() 함수로 names와 ages를 병렬적으로 묶음
zipped = zip(names, ages)

## 각 튜플을 출력
for name, age in zipped:
 print(f"Name: {name}, Age: {age}")

# while문
i = 0
while i <= 10:
  i += 3
  print(i)

# while, break문
i = 0
while True:
  i += 3
  if i > 10:
    break
  print(i)

# apply
import pandas as pd

data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print(df)

df.apply(sum, axis=0)
df.apply(max, axis=0)
df.apply(max, axis=1)

def my_func(x, const=3):
 return max(x)**2 + const

df.apply(my_func, axis=1)
df.apply(my_func, axis=1, const=5)

# apply 넘파이 ver.
array_2d = np.arange(1, 13).reshape((3, 4), order='F')
print(array_2d)

# 함수 환경
y = 2

def my_func(x):
 y = 1
 result = x + y
 return result

my_func(3)
print(y)

def add_many(*args):
  result = 0
  for i in args:
    result = result + i
  return result

add_many(1, 2, 3)