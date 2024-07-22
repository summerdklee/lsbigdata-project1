# 리스트(list)
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

## 빈 리스트 생성
empty_list1 = []
empty_list2 = list()

## 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))

## 인덱싱과 슬라이싱
range_list[3] = 'LS 빅데이터 스쿨'
range_list[2] = ['1st', '2nd', '3rd']
range_list[2][2]
range_list

## 리스트 내포(Comprehension)
# 1. 대괄호로 싸여있다 > '리스트'
# 2. 넣고 싶은 수식 표현을 x를 사용해 표현
# 3. for ... in ...을 사용해서 원소 정보 제공
list(range(10))
squares = [x**2 for x in range(10)]
squares

my_squares = [x**3 for x in [3, 5, 2, 15]]
my_squares

# 넘파이 어레이로도 가능
import numpy as np
my_squares_2 = [x**3 for x in np.array([3, 5, 2, 15])]
my_squares_2

# 판다스 시리즈로도 가능
import pandas as pd

exam = pd.read_csv('data/exam.csv')

my_squares_3 = [x**3 for x in exam['math']]
my_squares_3

## 리스트 연결, 반복
3 + 2
'안녕' + '하세요'

list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1 * 3) + (list2 * 5)

numbers = [5, 2, 3]

repeated_list = [x for x in numbers for _ in range(3)]
### '_'의 의미
### 1. 앞에 나온 값을 가리킴
5 + 4
_ + 6 # 앞서 연산된 수인 '9'가 '_'로 지칭됨

### 2. 값 생략, 자리 차지(placeholder)
a, _, b = (1, 2, 4)
a; b
_

### for 루프 문법
### for i in 범위:
### 작동방식
for x in [4, 1, 2, 3]:# 리스트를 넣으면, 리스트 내 숫자가 의미가 있는 게 아니라,
    print(x)          # 해당 리스트의 원소 개수만큼 반복됨

for i in range(5):
    print(i**2)
    
#### 예제: 리스트 생성 후 for 루프를 이용해 2, 4, 6, 8, ..., 20까지의 수를 채워보세요.
my_list = []
for x in range(1, 11):
    my_list.append(x * 2) # .append(): 리스트 마지막 자리에 원소 추가

my_list2 = [0] * 10
for i in range(10):
    my_list2[i] = (i + 1) * 2

my_list3 = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
my_list2 = [0] * 10
for i in range(10):
    my_list2[i] = my_list3[i]

#### 예제2: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에서 가져오기
my_list2 = [0] * 5
my_list3 = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
for i in range(5):
    my_list2[i] = my_list3[i * 2]

### 리스트 컴프리헨션으로 바꾸는 방법
#### 1. 바깥은 무조건 대괄호로 묶어줌 > 리스트로 반환하기 위해서
#### 2. for 루프의 :는 생략
#### 3. 실행 부분을 먼저 써줌
#### 4. 결과값을 발생하는 표현만 남겨두기
my_list = []
[x * 2 for x in range(1, 11)]
[i for i in numbers]

for i in [0, 1]:
    for j in [4, 5, 6]:
        print(i, j)

numbers = [5, 2, 3]
for i in numbers:
    for j in range(4):
        print(i)

[i for i in numbers for j in range(4)]

## 원소 체크
fruits = ["apple", "apple", "banana", "cherry"]

print("banana가 리스트에 포함되어 있나요?", "banana" in fruits)
print("grape가 리스트에 포함되어 있나요?", "grape" in fruits)

[x == "banana" for x in fruits]

mylist = []
for x in fruits:
    mylist.append(x == 'banana')
mylist

import numpy as np
fruits = np.array(fruits) # 리스트를 np어레이로 변경
int(np.where(fruits == 'banana')[0][0])

## 리스트 메서드
### .reverse(): 원소 거꾸로 정렬
fruits.reverse()
fruits

### .append(): 새로운 원소를 리스트 마지막 위치에 추가
fruits.append('watermelon')
fruits

### .insert(): 특정 위치에 원소 삽입
fruits.insert(0, 'mango')
fruits.insert(3, 'pineapple')
fruits

### .remove(): 지우려고 하는 원소 중, 첫번째로 나온 원소만 지워준다
fruits.remove('apple')
fruits

#### 넘파이로 제거하기
# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, items_to_remove)
# mask = ~np.isin(fruits, ['banana', 'apple']) > 위와 동일한 결과

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)
