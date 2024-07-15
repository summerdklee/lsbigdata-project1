# 벡터
import numpy as np
import pandas as pd

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

a.shape
b.shape


# 벡터 슬라이싱

## 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024) # seed: 랜덤 값을 고정해주는 역할
a = np.random.randint(1, 21, 10)
print(a)

?np.random.seed
?np.random.randint

## 두 번째 값 추출
print(a[1])

a[2:5]
a[-1] # 맨 끝에서부터 첫 번째 인덱스
a[-2]
a[::2] # a[start:stop:step]
a[0:10:2]
a[1:6:2]

# 벡터 슬라이싱 예제, 1~1000사이 3의 배수 합?
var = np.arange(1, 1001)
var

var[2:1001:3]
sum(var[2:1001:3])
sum(var[::3])

print(a[[0, 2, 4]])
print(np.delete(a, 3))
print(np.delete(a, [1, 3])) # 서로 다른 자리의 수를 한번에 삭제할때

a > 3
a[a > 3] # 3 이상인 원소들만 추출


print(b)

np.random.seed(2024) 
a = np.random.randint(1, 10000, 5)

a < 5000
a[(a > 2000) & (a < 5000)] # a[조건을 만족하는 논리형 벡터]


# 15 이상, 25 이하인 데이터 개수는?
import pydataset

df = pydataset.data('mtcars')
np_df = np.array(df['mpg'])
model_names = np.array(df.index)

sum(((np_df >= 15) & (np_df <= 25)))

# 평균 mpg보다 높은(이상) 자동차 대수는?
len(np_df)
sum(np_df >= np.mean(np_df)) #  np.mean() : 평균값

# 15보다 작거나, 22 이상인 데이터 개수는?
sum((np_df < 15) | (np_df >= 22))

# 15 이상, 25 이하인 자동차 모델은?
model_names[(np_df >= 15) & (np_df <= 20)]

# 평균 mpg보다 높은 연비를 가진 자동차 모델은?
model_names[np_df >= np.mean(np_df)]

# 평균 mpg보다 낮은 연비를 가진 잩동차 모델은?
model_names[np_df < np.mean(np_df)]


np.random.seed(2024) 
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]


a[a > 3000] = 3000
a


np.random.seed(2024)
a = np.random.randint(1, 100, 10)
np.where(a < 50) # np.where(): True가 있는 인덱스 자리의 위치를 반환

np.random.seed(2024)
b = np.random.randint(1, 26346, 1000)
b

# b에서 처음으로 22000보다 큰 숫자가 나왔을 때, 그 숫자의 위치와 그 숫자는 무엇일까요?
x = np.where(b >22000)
x
type(x)
my_index = x[0][0] # 조건에 맞는 숫자의 인덱스 위치 추출
b[my_index] # 조건에 맞는 숫자 추출

# b에서 처음으로 24000보다 큰 숫자가 나왔을 때, 그 숫자의 위치와 그 숫자?
y = np.where(b > 24000)
my_index2 = y[0][0]
b[my_index2]

# 처음으로 10000보다 큰 숫자가 나왔을때, 50번째로 나오는 숫자 위치와 그 숫자?
z = np.where(b > 10000)
my_index3 = z[0][49]
b[my_index3]

# 500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자 위치와 그 숫자?
n = np.where(b < 500)
my_index4 = n[0][-1]
b[my_index4]


a = np.array([20, np.nan, 13, 24, 309])
np.isnan(a)
a

a + 3
np.mean(a)
np.nanmean(a)

?np.nan_to_num
np.nan_to_num(a, nan = 0)


a = None
b = np.nan
a
b

a_filtered = a[~np.isnan(a)]
a_filtered


# 벡터 합치기
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str) # 벡터는 한 가지 데이터 타입만 허용 가능
mix_vec

combined_vec = np.concatenate([str_vec, mix_vec])
combined_vec

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16))) # np.row_stack은 이전 버전, np.vstack으로 사용 권장
row_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)

uneven_stacked = np.vstack((vec1,vec2))
uneven_stacked
uneven_stacked.shape

?np.resize
vec1 = np.resize(vec1, len(vec2))
vec1

a = np.array([12, 21, 35, 48, 5])
a[0::2] # 홀수번째 요소
a.max()

a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a) # 중복값 제거

# 주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성하세요.
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])

x = np.empty(6)
x[0::2] = a # 혹은 x[[0, 2, 4]] = a
x[1::2] = b # 혹은 x[[1, 3, 5]] = b
x
