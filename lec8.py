# lec8 행렬

import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
print("행렬:\n", matrix)

matrix.shape

# 행렬의 크기를 재어주는 shape 속성
print("행렬의 크기:", matrix.shape)

np.zeros(5)
np.zeros([5, 4])
np.arange(1, 7).reshape((2, 3))
np.arange(1, 7).reshape((2, -1)) # 직접 행/열 개수를 계산하기 힘들때, reshape 열 자리에 -1을 작성하면 알아서 만들어줌

# 예제: 0에서부터 99까지의 수 중에서 랜덤으로 50개 숫자를 추출하고, 5 by 10 행렬을 만드세요. 
np.random.seed(2024) # seed: 랜덤 값을 고정해주는 역할
a = np.random.randint(0, 100, 50) # np.random.randit('시작 숫자'부터, '끝 숫자-1'까지, n개 추출)
a.reshape((5, 10))

np.arange(1, 21).reshape((4, -1))
mat_a = np.arange(1, 21).reshape((4, 5), order = 'F') # 열 우선 순서 > 원소가 가로로 채워지도록 설정 / order = 'C'는 세로

# 인덱싱
mat_a[0, 0] # 1
mat_a[1, 1] # 2
mat_a[2, 3] # 15
mat_a[0:2, 3] # [13, 14]
mat_a[1:3, 1:4]

# 행/열 자리가 비어있는 경우: 전체 행, 또는 열 선택
mat_a[3, ] # 4번 행의 모든 열 전체
mat_a[3, :]
mat_a[3, ::2]

mat_b = np.arange(1, 101).reshape((20, -1))
mat_b[1::2, ] # 짝수 행만 가져오기
mat_b[[1, 4, 6, 14], ]

# 필터링: 조건에 맞는 데이터만 추출
x = np.arange(1, 11).reshape((5, 2)) * 2
x[[True, True, False, False, True], 0]

mat_b[:, 1] # 1차원(벡터)로 자동 변경
mat_b[:, 1:2] # 행렬(매트릭스) 형태 유지
mat_b[:, [1]] # 행렬(매트릭스) 형태 유지
mat_b[:, (1, )] # 행렬(매트릭스) 형태 유지

mat_b[:, 1].reshape((-1, 1)) # 1차원(벡터)를 행렬(매트릭스) 형태로 바꿀때

mat_b[mat_b[:, 1] % 7 == 0, :]

# 사진은 행렬이다
import matplotlib.pyplot as plt

## 난수를 생성하여 3X3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

a = np.random.randint(0, 256, 20).reshape(4, -1)

plt.clf()
plt.imshow(a / 255, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


x = np.arange(1, 11).reshape((5, 2)) * 2
x.transpose()

import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

# 이미지 읽기
! pip install imageio
import imageio

jelly = imageio.imread("img/jelly.png")
type(jelly)

jelly.shape # jelly 이미지는 3차원 > 행열이 총 4장 겹쳐 있음 
jelly[:, :, 0] # 첫 번째 장의 모든 행열 추출
jelly[:, :, 0].transpose() # 90도 돌리기

plt.imshow(jelly[:, :, 0].transpose())
plt.axis('off')
plt.show()
plt.clf()

mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape

# 다차원 행렬 필터링/결과를 읽을 때는 오른쪽에서부터 보기 (열, 행, 장수 번째, ...)
my_array2 = np.array([my_array, my_array])
my_array2
my_array2[0, :, :, :]
my_array2.shape

my_array[:, :, [0, 2]]
my_array[:, 0, :]
my_array[0, 1, [1, 2]] # my_array[0, 1, 1:3]도 같은 결과

mat_x = np.arange(1, 101).reshape((5, 5, 4))
mat_y = np.arange(1, 101).reshape((-1, 5, 2))
mat_x = np.arange(1, 101).reshape((5, -1, -1))
mat_x
mat_y


# 넘아피 배열 메서드
a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()
a.sum(axis = 0) # 열끼리 합계
a.sum(axis = 1) # 행끼리 합계

a.mean()
a.mean(axis = 0)
a.mean(axis = 1)

mat_b = np.random.randint(0, 100, 50).reshape((5, -1))

mat_b.max()
mat_b.max(axis = 0)
mat_b.max(axis = 1)

a = np.array([1, 3, 2, 5]).reshape((2, 2))

a.cumsum()
a.cumsum(axis = 0)
a.cumsum(axis = 1)

a.cumprod()
a.cumprod(axis = 0)
a.cumprod(axis = 1)

mat_b.reshape((2, 5, 5)).shape
mat_b.flatten()

d = np.array([1, 2, 3, 4, 5])

d.clip(2, 4)
d.tolist()


# 균일 확률 변수 만들기
import numpy as np

?np.random.rand
np.random.rand(1)

def X(num) :
    return np.random.rand(num)

X(1)    

# 베르누이 확률변수 만들기 (모수 : p)
def Y(num, p):
    x = np.random.rand(num)
    return np.where(x < p, 1, 0)

Y(1000000, 0.5).mean()

# 새로운 확률변수: 가질 수 있는 값(0, 1, 2) >> 이해 절대 안됨
def Z():
    x = np.random.rand(1)
    return np.where(x < 0.2, 0, np.where(x < 0.7, 1, 2))

Z()

p = np.array([0.2, 0.5, 0.3])
def Z(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1, 2))

Z(p)
