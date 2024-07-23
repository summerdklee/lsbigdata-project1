import pandas as pd
import numpy as np

# 챕터 5
# 데이터 탐색(파악) 함수

## head()
## tail()
## shape
## info()
## describe()

exam = pd.read_csv('data/exam.csv')
exam.head()
exam.tail()
exam.shape # 괄호 없이 입력함(shape는 어트리뷰트이기 때문에, 함수가 아님!! > 어트리뷰트는 괄호가 없다)
exam.info()
exma.describe()


type(exam) # 판다스 데이터프레임
var = [1, 2, 3] # 리스트
exam # 객체
var # 각각의 객체에 제공되는 메서드가 달라짐, 리스트에는 head()를 사용할 수 없는 등


exam2 = exam.copy() # 데이터프레임 복제
exam2 = exam2.rename(columns = {'nclass' : 'class'}) # 변수명 수정


exam2['total'] = exam2['math'] + exam2['english'] + exam2['science'] # 각 과목 점수 합을 기재한 total열 추가
exam2

exam2['test'] = np.where(exam2['total'] >= 200, 'pass', 'fail') # total 점수가 200점 이상이면 pass,
exam2                                                           # 미만이면 fail을 기재한 text열 추가


exam2['test'].value_counts() # test열의 빈도표 생성

import matplotlib.pyplot as plt
count_test = exam2['test'].value_counts()
count_test.plot.bar(rot = 0)
plt.show()
plt.clf() # 현재 그려진 그래프 클리어

exam2.describe()
exam2['grade'] = np.where(exam2['total'] >= 230, "A",
                 np.where(exam2['total'] >= 210, "B",
                 np.where(exam2['total'] >= 190, "C", "D")))
exam2

exam2['retake'] = np.where(exam2['grade'].isin(['C', 'D']), 'Y', 'N')
exam2


# 챕터 6
# 데이터 전처리 함수

# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam = pd.read_csv('data/exam.csv')

# .query(): 조건에 맞는 행을 걸러내줌
exam.query("nclass == 1") # exam[exam['nclass'] == 1]도 동일 결과
exam.query('nclass != 2')

exam.query('math > 50')
exam.query('math < 50')
exam.query('english >= 50')
exam.query('english <= 80')

exam.query('nclass == 1 & math >= 50')
exam.query('nclass == 2 & english >=80')

exam.query('math >= 90 | english >=90')
exam.query('english < 90 | science < 50')

exam.query('nclass == 1 | nclass == 3 | nclass == 5')
exam.query('nclass in [1, 3, 5]')
exam.query('nclass == [1, 3, 5]')
exam.query('nclass not in [1, 3, 5]')
exam.query('nclass != [1, 3, 5]')
exam[exam['nclass'].isin([1, 3, 5])]
exam[~exam['nclass'].isin([1, 3, 5])]

# df[]: 필요한 변수만 추출
exam['nclass'] > 3 # type: 판다스 시리즈
exam[['id', 'nclass']] # type: 데이터 프레임, 판다스 시리즈가 아닌 DF로 유지하고 싶은 경우 대괄호 2개 사용

# .drop(): 변수 제거
exam.drop(columns = 'math') # 일시 삭제, 진짜 삭제하고 싶은 경우에는 해당 문구를 변수에 지정해줘야 함
exam.drop(columns = ['math', 'english'])


exam.query('nclass == 1')[['math', 'english']]
exam.query('nclass == 1') \ # '\'을 사용하면 줄을 나누어 작성할 수 있음
[['math', 'english']]


# .sort_value(): 정렬
exam.sort_values('math')
exam.sort_values('math', ascending = False)
exam.sort_values(['nclass', 'math'], ascending = [True, False])

# .assign(): 변수 추가
exam = exam.assign(
    total = exam['math'] + exam['english'] + exam['science'],
    mean = (exam['math'] + exam['english'] + exam['science']) / 3) \
    .sort_values('total', ascending = False)

# lambda 함수
exam2 = pd.read_csv('data/exam.csv')

exam2 = exam2.assign(
    total = lambda x: x['math'] + x['english'] + x['science'],
    mean = lambda x: x['total'] / 3 \
    .sort_values('total', ascending = False)
    
# .agg(): 요약
# .groupby(): 그룹 나누기
exam2.agg(mean_math = ('math', 'mean'))
exam2.groupby('nclass') \               # nclass를 기준으로
     .agg(mean_math = ('math', 'mean')) # math열의 평균을 구해라
     
## 예제: 반별, 과목별 평균
exam2.groupby('nclass') \
     .agg(mean_math = ('math', 'mean'),
          mean_english = ('english', 'mean'),
          mean_science = ('science', 'mean'),
          mean_nclass = ('mean', 'mean'))

## 예제
mpg = pd.read_csv('data/mpg.csv')

mpg.query('category == "suv"') \                    # category열의 suv를 추출
   .assign(total = (mpg['hwy'] + mpg['cty']) / 2) \ # 합산 연비(total) 변수 생성
   .groupby('manufacturer') \                       # manufacturer열 기준으로 분리
   .agg(mean_tot = ('total', 'mean')) \             # 합산 연비 평균(mean_tot) 구하기
   .sort_values('mean_tot', ascending = False) \    # mean_tot을 내림차순으로 정렬
   .head()
