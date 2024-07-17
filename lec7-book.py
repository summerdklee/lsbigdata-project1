# 167p. 데이터 합치기

import pandas as pd
import numpy as np

# 가로로 합치기
## 중간고사 데이터
test1 = pd.DataFrame({'id'       : [1, 2, 3, 4, 5],
                      ' midterm' : [60, 80, 70 ,90, 85]})
                      
## 기말고사 데이터
test2 = pd.DataFrame({'id'    : [1, 2, 3, 4, 5],
                      'final' : [70, 83, 65, 95, 80]})

## Left Join
total = pd.merge(test1, test2, how = 'left', on = 'id') # how: 오른쪽 데이터를 왼쪽에 붙일때 'left' > 왼쪽 데이터가 기준이 됨
total                                                   # on: 기준이 되는 변수 'id'

## Right Join
total = pd.merge(test1, test2, how = 'right', on = 'id')
total

## Inner Join: 교집합 값들만 합치기
total = pd.merge(test1, test2, how = 'inner', on = 'id')
total

## Outer Join: 합집합으로 데이터 전체 합치기
total = pd.merge(test1, test2, how = 'outer', on = 'id')
total

## 담임교사 데이터
exam = pd.read_csv('data/exam.csv')

name = pd.DataFrame({'nclass'  : [1, 2, 3, 4, 5],
                     'teacher' : ['kim', 'lee', 'park', 'choi', 'jung']})
name

## exam 데이터에 name 데이터 합치기
exam_new = pd.merge(exam, name, how = 'left', on = 'nclass')
exam_new


# 세로로 쌓기
score1 = pd.DataFrame({'id'   : [1, 2, 3, 4, 5],
                      'score' : [60, 80, 70 ,90, 85]})
                      
score2 = pd.DataFrame({'id'    : [6, 7, 8, 9, 10],
                      'score'  : [70, 83, 65, 95, 80]})
                      
## concat()
score_all = pd.concat([score1, score2])
score_all

pd.concat([test1, test2], axis = 1)


--------------------------------------------------------------------------------
# 178p. 데이터 정제

## 결측치 만들기
df = pd.DataFrame({'sex'   : ['M', 'F', np.nan, 'M', 'F'],
                   'score' : [5, 4, 3, 4, np.nan]})
df

df['score'] + 1 # nan값은 연산해도 nan

pd.isna(df) # nan값만 True값을 반환해주는 함수, 데이터 프레임 형식
pd.isna(df).sum()

## 결측치 제거하기
df.dropna(subset = 'score') # score 변수에서 결측치 제거
df.dropna(subset = ['score', 'sex']) # 여러 변수에서 결측치 제거
df.dropna() # 모든 변수 결측치 제거

## 결측치 대체하기
exam = pd.read_csv('data/exam.csv')

### df.loc[행 인덱스, 열 이름]:데이터 프레임 location을 사용한 인덱싱
exam.loc[[2, 7, 14], ['math']] = np.nan
exam

### df.iloc[행 인덱스, 열 인덱스]: 조회하려면 무조건 숫자 벡터로 이루어져야 함

exam.iloc[[2, 7, 14], 2] = np.nan
exam

df[df['score'] == 3.0]['score'] = 4.0

## 예제: 수학 점수가 50점 이하인 학생들의 점수를 50점으로 상향 조정
exam = pd.read_csv('data/exam.csv')

exam.loc[exam['math'] <= 50, ['math']] = 50

## 예제2: 영어 점수가 90점 이상인 학생들의 점수를 90점으로 하향 조정
exam.iloc[exam['english'] >= 90, 3] = 90
exam.iloc[exam[exam['english'] >= 90].index, 3] = 90


# 예제3 : 수학 점수 50점 이하를 -로 변경
exam = pd.read_csv('data/exam.csv')

exam.loc[exam['math'] <= 50, 'math'] = "-"
exam

# 예제4 : '-' 결측치를 수학 점수 평균으로 변경
## 풀이1
exam = pd.read_csv('data/exam.csv')
exam.loc[exam['math'] <= 50, 'math'] = "-"

## 풀이1
math_mean = exam.loc[exam['math'] != '-', 'math'].mean() # - 값을 제외한 mean값
math_mean
exam.loc[exam['math'] == '-', 'math'] = math_mean

## 풀이2
math_mean2 = exam.loc[exam['math'] == '-', 'math'] = exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean2

## 풀이3
math_mean3 = exam.loc[exam['math'] != '-', 'math'].mean()
exam.loc[exam['math'] == '-', 'math'] = np.nan
exam.loc[pd.isna(exam['math']), 'math'] = math_mean3

## 풀이4
vector = np.nanmean(np.array([np.nan if x == '-' else float(x) for x in exam['math']])) # np.nanmean(): 넘파이에서 nan 값을 제거하고 평균을 구해주는 함수
# vector2 = np.array([float(x) if x != '-' else np.nan for x in exam['math']])
exam['math'] = np.where(exam['math'] == '-', vector, exam['math'])

## 풀이5
math_mean4 = exam.loc[exam['math'] != '-', 'math'].mean()
exam['math'] = exam['math'].replace('-', math_mean4)
