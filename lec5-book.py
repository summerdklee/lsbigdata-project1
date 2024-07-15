import pandas as pd
import numpy as np

df = pd.DataFrame({'name' : ['김지훈', '이유진', '박동현', '김민지'],
                'english' : [90, 80, 60, 70],
                'math' : [50, 60, 100, 20]})
df
type(df)

df["name"]
type(df["name"]) # Series: 열 이름이 맨 마지막에 나오고, 데이터 타입을 보여줌

sum(df["english"]) / 4

# 84p 예제
fruits = pd.DataFrame({'product' : ['사과', '딸기', '수박'],
                        'price' : [1800, 1500, 3000],
                        'selling_amount' : [24, 38, 13]})
fruits

price_avg = sum(fruits['price']) / 3
selling_amount_avg = sum(fruits['selling_amount']) / 3
price_avg
selling_amount_avg

df_exam = pd.read_excel("data/excel_exam.xlsx")
df_exam

sum(df_exam["math"]) / 20
sum(df_exam["english"])/ 20
sum(df_exam["science"])/ 20

len(df_exam) # 행 개수 반환
df_exam.shape # (행 개수, 열 개수) 튜플로 반환
df_exam.size # 행 개수 * 열 개수 반환

df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam

df_exam["mean"] = df_exam["total"] / 3
df_exam

df_exam[df_exam["math"] > 50] # math > 50가 True인 데이터만 추출됨
df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)]


# 예제: 수학는 평균 이상인 학생 중 영어는 평균 이하인 학생은?
mean_m = np.mean(df_exam['math'])
mean_e = 

df_exam[(df_exam["math"] > mean_m)]

# 예제: 3반 학생 데이터만 추출
df_exam['nclass'] == 3
df_exam[df_exam['nclass'] == 3]

# 예제: 3반 학생 데이터 중 math, english, science 데이터만 추출
df_nc3 = df_exam[df_exam['nclass'] == 3]
df_nc3[["math", 'english', 'science']]
df_nc3[0:1]

df_exam
df_exam[7:16]
df_exam[0::2]

df_exam.sort_values('math', ascending = False)
df_exam.sort_values(['nclass', 'math'], ascending = [True, False])

np.where(a > 3) # 조건에 부합하는 요소의 위치를 찾아서 추출, 튜플 형태로 추출
np.where(a > 3, "UP", "Down") # True면 Up, False면 Down 추출, np array 형태로 추출

df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")
df_exam
