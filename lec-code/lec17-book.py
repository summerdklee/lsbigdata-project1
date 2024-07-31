# @9장

! pip install pyreadstat 

# 아래 순서는 기본 데이터 분석을 위한 준비 절차임
# 0. 패키지 로드
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
raw_welfare = pd.read_spss('data/Koweps_hpwc14_2019_beta2.sav')
welfare = raw_welfare.copy()

# 2. 데이터 검토
welfare.head()
welfare.shape
welfare.info()
# welfare.describe()

# 3. 변수명 변경
welfare = welfare.rename(
    columns = {'h14_g3'     : 'sex',
               'h14_g4'     : 'birth',
               'h14_g10'    : 'marriage_type',
               'h14_g11'    : 'religion',
               'p1402_8aq1' : 'income',
               'h14_eco9'   : 'code_job',
               'h14_reg7'   : 'code_region'})
               
welfare = welfare[['sex', 'birth', 'marriage_type', 'religion', 'income', 'code_job', 'code_region']]
welfare.head(20)
welfare.shape

# 아래 순서는 데이터 분석을 위한 데이터 전처리 절차임

# [성별에 따른 월급 차이 - 성별에 따라 월급이 다를까?]
# (1). 성별 변수 검토 및 전처리
# 0. 변수 검토
# 0-1. 변수 타입 파악
welfare['sex'].dtypes # 변수 타입 파악

# 0-2. 범주마다 몇 명이 있는지, 이상치 파악
welfare['sex'].value_counts() 

# 1. 데이터 전처리
# 1-1. (이상치가 있다면) 이상치 결측 처리

# 1-2. 이해하기 쉽도록 변수값 변경
welfare['sex'] = np.where(welfare['sex'] == 1, 'male', 'female')
welfare['sex'].value_counts()

# 1-3. 빈도 막대 그래프 만들기
sns.countplot(data = welfare, x = 'sex', hue = 'sex')
plt.show()
plt.clf()

# (2). 월급 변수 검토 및 전처리
# 0. 변수 검토
# 0-1. 변수 타입 파악
welfare['income'].dtypes

# 0-2. 요약 통계량 구하기
welfare['income'].describe()

# 0-3. 분포 확인
sns.histplot(data = welfare, x = 'income')
plt.show()
plt.clf()

# 1. 데이터 전처리
# 1-1. 이상치 확인, (이상치가 있다면) 이상치 결측 처리
welfare['income'].describe()

# 1-2. 결측치 확인
welfare['income'].isna().sum()

# 1-3. 성별 월급 평균표(요약표) 생성
sex_income = welfare.dropna(subset = 'income') \
                    .groupby('sex', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))
sex_income

# 1-4. 평균표 막대 그래프 생성
sns.barplot(data = sex_income, x = 'sex', y = 'mean_income', hue = 'sex')
plt.show()
plt.clf()

## 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기
## 위 아래 검정색 막대기로 표시

welfare['birth']
welfare['birth'].describe()

sns.histplot(data = welfare, x = 'birth')
plt.show()
plt.clf()

welfare['birth'].isna().sum()

welfare = welfare.assign(age = 2019 - welfare['birth'] + 1)
welfare['age']

sns.histplot(data = welfare, x = 'age')
plt.show()
plt.clf()

age_income = welfare.dropna(subset = 'income') \
                    .groupby('age') \
                    .agg(mean_income = ('income', 'mean'))
age_income

sns.lineplot(data = age_income, x = 'age', y = 'mean_income')
plt.show()
plt.clf()

# 나이별 income 열에서 na 개수 계산
my_df = welfare.assign(income_na = welfare['income'].isna()) \ 
               .groupby('age', as_index = False) \
               .agg(n = ('income_na', 'sum'))
 
sns.barplot(data = my_df, x = 'age', y = 'n')
plt.show()
plt.clf()

# 연령 범주 설정
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, 'young',
                                np.where(welfare['age'] <=59, 'middle', 'old')))

# 연령 빈도 막대 그래프
sns.countplot(data = welfare, x = 'ageg', hue = 'ageg')
plt.show()
plt.clf()

# 연령별 소득 막대 그래프
ageg_income = welfare.dropna(subset = 'income') \
                     .groupby('ageg', as_index = False) \
                     .agg(mean_income = ('income', 'mean'))

# 연령별 소득 막대 그래프 - 정렬                    
sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income', hue = 'ageg',
                   order = ['young', 'middle', 'old'])
plt.show()
plt.clf()

# 나이 0~9, 10~19, 20~29, 30~39 ... 로 나눠서
bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
pd.cut(welfare['age'], bin_cut)

welfare = welfare.assign(age_group = pd.cut(welfare['age'], bin_cut,
                                  labels = (np.arange(12) * 10).astype(str) + '대'))

welfare['age_group'] = welfare['age_group'].astype('object')

age_income = welfare.groupby('age_group', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))
                    
sns.barplot(data = age_income, x = 'age_group', y = 'mean_income', hue = 'age_group')
plt.rcParams.update({'font.family' : 'Malgun Gothic'})
plt.show()
plt.clf()

sex_age_income = welfare.dropna(subset = 'income') \
                        .groupby(['age_group', 'sex'], as_index = False) \
                        .agg(mean_income = ('income', 'mean'))

sns.barplot(data = sex_age_income, x = 'age_group', y = 'mean_income', hue = 'sex')
plt.show()
plt.clf()

# ==============================================================================

# 9-6장
welfare['code_job']
welfare['code_job'].value_counts()

list_job = pd.read_excel('data/Koweps_Codebook_2019.xlsx', sheet_name = '직종코드')
list_job.head()
list_job.shape

welfare = welfare.merge(list_job, how = 'left', on = 'code_job')
welfare.head()

welfare.dropna(subset = ['job', 'income'])[['income', 'job']]

job_income = welfare.dropna(subset = ['job', 'income']) \
                    .groupby('job', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))
job_income.head()

plt.rcParams.update({'font.family' : 'Malgun Gothic'})

top10 = job_income.sort_values('mean_income', ascending = False).head(10)
sns.barplot(data = top10, y = 'job', x = 'mean_income', hue = 'job')
plt.show()
plt.clf()

bottom10 = job_income.sort_values('mean_income', ascending = True).head(10)
sns.barplot(data = bottom10, y = 'job', x = 'mean_income', hue = 'job').set(xlim = [0, 800])
plt.show()
plr.clf()

# 9-7장
job_male = welfare.dropna(subset = 'job') \
                  .query('sex == "male"') \
                  .groupby('job', as_index = False) \
                  .agg(n = ('job', 'count')) \
                  .sort_values('n', ascending = False) \
                  .head(10)

job_female = welfare.dropna(subset = 'job') \
                    .query('sex == "female"') \
                    .groupby('job', as_index = False) \
                    .agg(n = ('job', 'count')) \
                    .sort_values('n', ascending = False) \
                    .head(10)

# 9-8장
rel_div = welfare.query('marriage_type != 5') \
                 .groupby('religion', as_index = False) \
                 ['marriage_type'] \
                 .value_counts(normalize = True) # 핵심!! value_counts()에서 normalize = True를 쓰면 비율을 구해준다

rel_div = rel_div.query('marriage_type == 1') \
                 .assign(proportion = rel_div['proportion'] * 100) \
                 .round(1)
