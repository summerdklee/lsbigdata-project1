import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

tab3 = pd.read_csv('data/tab3.csv')

tab1 = pd.DataFrame({'id' : np.arange(1, 13),
                    'score' : tab3['score']})

tab2 = pd.DataFrame({'id' : np.arange(1, 13),
                    'score' : tab3['score'],
                    'gender' : np.where('id < 8', 'female', 'male')})

tab2 = tab1.assign(gender = ['female']*7 + ['male']*5)

# 1표본 t 검정 (그룹 1개)
## 귀무가설 vs. 대립가설
## H0 : mu = 10 vs. Ha: mu != 10
## 유의수준 5%
result = ttest_1samp(tab1['score'], popmean = 10, alternative = 'two-sided')

t_value = result[0] # t 검정통계량
p_value = result[1] # 유의확률 (p-value)
tab1['score'].mean() # 표본평균

result.statistic # t 검정통계량
result.pvalue # 유의확률 (p-value)
result.df # 자유도
### 유의확률 0.0645이 유의수준 0.05보다 크므로, 귀무가설을 기각하지 못한다.
#### 귀무가설이 참일때, 11.53이 관찰될 확률이 6.48%이므로,
#### 이것은 0.05보다(유의수준) 크므로, 귀무가설을 거짓이라 판단하기 힘들다.
#### 유의확률 0.0648이 유의수준 0.05보다 크므로, 귀무가설을 기각하지 못한다.

ci = result.confidence_interval(confidence_level = 0.95) # 신뢰구간

# 2표본 t 검정 (그룹 2개) - 분산이 같고, 다를때
# 분산이 같은 경우: 독립 2표본 t 검정
# 분산이 다를 경우: 웰치스 t 검정

## 귀무가설 vs. 대립가설
## H0 : mu_m = mu_f vs. Ha: mu_m > mu_f
## 유의수준 1%
male = tab2[tab2['gender'] == 'male']
female = tab2[tab2['gender'] == 'female']

# alternative = 'less'의 의미:
# 대립가설이 '첫 번째 입력 그룹의 평균이 두 번째 입력 그룹의 평균보다 작다.'고 설정된 경우
result2 = ttest_ind(female['score'], male['score'], equal_var = True, alternative = 'less')

t_value2 = result2[0] # t 검정통계량
p_value2 = result2[1] # 유의확률 (p-value)
female['score'].mean() # 표본평균
male['score'].mean() # 표본평균

result2.statistic # t 검정통계량
result2.pvalue # 유의확률 (p-value)
result2.df # 자유도

ci2 = result2.confidence_interval(0.99) # 신뢰구간

# 3그룹 대응표본 t 검정 (짝지을 수 있는 표본)
## 귀무가설 vs. 대립가설
## (1) H0 : mu_before = mu_after vs. Ha: mu_after > mu_before
## (2) H0 : mu_diff = 0 vs. Ha: mu_diff > 0
### md_diff = mu_after - mu_before = 0
## 유의수준 1%

## mu_diff에 대응하는 표본으로 변환: (2) H0 : mu_diff = 0 vs. Ha: mu_diff > 0
### pivot 활용: id열은 그대로 index, group열 내 유니크한 value를 기준으로, score값을 채워줌
tab3
tab3_data = tab3.pivot_table(index = 'id', columns = 'group', values = 'score').reset_index() # long to wide
# tab3_data.melt(id_vars = 'id', value_var = ['A', 'B'], var_name = "group", value_name = 'score') # wide to long
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
tab3_data = tab3_data[['score_diff']]
tab3_data

result3 = ttest_1samp(tab3_data['score_diff'], popmean = 0, alternative = 'greater')

t_value3 = result3[0] # t 검정통계량
p_value3 = result3[1] # 유의확률 (p-value)

# 연습1
df = pd.DataFrame({'id' : [1, 2, 3],
                   'A' : [10, 20, 30],
                   'B' : [40, 50, 60]})
                   
df_melt = df.melt(id_vars = 'id',
                  value_vars = ['A', 'B'],
                  var_name = 'group',
                  value_name = 'score')

df_pivot = df_melt.pivot_table(index = 'id',
                               columns = 'group',
                               values = 'score').reset_index()

# 연습2
tips = sns.load_dataset('tips')

## long to wide - 요일 기준
tips_wide = 
tips_wide = tips.reset_index(drop = False) \
                .pivot_table(index = 'index', columns = 'day', values = 'tip').reset_index()
