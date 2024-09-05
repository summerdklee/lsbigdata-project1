import numpy as np
from scipy.stats import chi2 # 독립성
from scipy.stats import chi2_contingency # 동질성
from scipy.stats import chisquare # 적합도
from statsmodels.stats.proportion import proportions_ztest # 비율

##### 귀무가설 : 동일하다/현재 결과에 변함이 없다. #####
##### p-value가 특정 유의 수준보다 작으면 귀무가설 기각 #####

# 112p 동질성 검정
## 문제 1
## 귀무가설 : 정당 지지와 핸드폰 사용 유무는 독립이다. (관련없다.)
## 대립가설 : 정당 지지와 핸드폰 사용 유무는 독립이 아니다.

mat_a = np.array([[49, 47], [15, 27], [32, 30]])

chi2, p, df, expected = chi2_contingency(mat_a, correction=False)
chi2.round(3)
p.round(4)

### 결론 : 유의수준 0.05보다 p값이 크므로, 귀무가설을 기각할 수 없다.
expected

# 104p 적합도 검정
observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected)

print("Test statistic: ", statistic.round(3))
print("p༡value: ", p_value.round(3))
print("Expected: ", expected)

# 106p 비율 검정
z_score, p_value = proportions_ztest(45, 82, 0.5, alternative='larger')
z_score2, p_value2 = proportions_ztest(45, 82, 0.7, alternative='larger')

print("x༡sqaured:",z_score**2)
print("x༡sqaured2:",z_score2**2)

# 112p
## 귀무가설 : 선거구별 후보A의 지지율이 동일하다.
## 대립가설 : 선거구별 후보A의 지지율이 동일하지 않다.

mat_b = np.array([[176, 124], [193, 107], [159, 141]])

chi2, p, df, expected = chi2_contingency(mat_b, correction=False)
chi2.round(3)
p.round(4)

### 결론 : 유의수준 0.05보다 p값이 작으므로, 귀무가설을 기각한다.

# =======================================================================

# [카이제곱 검정]
# - 독립성 검정 : 두 변수가 독립인지 vs. 아닌지
# - 동질성 검정 : 두 그룹별 분포가 동일한지 vs. 아닌지
#   - 3 sample 이상 비율 검정 : p1 = p2 = p3 vs. 다른게 하나라도 있는지 > 찬성/반대
# - 적합도 검정 : 데이터가 특정 분포를 따르는지 vs. 아닌지

# [비율 검정 (z 검정)] > t 검정과 거의 동일하다고 봐도 무방함
# - 1 sample : p(모비율) = p0(귀무가설이 생각하는 p값) vs. p != p0
# - 2 sample : p1 = p2 vs. p1 != p2