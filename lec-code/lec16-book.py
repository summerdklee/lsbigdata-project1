# 9장

! pip install pyreadstat

# 패키지 로드
import pandas as pd
import numpy as np
import seaborn as sns

# 데이터 로드
raw_welfare = pd.read_spss('data/Koweps_hpwc14_2019_beta2.sav')
welfare = raw_welfare.copy()

# 데이터 검토
welfare.head()
welfare.shape
welfare.info()
welfare.describe()
