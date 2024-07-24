from scipy.stats import bernoulli

# 확률질량함수 pmf
# 확률 변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
bernoulli.pmf(1, 0.3)
bernoulli.pmf(0, 0.3)


# Kaggle 데이터 - house price
import pandas as pd
import numpy as np

house_df = pd.read_csv('data/kaggle/houseprice/train.csv')
house_df.info()
house_df.head()
house_df.tail()
house_df.describe()

price_mean = house_df['SalePrice'].mean()
price_mean

sub_df = pd.read_csv('data/kaggle/houseprice/sample_submission.csv')
sub_df

sub_df['SalePrice'] = price_mean
sub_df

sub_df.to_csv('data/kaggle/houseprice/sample_submission.csv', index = False)

