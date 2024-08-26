import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 데이터 로드
house_train = pd.read_csv('../data/kaggle/houseprice/train.csv')
house_test = pd.read_csv('../data/kaggle/houseprice/test.csv')
sub_df = pd.read_csv('../data/kaggle/houseprice/sample_submission.csv')

# train + test 데이터 셋 합치기
combine_df = pd.concat([house_train, house_test], ignore_index = True) # ignore_index: 인덱스 생성 무시

# 더미변수 생성
neighborhood_dummies = pd.get_dummies(
    combine_df["Neighborhood"],
    drop_first=True)

# 더미변수 데이터를 train, test으로 분리
train_dummies = neighborhood_dummies.iloc[:1460,]

test_dummies = neighborhood_dummies.iloc[1460:,]
test_dummies = test_dummies.reset_index(drop=True)

# 필요한 변수들만 골라서 더미 데이터 합치기
my_train = pd.concat([house_train[["SalePrice", "GrLivArea", "GarageArea"]],
               train_dummies], axis=1)

my_test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]],
               test_dummies], axis=1)

# train 데이터의 길이
train_n = len(my_train) # 1460

## Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438,
                 replace = False) #30% 정도의 갯수를 랜덤으로 고르기.

new_valid = my_train.loc[val_index]  # 30% 438개
new_train = my_train.drop(val_index) # 70% 1022개

# 이상치 탐색 및 삭제
new_train = new_train.query("GrLivArea <= 4500")

# train 데이터에서 가격 분리
train_x = new_train.iloc[:,1:]
train_y = new_train[["SalePrice"]]

valid_x = new_valid.iloc[:,1:]
valid_y = new_valid[["SalePrice"]]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)

# 성능 측정
y_hat = model.predict(valid_x)
np.mean(np.sqrt((valid_y-y_hat)**2)) #26265
