import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("../data/kaggle/blueberry/train.csv")
df_test = pd.read_csv("../data/kaggle/blueberry/test.csv")
df_sub = pd.read_csv("../data/kaggle/blueberry/sample_submission.csv")

df_train.head()
df_train.info()
df_train.describe()
df_train.shape
df_train.mean()

df_test.head()
df_test.info()
df_test.describe()
df_test.shape

## 데이터 전처리
X_num = df_train.iloc[:, 1:-1]
y_num = df_train["yield"].tolist()
y_num = np.array(y_num)

## 교차 검증 설정
kf = KFold(n_splits=3, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_num, y_num, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return score

## 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 100, 0.1)
mean_scores = np.zeros(len(alpha_values))

k = 0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_num, y_num)  # 이 부분도 모델을 학습시켜야 합니다.
    mean_scores[k] = rmse(lasso)
    k += 1

## 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

## 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

## Lasso 모델 생성 및 학습
lasso = Lasso(alpha=optimal_alpha)
lasso.fit(X_num, y_num)  # 모델 학습

## coef_와 intercept_ 출력
coef = lasso.coef_
intercept = lasso.intercept_

print("Lasso Coefficients:", coef)
print("Lasso Intercept:", intercept)

## submission
X_test = df_test.iloc[:, 1:]
y_test_pred = lasso.predict(X_test)

df_sub['yield'] = y_test_pred

df_sub.to_csv("../data/kaggle/blueberry/sample_submission3.csv", index=False)

# ============================================================== alpha = 0 / 362.76102, 368.38262

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("../data/kaggle/blueberry/train.csv")
df_test = pd.read_csv("../data/kaggle/blueberry/test.csv")
df_sub = pd.read_csv("../data/kaggle/blueberry/sample_submission.csv")

## 데이터 전처리
X_num = df_train.iloc[:, 1:-1]
y_num = df_train["yield"].tolist()
y_num = np.array(y_num)

## 교차 검증 설정
kf = KFold(n_splits=200, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_num, y_num, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return score

## 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 500, 1)
mean_scores = np.zeros(len(alpha_values))

k = 0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_num, y_num)  # 이 부분도 모델을 학습시켜야 합니다.
    mean_scores[k] = rmse(lasso)
    k += 1

## 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

## 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

## Lasso 모델 생성 및 학습
lasso = Lasso(alpha=optimal_alpha)
lasso.fit(X_num, y_num)  # 모델 학습

## coef_와 intercept_ 출력
coef = lasso.coef_
intercept = lasso.intercept_

print("Lasso Coefficients:", coef)
print("Lasso Intercept:", intercept)

## submission
X_test = df_test.iloc[:, 1:]
y_test_pred = lasso.predict(X_test)

df_sub['yield'] = y_test_pred

df_sub.to_csv("../data/kaggle/blueberry/sample_submission.csv", index=False)

# =========================================================== alpha = 0.25 / 362.82659, 368.59640


# 특성 표준화
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# 표준화된 데이터를 사용하여 교차 검증 및 모델 학습
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_num_scaled, y_num, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return score

# 다양한 알파 값에 대한 교차 검증
alpha_values = np.arange(0, 1, 0.01)
mean_scores = np.zeros(len(alpha_values))

k = 0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_num_scaled, y_num)
    mean_scores[k] = rmse(lasso)
    k += 1

# 최적의 알파 값 찾기
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 최적의 알파 값으로 Lasso 모델 학습
lasso = Lasso(alpha=optimal_alpha)
lasso.fit(X_num_scaled, y_num)

# 계수와 절편 출력
coef = lasso.coef_
intercept = lasso.intercept_

print("Lasso Coefficients:", coef)
print("Lasso Intercept:", intercept)

# 제출 파일 준비
X_test_scaled = scaler.transform(df_test.iloc[:, 1:])
y_test_pred = lasso.predict(X_test_scaled)

df_sub['yield'] = y_test_pred
df_sub.to_csv("../data/kaggle/blueberry/sample_submission4.csv", index=False)
