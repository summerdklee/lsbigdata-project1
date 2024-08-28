import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
poly = PolynomialFeatures(degree=20, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=3, shuffle=True, random_state=2024)

# 알파 값 설정 : lambda
alpha_values = np.arange(0, 10, 0.01)

# 각 알파 값에 대한 교차 검증 점수 저장
mean_scores = []

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha, max_iter=5000) # max_iter : alpha의 coef를 minimize를 통해 구할때, max 5000번까지 시행해서 구해라
    scores = cross_val_score(lasso, X_poly, y, cv=kf, scoring='neg_mean_squared_error') # valid score
    mean_scores.append(np.mean(scores))

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df['validation_error'].max()

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# ==========================================================================================

df_train = pd.read_csv("../data/kaggle/houseprice/train.csv")
df_test = pd.read_csv("../data/kaggle/houseprice/test.csv")
df_sub = pd.read_csv("../data/kaggle/houseprice/sample_submission.csv")

# 숫자형
## 데이터 전처리
df_train_num = df_train.select_dtypes(include=['number'])
df_train_num = df_train_num.fillna(df_train_num.mean())

X_num = df_train_num.iloc[:, 1:-1]
y_num = df_train_num["SalePrice"].tolist()
y_num = np.array(y_num)

## 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_num, y_num, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return score

## Lasso 모델 생성 및 학습
lasso = Lasso(alpha=0.01)
lasso.fit(X_num, y_num)  # 모델 학습

## coef_와 intercept_ 출력
coef = lasso.coef_
intercept = lasso.intercept_

print("Lasso Coefficients:", coef)
print("Lasso Intercept:", intercept)

## 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 300, 0.01)
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

## submission
df_test_num = df_test.select_dtypes(include=['number'])
df_test_num = df_test_num.fillna(df_train_num.mean())

X_test_num = df_test_num.iloc[:, 1:]
y_test_pred = lasso.predict(X_test_num)

df_sub['SalePrice'] = y_test_pred

df_sub.to_csv("../data/kaggle/houseprice/sample_submission8.csv", index=False)

# 범주형
## 데이터 전처리
a = df_train.select_dtypes(include=['object', 'string'])
b = df_test.select_dtypes(include=['object', 'string'])

df_str = pd.concat([a, b], axis=0, ignore_index=True)
df_str = df_str.fillna('unknown')

str_dummies = pd.get_dummies(df_str, drop_first=True)
df_train_str = str_dummies.iloc[:1460, :]
df_test_str = str_dummies.iloc[1460:, :]

df_train_str.describe()
df_test_str.describe()

X_str = df_train_str
y_str = df_train["SalePrice"]
y_str = np.array(y_str)

## 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_str, y_str, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return score

## Lasso 모델 생성 및 학습
lasso = Lasso(alpha=0.01)
lasso.fit(X_str, y_str)  # 모델 학습

## coef_와 intercept_ 출력
coef = lasso.coef_
intercept = lasso.intercept_

print("Lasso Coefficients:", coef)
print("Lasso Intercept:", intercept)

## 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(100, 300, 0.1)
mean_scores = np.zeros(len(alpha_values))

k = 0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_str, y_str)  # 이 부분도 모델을 학습시켜야 합니다.
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

## submission
X_test_str = df_test_str
y_test_str_pred = lasso.predict(df_test_str)

df_sub['SalePrice'] = y_test_str_pred

df_sub.to_csv("../data/kaggle/houseprice/sample_submission9.csv", index=False)