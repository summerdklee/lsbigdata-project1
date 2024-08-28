import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x})

train_df = df.loc[:19]

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

valid_df = df.loc[20:]

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y = valid_df["y"]

# [그래프 그리기]
model = Lasso(alpha=0.03)
model.fit(train_x, train_y)

x_pred = np.arange(-4, 4, 0.01)
x_pred_poly = np.column_stack([x_pred ** i for i in range(1, 21)])  # 상수항 추가하지 않음
y_pred = model.predict(x_pred_poly)

plt.scatter(valid_df["x"], valid_df["y"], color="blue", label="Validation Data")
plt.plot(x_pred, y_pred, color="red", label="Lasso Prediction")
plt.xlabel("x")
plt.ylabel("y")

# [train/valid 셋 3개로 그래프 그리기]
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({
    "y" : y,
    "x" : x})

for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

myindex = np.random.choice(30, 30, replace=False)

def make_tr_val(fold_num, df, cv_num=3):
    np.random.seed(2024)
    myindex = np.random.choice(30, 30, replace=False)

    # valid index
    val_index = myindex[(fold_num*10):((fold_num*10)+10)]

    # valid set, train set
    valid_set = df.loc[val_index]
    train_set = df.drop(val_index)

    train_X = train_set.iloc[:, 1:]
    train_y = train_set.iloc[:, 0]

    valid_X = valid_set.iloc[:, 1:]
    valid_y = valid_set.iloc[:, 0]

    return (train_X, train_y, valid_X, valid_y)

val_result_total = np.repeat(0.0, 300).reshape(3, -1)
tr_result_total = np.repeat(0.0, 300).reshape(3, -1)

for j in np.arange(0, 3):
    train_X, train_y, valid_X, valid_y = make_tr_val(fold_num=j, df=df)

    # 결과 받기 위한 벡터 만들기
    val_result = np.repeat(0.0, 100)
    tr_result = np.repeat(0.0, 100)

    for i in np.arange(0, 100):
        model = Lasso(alpha=i*0.01)
        model.fit(train_X, train_y)

        # 모델 성능
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        perf_train = sum((train_y - y_hat_train)**2)
        perf_val = sum((valid_y - y_hat_val)**2)
        tr_result[i]=perf_train
        val_result[i]=perf_val

    tr_result_total[j,:] = tr_result
    val_result_total[j,:] = val_result

tr_result_total.mean(axis=0)
val_result_total.mean(axis=0)

