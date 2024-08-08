# ! pip install plotly

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_covid19_100 = pd.read_csv('data/df_covid19_100.csv')
df_covid19_100
df_covid19_100.info()

# data 속성
fig = go.Figure(
      data = [
        {"type" : "scatter",
         "mode" : "markers",
         "x" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
         "y" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
         "marker" : {"color" : "orange"} 
        },
        
        {"type" : "scatter",
         "mode" : "lines",
         "x" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
         "y" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
         "line" : {"color" : "#5E88FC", "dash" : "dash"}
         }
])
fig.show()

# layout 속성
margins_P = {
    'l' : 50,
    'r' : 50,
    't' : 50,
    'b' : 50
}

fig = go.Figure(
      data = 
        {"type" : "scatter",
         "mode" : "markers+lines",
         "x" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
         "y" : df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
         "marker" : {"color" : "orange"},
         "line" : {"color" : "#5E88FC", "dash" : "dash"}
        },
      layout = 
      {'title' : '코로나19 발생 현황',
      'xaxis' : {'title' : '날짜', 'showgrid' : False},
      'yaxis' : {'title' : '확진자수'},
      'margin' : margins_P})
fig.show()

# 애니메이션
frames = []
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique()

for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date)
    }
    frames.append(frame_data)
    

## x축과 y축의 범위 설정
x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]


## 애니메이션을 위한 레이아웃 설정
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False, "range": x_range},
    "yaxis": {"title": "확진자수", "range": y_range},
    "margin": margins_P,
    "updatemenus": [{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

## Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)
fig.show()

# ==============================================================================
# ! pip install palmerpenguins
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()
penguins.columns

fig = px.scatter(penguins,
                 x = 'bill_length_mm',
                 y = 'bill_depth_mm',
                 color = 'species',
                 trendline = 'ols') # 134p

## 책보고 한거
fig.update_layout(title = dict(text = '<b>팔머펭귄 종별 부리 길이 vs. 깊이</b>',
                               x = 0.5, font = dict(color = 'white')),
                  margin = dict(t = 50, b = 25, l = 25, r = 25),
                  paper_bgcolor = 'black', plot_bgcolor = 'black',
                  xaxis = dict(color = 'white', ticksuffix = '부리 길이(mm)', showgrid = False),
                  yaxis = dict(color = 'white', gridcolor = 'gray',
                               ticksuffix = '부리 깊이(mm)', dtick = 100),
                  legend = dict(font = dict(color = 'white')))
                  
## issac 샘                  
fig.update_layout(
    title=dict(text="<b>팔머펭귄 종별 부리 길이 vs. 깊이</b>", x = 0.5, font=dict(color="white", size = 25)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(title = dict(text = '펭귄 종', font = dict(color = 'white')), font=dict(color="white"))
)

fig.update_traces(marker=dict(size=10)) # 점 크기 크게
fig.show()

# ==============================================================================
# 선형회귀
from sklearn.linear_model import LinearRegression

model = LinearRegression()

penguins = penguins.dropna()

x = penguins[['bill_length_mm']]
y = penguins['bill_depth_mm']

model.fit(x, y) # 데이터에 결측치가 있으면 안 돌아간다
linear_fit = model.predict(x)

model.coef_ # 기울기
model.intercept_ # 절편

fig.add_trace(go.Scatter(
    mode = 'lines',
    x = penguins['bill_length_mm'],
    y = linear_fit,
    name = '선형추세선',
    line = dict(dash = 'dot', color = 'white')
))
fig.show() # Simpson's Paradox 결과

# 범주형 변수로 회귀분석 진행
# 범주형 변수인 'species'를 더미 변수로 변환
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=True)
penguins_dummies.columns
penguins_dummies.iloc[:, -3:]

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_ # 기울기
model.intercept_ # 절편

regline_y = model.predict(x)

sns.scatterplot(x = x['bill_length_mm'], y = y, hue = penguins['species'], palette = 'deep', legend = False)
sns.scatterplot(x = x['bill_length_mm'], y = model.predict(x), color = 'black')
plt.show()
plt.clf()

# ==============================================================================

house_train=pd.read_csv("data/kaggle/houseprice/train.csv")
house_test=pd.read_csv("data/kaggle/houseprice/test.csv")
sub_df=pd.read_csv("data/kaggle/houseprice/sample_submission.csv")

# house_train=house_train.query("GrLivArea <= 4500")

neighborhood_dummies = pd.get_dummies(
                                      house_train['Neighborhood'],
                                      drop_first=True)
                   
x = pd.concat([house_train[['GrLivArea', 'GarageArea']],
              neighborhood_dummies], axis = 1)
y = house_train["SalePrice"]

model = LinearRegression()

model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

model.coef_      # 기울기 a
model.intercept_ # 절편 b

neighborhood_dummies_test = pd.get_dummies(
                                           house_test['Neighborhood'],
                                           drop_first=True)

test_x = pd.concat([house_test[['GrLivArea', 'GarageArea']],
                   neighborhood_dummies_test], axis = 1)

test_x.isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("data/kaggle/houseprice/sample_submission6.csv", index=False)
