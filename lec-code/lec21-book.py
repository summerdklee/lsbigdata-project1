import json
import pandas as pd
import numpy as np

# 데이터 준비
geo = json.load(open('data/SIG.geojson', encoding = 'UTF-8'))
geo.keys()

## 행정 구역 코드 출력
geo['features'][0]['properties']

## 위도, 경도 좌표 출력
geo['features'][0]['geometry']


# 시군구별 인구 데이터 준비
df_pop = pd.read_csv('data/Population_SIG.csv')
df_pop.head()
df_pop.info()

df_pop['code'] = df_pop['code'].astype(str)
