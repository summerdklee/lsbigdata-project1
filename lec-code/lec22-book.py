# ! pip install folium

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
import webbrowser

# 데이터 준비
geo_seoul = json.load(open('data/SIG_Seoul.geojson', encoding = 'UTF-8'))
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul['features']
geo_seoul['features'][0]

## 행정 구역 코드 출력
geo_seoul['features'][0]['properties']
geo_seoul['features'][1]['properties']
geo_seoul['features'][2]['properties']
geo_seoul['features'][3]['properties']

## 위도, 경도 좌표 출력
geo_seoul['features'][0]['geometry']
coordinate_list = geo_seoul['features'][5]['geometry']['coordinates']
len(coordinate_list[0][0])

coordinate_list = np.array(coordinate_list[0][0])
x = coordinate_list[:, 0]
y = coordinate_list[:, 1]

plt.plot(x[::3], y[::3])
plt.show()
plt.clf()

## 함수 생성
def draw_seoul(num):
    gu_name = geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_list = np.array(coordinate_list[0][0])
    x = coordinate_list[:, 0]
    y = coordinate_list[:, 1]
    
    plt.rcParams.update({'font.family':'Malgun Gothic'})
    plt.plot(x[::2], y[::2], color = 'black')
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None

draw_seoul(23)

# 서울시 전체 지도 그리기
location = np.arange(0, 25)
df1 = pd.DataFrame({})

# 리스트를 쌓는 반복문
name = []
x = []
y = []
gu_num = len(geo_seoul["features"])

for i in range(gu_num):
    coordinate_list = geo_seoul["features"][i]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    gu_x = coordinate_array[:, 0]
    gu_y = coordinate_array[:, 1]
    x.extend(gu_x)
    y.extend(gu_y)
    coor_num = len(coordinate_array)
    for j in range(coor_num):
        gu_name = geo_seoul["features"][i]["properties"]['SIG_KOR_NM']
        name.append(gu_name)


len(name) #38231 # 총 좌표 갯수임.
len(x)
len(y)

# 리스트를 합쳐서 데이터프레임 만들기
mydata = pd.DataFrame({"name" : name,
                       "x" : x,
                       "y" : y})
mydata.head()
mydata.tail()

plt.plot(x, y)
plt.show()
plt.clf()

# 구 이름 만들기 1 - for문
gu_name = list()

for i in range(25):
    gu_name.append(geo_seoul['features'][i]['properties']['SIG_KOR_NM'])
    
gu_name

# 구 이름 만들기 2 - list comprehention
gu_name2 = [geo_seoul['features'][i]['properties']['SIG_KOR_NM'] for i in range(25)]
gu_name2

# x, y 판다스 데이터 프레임 만들기
def make_df(num):
    gu_name = geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_list = np.array(coordinate_list[0][0])
    
    x = coordinate_list[:, 0]
    y = coordinate_list[:, 1]
    
    return pd.DataFrame({'gu_name' : gu_name, 'x' : x, 'y' : y})

result = pd.DataFrame({})
for i in range(25):
    result = pd.concat([result, make_df(i)], ignore_index = True)

result

sns.scatterplot(data = result, x = 'x', y = 'y', hue = 'gu_name', s = 5, legend = False, palette = 'viridis')
plt.show()
plt.clf()

gangnam_df = result.assign(is_gangnam = np.where(result['gu_name'] == '강남구', 'Y', 'N'))
gangnam_df

sns.scatterplot(data = gangnam_df, x = 'x', y = 'y',
                hue = 'is_gangnam',
                s = 5,
                legend = False,
                palette = ['grey', 'red'])
plt.show()
plt.clf()

# ==============================================================================
geo_seoul['features'][0]['properties']
df_pop = pd.read_csv('data/Population_SIG.csv')
df_seoulpop = df_pop.iloc[1:26]

df_seoulpop['code'] = df_seoulpop['code'].astype(str)
df_seoulpop.info()

map_sig = folium.Map(location = [37.55, 126.97],
                     zoom_start = 12,
                     tiles = 'cartodbpositron')

bins = list(df_seoulpop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ('code', 'pop'),
    bins = bins,
    fill_color = 'viridis',
    fill_opacity = 0.5,
    line_opacity = 0.5,
    key_on = 'feature.properties.SIG_CD') \
    .add_to(map_sig)

folium.Marker([37.583744, 126.983800], popup = '종로구').add_to(map_sig)
map_sig.save('map_seoul.html')

# ==============================================================================
df_house = pd.read_csv('data/kaggle/houseprice/houseprice-with-lonlat.csv')
house_location = df_house[['Longitude', 'Latitude']]
house_location_list = house_location.values.tolist()
house_x = df_house['Longitude'].mean()
house_y = df_house['Latitude'].mean()

map_sig2 = folium.Map(location = [42.03448223395904, -93.64289689856655],
                     zoom_start = 13,
                     tiles = 'cartodbpositron')

for i in range(len(house_location)):
    x_point = house_location[["Latitude"]].iloc[i,0]
    x_float = float(x_point)
    y_point = house_location[["Longitude"]].iloc[i,0]
    y_float = float(y_point)
    folium.Marker([x_float, y_float]).add_to(map_sig2)

# folium.Marker([37.583744, 126.983800], popup = '종로구').add_to(map_sig)
map_sig2.save('map_house.html')
