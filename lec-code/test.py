import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 불러오기
train_df = pd.read_csv('../data/kaggle/houseprice/houseprice-with-lonlat.csv')
train_df = train_df.rename(columns = {'Unnamed: 0' : 'Id'})
train_df['Id'] = train_df['Id']-1

loc = train_df.copy()

# Gr_Liv_Area를 수치화한 Live_Score 만들기
bins=[0, round(146.5,0), round(146.5*2,0), round(146.5*3,0), round(146.5*4,0),round(146.5*5,0),round(146.5*6,0),round(146.5*7,0),round(146.5*8,0),round(146.5*9,0),round(146.5*10,0),round(146.5*11,0),round(146.5*12,0),round(146.5*13,0),round(146.5*14,0),round(146.5*15,0),round(146.5*16,0),round(146.5*17,0),round(146.5*18,0),round(146.5*19,0)]
bins = list(map(int,bins))
loc = loc.sort_values('Gr_Liv_Area').reset_index(drop=True)
for i in range(len(bins)):
    if i != 19:
        loc.loc[bins[i]:bins[i+1]-1 , 'Live_Score'] = i+1
    if i == 19 :
        loc.loc[bins[i]: , 'Live_Score'] = i+1

loc = loc.sort_values('Id').reset_index(drop=True)


#| title: 전체적인 상태 등급 (Overall_Cond)

# Overall_Cond을 수치화한 Overall_Score 만들기
# 품질 순위 : 'Very_Poor','Poor','Fair','Below_Average','Average','Above_Average','Good','Very_Good','Excellent','Very_Excellent'
rank=['Very_Poor','Poor','Fair','Below_Average','Average','Above_Average','Good','Very_Good','Excellent','Very_Excellent']
for i in range(len(rank)):
    loc.loc[loc['Overall_Cond']==rank[i],'Overall_Score']=math.floor((i+1)*2)


#| title: 지상 총 침실 수 (Bedroom_AbvGr)

# Bedroom_AbvGr을 수치화한 Bedroom_Score 만들기   (오류 메시지가 뜨긴 하지만, 문제 없음)
room=loc['Bedroom_AbvGr'].unique()
for i in range(len(room)):
    if room[i] == 8:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=20
    elif room[i] == 7:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=18
    elif room[i] == 6:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=16
    elif room[i] == 5:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=14
    elif room[i] == 4:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=12
    elif room[i] == 3:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=10
    elif room[i] == 2:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=7
    elif room[i] == 1:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=4
    else:
        loc.loc[loc['Bedroom_AbvGr']==room[i],'Bedroom_Score']=1


#| title: 전체 지하실 면적 (Total_Bsmt_SF)

# Total_Bsmt_SF를 수치화한 Base_Score 만들기
loc = loc.sort_values('Total_Bsmt_SF').reset_index(drop=True)
for i in range(len(bins)):
    if i != 19:
        loc.loc[bins[i]:bins[i+1]-1 , 'Base_Score'] = i+1
    if i == 19 :
        loc.loc[bins[i]: , 'Base_Score'] = i+1
loc = loc.sort_values('Id').reset_index(drop=True)


#| title: 주차장 면적 (Garage_Area)

# Garage_Area를 수치화한 Garage_Score 만들기
loc = loc.sort_values('Garage_Area').reset_index(drop=True)
for i in range(len(bins)):
    if i != 19:
        loc.loc[bins[i]:bins[i+1]-1 , 'Garage_Score'] = i+1
    if i == 19 :
        loc.loc[bins[i]: , 'Garage_Score'] = i+1

loc = loc.sort_values('Id').reset_index(drop=True)


# 타입 int로 바꾸기
loc['Overall_Score'] = loc['Overall_Score'].astype(int)
loc['Bedroom_Score'] = loc['Bedroom_Score'].astype(int)
loc['Live_Score'] = loc['Live_Score'].astype(int)
loc['Base_Score'] = loc['Base_Score'].astype(int)
loc['Garage_Score'] = loc['Garage_Score'].astype(int)

# 총합 Total_Sum 변수 만들기
loc['Total_Sum'] = loc['Overall_Score']+loc['Bedroom_Score']+loc['Live_Score']+loc['Base_Score']+loc['Garage_Score']

loc1 = loc[loc['Neighborhood']=='North_Ames']
loc2 = loc[loc['Neighborhood']=='Northpark_Villa']
loc3 = loc[loc['Neighborhood']=='Briardale']
loc4 = loc[loc['Neighborhood']=='Edwards']
loc5 = loc[loc['Neighborhood']=='College_Creek']

northpark_villa_top5 = loc1.sort_values('Total_Sum', ascending = False).head(1)
north_ames_top5 = loc2.sort_values('Total_Sum', ascending = False).head(1)
briardale_top5 = loc3.sort_values('Total_Sum', ascending = False).head(1)
edwards_top5 = loc4.sort_values('Total_Sum', ascending = False).head(1)
college_creek_top5 = loc5.sort_values('Total_Sum', ascending = False).head(1)

a = northpark_villa_top5[['Id', 'Sale_Price', 'Longitude', 'Latitude', 'Total_Sum']]
b=north_ames_top5[['Id', 'Sale_Price', 'Longitude', 'Latitude', 'Total_Sum']]
c=briardale_top5[['Id', 'Sale_Price', 'Longitude', 'Latitude', 'Total_Sum']]
d=edwards_top5[['Id', 'Sale_Price', 'Longitude', 'Latitude', 'Total_Sum']]
e=college_creek_top5[['Id', 'Sale_Price', 'Longitude', 'Latitude', 'Total_Sum']]

df = pd.concat([a, b, c, d, e])

