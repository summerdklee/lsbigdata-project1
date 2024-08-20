import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium import LinearColormap

# Initialize the map
map_ames = folium.Map(location=[42.030806089293755, -93.6304970070205],
                     zoom_start=13,
                     tiles='cartodbpositron')

# Load the data
df = pd.read_csv('../data/kaggle/houseprice/houseprice-with-lonlat.csv')
ames_loc = df[['Latitude', 'Longitude', 'Neighborhood', 'Sale_Price']]
ames_sch_hos = pd.read_csv('../data/kaggle/houseprice/aims_school_hospital.csv')

# Define color scale based on neighborhood
neighborhoods = ames_loc['Neighborhood'].unique()

# Generate a list of random colors for each neighborhood
color_dict = {}
for neighborhood in neighborhoods:
    color_dict[neighborhood] = '#{:06x}'.format(random.randint(0, 0xFFFFFF))

# Define marker size range based on price
min_price = ames_loc['Sale_Price'].min()
max_price = ames_loc['Sale_Price'].max()
min_size = 1
max_size = 15

# Add CircleMarkers for housing data
for _, row in ames_loc.iterrows():
    price = row['Sale_Price']
    lat = row['Latitude']
    lon = row['Longitude']
    
    # Calculate marker size and color
    size = min_size + (price - min_price) / (max_price - min_price) * (max_size - min_size)
    color = color_dict[row['Neighborhood']]
    
    # Create tooltip with neighborhood name and price
    tooltip_text = f"{row['Neighborhood']}<br>Sale Price: ${price:,.0f}"
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        tooltip=tooltip_text
    ).add_to(map_ames)

# Add markers for schools and hospitals with custom icons
for _, row in ames_sch_hos.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    name = row['Name']
    type_ = row['type']
    
    # Define icon and marker color based on type
    if type_ == 'medical':
        icon = folium.Icon(icon='plus', color='blue')  # 병원 아이콘
    elif type_ == 'school':
        icon = folium.Icon(icon='book', color='orange')  # 학교 아이콘
    
    # Create tooltip with name and type
    tooltip_text = f"{name}<br>Type: {type_}"
    
    folium.Marker(
        location=[lat, lon],
        popup=tooltip_text,
        icon=icon
    ).add_to(map_ames)

# Save the map
map_ames.save('map_ames_total.html')

# ===========

df['Overall_Cond'].unique()

# 품질 순위 : 'Very_Poor','Poor','Fair','Below_Average','Average','Above_Average','Good','Very_Good','Excellent'
rank=['Very_Poor','Poor','Fair','Below_Average','Average','Above_Average','Good','Very_Good','Excellent']
for i in range(len(rank)):
    df.loc[df['Overall_Cond']==rank[i],'Overall_Cond']=i+1

# ================

import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium import LinearColormap

# Initialize the map
map_ames = folium.Map(location=[42.030806089293755, -93.6304970070205],
                     zoom_start=13,
                     tiles='cartodbpositron')

# Load the data
df = pd.read_csv('../data/kaggle/houseprice/houseprice-with-lonlat.csv')
ames_loc = df[['Latitude', 'Longitude', 'Neighborhood', 'Sale_Price']]
ames_sch_hos = pd.read_csv('../data/kaggle/houseprice/aims_school_hospital.csv')

# Define color scale based on neighborhood
# Assign specific colors for the specified neighborhoods
color_dict = {
    'Northpark_Villa': '#FF1493',  # 진한 핑크
    'North_Ames': '#8A2BE2',       # 주황색
    'Briardale': '#1E90FF',        # 보라색
    'Edwards': '#32CD32',          # 라임 그린
    'College_Creek': '#FFD700'     # 금색
}

# Default color for other neighborhoods
default_color = '#808080'  # 회색

# Define marker size range based on price
min_price = ames_loc['Sale_Price'].min()
max_price = ames_loc['Sale_Price'].max()
min_size = 1
max_size = 15

# Add CircleMarkers for housing data
for _, row in ames_loc.iterrows():
    price = row['Sale_Price']
    lat = row['Latitude']
    lon = row['Longitude']
    
    # Calculate marker size and color
    size = min_size + (price - min_price) / (max_price - min_price) * (max_size - min_size)
    color = color_dict.get(row['Neighborhood'], default_color)
    
    # Create tooltip with neighborhood name and price
    tooltip_text = f"{row['Neighborhood']}<br>Sale Price: ${price:,.0f}"
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        tooltip=tooltip_text
    ).add_to(map_ames)

# Add markers for schools and hospitals with custom icons
for _, row in ames_sch_hos.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    name = row['Name']
    type_ = row['type']
    
    # Define icon and marker color based on type
    if type_ == 'medical':
        icon = folium.Icon(icon='plus', color='blue')  # 병원 아이콘
    elif type_ == 'school':
        icon = folium.Icon(icon='book', color='orange')  # 학교 아이콘
    
    # Create tooltip with name and type
    tooltip_text = f"{name}<br>Type: {type_}"
    
    folium.Marker(
        location=[lat, lon],
        popup=tooltip_text,
        icon=icon
    ).add_to(map_ames)

# Save the map
map_ames.save('map_ames_total.html')

# ===========

import numpy as np
import pandas as pd
import folium

# Load the data
ames_sch_hos = pd.read_csv('../data/kaggle/houseprice/aims_school_hospital.csv')

# Initialize the map
map_ames = folium.Map(location=[42.030806089293755, -93.6304970070205],
                     zoom_start=13,
                     tiles='cartodbpositron')

# Define the new data
new_data = pd.DataFrame({
    'Id': [669, 1047, 403, 2737, 1425],
    'Sale_Price': [200000, 143000, 125000, 415000, 475000],
    'Longitude': [-93.610649, -93.625986, -93.628119, -93.660664, -93.686980],
    'Latitude': [42.041240, 42.050680, 42.052338, 42.028191, 42.027368],
    'Total_Sum': [81, 52, 45, 89, 79]
})

# Define color scale based on neighborhood
color_dict = {
    'Northpark_Villa': 'cadetblue',
    'North_Ames': 'purple',
    'Briardale': 'darkred',
    'Edwards': 'lightgray',
    'College_Creek': 'darkgreen'
}

# 각 Neighborhood에 해당하는 이미지 URL을 딕셔너리로 매핑
image_dict = {
    'Northpark_Villa': 'NorParkVil.png',
    'North_Ames': 'NorAmes.png',
    'Briardale': 'Briadale.png',
    'Edwards': 'Edwards.png',
    'College_Creek': 'ColCreek.png',
}

# Add regular Markers for selected houses
for _, row in new_data.iterrows():
    price = row['Sale_Price']
    lat = row['Latitude']
    lon = row['Longitude']
    score = row['Total_Sum']  # 총합 점수로 'Total_Sum' 사용
    
    # Get the image URL from the dictionary based on Neighborhood (assuming it's correctly mapped)
    neighborhood = list(color_dict.keys())[_ % len(color_dict)]  # 임의로 매핑된 동네 이름 사용
    image_url = image_dict.get(neighborhood, 'https://www.example.com/default.jpg')

    # Create the HTML for the popup, including the image
    popup_html = f"""
    <div style="width:200px">
        <h4>{neighborhood}</h4>
        <img src="{image_url}" width="180px">
        <p>Sale Price: ${price:,.0f}</p>
        <p>Score: {score}</p>
    </div>
    """

    # Get the marker color from the color_dict
    marker_color = color_dict.get(neighborhood, 'blue')

    # Create the Marker with the popup and tooltip
    folium.Marker(
        location=[lat, lon],
        tooltip=f"{neighborhood}<br>Sale Price: ${price:,.0f}",
        popup=folium.Popup(popup_html, max_width=250),
        icon=folium.Icon(color=marker_color, icon='home')
    ).add_to(map_ames)

# Add markers for schools and hospitals with custom icons
for _, row in ames_sch_hos.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    name = row['Name']
    type_ = row['type']
    
    # Define icon and marker color based on type
    if type_ == 'medical':
        icon = folium.Icon(icon='plus', color='blue')  # 병원 아이콘
    elif type_ == 'school':
        icon = folium.Icon(icon='book', color='orange')  # 학교 아이콘
    
    # Create tooltip with name and type
    tooltip_text = f"{name}<br>Type: {type_}"
    
    folium.Marker(
        location=[lat, lon],
        popup=tooltip_text,
        icon=icon
    ).add_to(map_ames)

# Save the map
map_ames.save('map_ames_total2.html')

