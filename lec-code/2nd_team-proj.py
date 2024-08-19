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

# import numpy as np
# import pandas as pd
# import random
# import plotly.graph_objects as go
# import plotly.express as px
# import folium
# from folium import LinearColormap

# # Initialize the map
# map_ames = folium.Map(location=[42.030806089293755, -93.6304970070205],
#                      zoom_start=13,
#                      tiles='cartodbpositron')

# # Load the data
# df = pd.read_csv('../data/kaggle/houseprice/houseprice-with-lonlat.csv')
# ames_loc = df[['Latitude', 'Longitude', 'Neighborhood', 'Sale_Price']]
# ames_sch_hos = pd.read_csv('../data/kaggle/houseprice/aims_school_hospital.csv')

# # Define color scale based on neighborhood
# neighborhoods = ames_loc['Neighborhood'].unique()

# # Assign colors: Red for 'College_Creek' and 'North_Ames', gray for others
# color_dict = {}
# for neighborhood in neighborhoods:
#     if neighborhood in ['College_Creek', 'North_Ames']:
#         color_dict[neighborhood] = '#FF1493'  # 빨간색
#     else:
#         color_dict[neighborhood] = '#808080'  # 회색

# # Define marker size range based on price
# min_price = ames_loc['Sale_Price'].min()
# max_price = ames_loc['Sale_Price'].max()
# min_size = 1
# max_size = 15

# # Add CircleMarkers for housing data
# for _, row in ames_loc.iterrows():
#     price = row['Sale_Price']
#     lat = row['Latitude']
#     lon = row['Longitude']
    
#     # Calculate marker size and color
#     size = min_size + (price - min_price) / (max_price - min_price) * (max_size - min_size)
#     color = color_dict[row['Neighborhood']]
    
#     # Create tooltip with neighborhood name and price
#     tooltip_text = f"{row['Neighborhood']}<br>Sale Price: ${price:,.0f}"
    
#     folium.CircleMarker(
#         location=[lat, lon],
#         radius=size,
#         color=color,
#         fill=True,
#         fill_color=color,
#         fill_opacity=0.6,
#         tooltip=tooltip_text
#     ).add_to(map_ames)

# # Add markers for schools and hospitals with custom icons
# for _, row in ames_sch_hos.iterrows():
#     lat = row['Latitude']
#     lon = row['Longitude']
#     name = row['Name']
#     type_ = row['type']
    
#     # Define icon and marker color based on type
#     if type_ == 'medical':
#         icon = folium.Icon(icon='plus', color='blue')  # 병원 아이콘
#     elif type_ == 'school':
#         icon = folium.Icon(icon='book', color='orange')  # 학교 아이콘
    
#     # Create tooltip with name and type
#     tooltip_text = f"{name}<br>Type: {type_}"
    
#     folium.Marker(
#         location=[lat, lon],
#         popup=tooltip_text,
#         icon=icon
#     ).add_to(map_ames)

# # Save the map
# map_ames.save('map_ames_total.html')

# # ===========

# df['Overall_Cond'].unique()

# # 품질 순위 : 'Very_Poor','Poor','Fair','Below_Average','Average','Above_Average','Good','Very_Good','Excellent'
# rank=['Very_Poor','Poor','Fair','Below_Average','Average','Above_Average','Good','Very_Good','Excellent']
# for i in range(len(rank)):
#     df.loc[df['Overall_Cond']==rank[i],'Overall_Cond']=i+1
