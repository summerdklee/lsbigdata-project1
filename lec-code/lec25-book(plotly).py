import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins

# subplot
penguins = load_penguins()
penguins.head()

fig = px.scatter(penguins,
                 x = 'bill_length_mm',
                 y = 'bill_depth_mm',
                 color = 'species',
                 # trendline = 'ols' 134p
                 ) 
fig.show()

# subplot 관련 패키지 불러오기
from plotly.subplots import make_subplots

fig_subplot = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Adelie', 'Gentoo', 'Chinstrap')
)

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Adelie"')['bill_length_mm'],
   'y' : penguins.query('species=="Adelie"')['bill_depth_mm'],
   'name' : 'Adelie'
  },
  row=1, col=1
 )

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Gentoo"')['bill_length_mm'],
   'y' : penguins.query('species=="Gentoo"')['bill_depth_mm'],
   'name' : 'Gentoo'
  },
  row=1, col=2
 )

fig_subplot.add_trace(
  {
   'type' : 'scatter',
   'mode' : 'markers',
   'x' : penguins.query('species=="Chinstrap"')['bill_length_mm'],
   'y' : penguins.query('species=="Chinstrap"')['bill_depth_mm'],
   'name' : 'Chinstrap'
  },
  row=1, col=3
 )

fig_subplot.update_layout(
    title=dict(text='펭귄 종별 부리 길이 vs. 깊이',
               x=0.5)
)


