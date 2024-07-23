# 08. 그래프 만들기

## 산점도
import pandas as pd

mpg = pd.read_csv('data/mpg.csv')

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy')
plt.show()
plt.clf()

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy') \
   .set(xlim = [3, 6])
plt.show()
plt.clf()

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy') \
   .set(xlim = [3, 6], ylim = [10, 30])
plt.show()
plt.clf()

# plt.figure(figsize = (5, 4)) > 그래프 사이즈 조정

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy', hue = 'drv')
plt.show()
plt.clf()


## 막대그래프
# mpg['drv'].unique() > 유니크한 요소만 보기 

df_mpg = mpg.groupby('drv', as_index = False) \
   .agg(mean_hwy = ('hwy', 'mean'))
df_mpg

sns.barplot(data = df_mpg, x = 'drv', y = 'mean_hwy', hue = 'drv')
plt.show()
plt.clf()

df_mpg = df_mpg.sort_values('mean_hwy', ascending = False)
sns.barplot(data = df_mpg, x = 'drv', y = 'mean_hwy', hue = 'drv')
plt.show()
plf.clf()

df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv', 'count'))
sns.barplot(data = df_mpg, x = 'drv', y = 'n', hue = 'drv')
plt.show()
plt.clf()

sns.countplot(data = mpg, x = 'drv', hue = 'drv')
plt.show()
plt.clf()
