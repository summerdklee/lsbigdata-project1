import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


economics = pd.read_csv('data/economics.csv')
economics.head()
economics.info()

sns.lineplot(data = economics, x = 'date', y = 'unemploy')
plt.show()
plt.clf()

economics['date2'] = pd.to_datetime(economics['date'])
economics.head()
economics.info()

economics[['date', 'date2']]
economics['date2'].dt.year
economics['date2'].dt.month
economics['date2'].dt.day
economics['date2'].dt.month_name()
economics['date2'].dt.quarter

economics['quarter'] = economics['date2'].dt.quarter
economics[['date2', 'quarter']]

economics['date2'].dt.day_name()

economics['date2'] + pd.DateOffset(days = 30)
economics['date2'] + pd.DateOffset(months = 1)

economics['year'] = economics['date2'].dt.year
sns.lineplot(data = economics, x = 'year', y = 'unemploy', errorbar = None)
plt.show()


my_df = economics.groupby('year', as_index = False) \
                 .agg(mon_mean = ('unemploy', 'mean'),
                      mon_std = ('unemploy', 'std'),
                      mon_n = ('unemploy', 'count'))

my_df['left_ci'] = my_df['mon_mean'] - 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df['right_ci'] = my_df['mon_mean'] + 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df.head()

x = my_df['year']
y = my_df['mon_mean']

plt.plot(x, y, color = 'black')
plt.scatter(x, my_df['left_ci'], color = 'red', s = 5)
plt.scatter(x, my_df['right_ci'], color = 'green', s = 5)
plt.show()
plt.clf()


