import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/kaggle/houseprice/train.csv')
house_train = df.copy()

house_train = house_train[['Id', 'GarageCars', 'SalePrice']]
house_train.head()
house_train.describe()

car_capacity = house_train.groupby('GarageCars', as_index = False) \
                          .agg(mean_price = ('SalePrice', 'mean')) \
                          .sort_values('mean_price', ascending = False)
car_capacity.head()

sns.barplot(data = car_capacity, x = 'GarageCars', y = 'mean_price', hue = 'GarageCars')
plt.show()
plt.clf()
