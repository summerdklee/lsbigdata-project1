import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/kaggle/houseprice/train.csv')
house_train = df.copy()

price_h = house_train.sort_values("SalePrice", ascending = False).head(140)
price_h = price_h.select_dtypes(include = [int, float])
price_h = price_h.iloc[:, 1:-1]
price_h = price_h.mean().reset_index()

price_l = house_train.sort_values("SalePrice", ascending = False).tail(140)
price_l = price_l.select_dtypes(include = [int, float])
price_l = price_l.iloc[:, 1:-1]
price_l = price_l.mean().reset_index()

comp_val = pd.merge(price_h, price_l, how = 'left', on = 'index')
comp_val = comp_val.rename(columns = {'0_x' : 'price_h', '0_y' : 'price_l'})
comp_val.transpose()

sns.lineplot(data = comp_val, x = 'index', y = 'price_h', color = 'black')
sns.lineplot(data = comp_val, x = 'index', y = 'price_l', color = 'red')
plt.ylim(0, 2500)
plt.xticks(rotation = 90)
plt.show()
plt.clf()
