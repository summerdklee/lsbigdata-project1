import pandas as pd
import numpy as np

df = pd.DataFrame({'name' : ['김지훈', '이유진', '박동현', '김민지'],
                'english' : [90, 80, 60, 70],
                'math' : [50, 60, 100, 20]})
df
type(df)

df["name"]
type(df["name"]) # Series: 열 이름이 맨 마지막에 나오고, 데이터 타입을 보여줌

sum(df["english"]) / 4

# 84p 예제
fruits = pd.DataFrame({'product' : ['사과', '딸기', '수박'],
                        'price' : [1800, 1500, 3000],
                        'selling_amount' : [24, 38, 13]})
fruits

price_avg = sum(fruits['price']) / 3
selling_amount_avg = sum(fruits['selling_amount']) / 3
price_avg
selling_amount_avg
