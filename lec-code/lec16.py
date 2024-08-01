! pip install gspread

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 구글 스프레드시트 판다스로 불러오기
gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2"
df = pd.read_csv(gsheet_url)
df.head()

# 랜덤 2명 추출
np.random.seed(20240730)
np.random.choice(df['이름'], 2, replace = False)

np.random.seed(20240801)
np.random.choice(np.arange(7) + 1, 7, replace = False)
np.random.choice(np.arange(4) + 1, 1, replace = False)

