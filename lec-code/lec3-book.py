# 교재 3장


## 패키지

# ! pip install seaborn

import seaborn as sns # seaborn 패키지를 불러와, 이제부터 이 패키지를 sns로 부를게게
import matplotlib.pyplot as plt

var = ['a', 'a', 'b', 'c']
var

sns.countplot(x = var)
plt.show()
plt.clf() # .clf(): 앞선 데이터 삭제

df = sns.load_dataset('titanic')
sns.countplot(data = df, x = "sex", hue = "sex")
plt.show()
plt.clf()

sns.countplot(data = df, x = "class")
plt.clf()

sns.countplot(data = df, x = "class", hue = "alive")
plt.show()
plt.clf()

sns.countplot(data = df,
              x = "sex",
              hue = "survived"
              ) # 가독성을 위해 나눠서 쓰면 좋음
plt.show()

?sns.countplot # 함수의 사용법을 알려줌


## 모듈

# ! pip install scikit-learn

import sklearn.metrics # 해당 방식과 아래 두 가지 방식은 동일함
sklearn.metrics.accuracy_score()

from sklearn import metrics
metrics.accuracy_score()

from sklearn import metrics as met
met.accuracy_score()
