# Supervised_Regression_Models_Tutorials


1-) Linear Regression
---
```
#import data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("linear_regression_dataset.csv", sep=";")
print(df)

#plot data
"""
plt.scatter(df.deneyim, df.maas)
plt.show()
"""
#linear regression

#sklearn library

from sklearn.linear_model import LinearRegression

#linear regression model

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)

y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#prediction

b0 = linear_reg.predict([[0]]) #y eksenini kestiği nokta

b0_ = linear_reg.intercept_

b1 = linear_reg.coef_ #b1 eğim

#maas = 1663 + 1138 * deneyim

maas_yeni  = 1663 + 1138 *11
print(maas_yeni)

print(linear_reg.predict([[11]]))

array  = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)#Deneyimimiz #sklearn shape yaptığımızda (16,"bir değer") görmek istediği için
plt.scatter(x,y)
plt.show()
y_head = linear_reg.predict(array)#maas
plt.plot(array, y_head,color = "red")
```


2-) Multiple Linear Regression
---
```
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression


df = pd.read_csv("multiple_linear_regression_dataset.csv", sep = ";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0 : ", multiple_linear_regression.intercept_)
print("b1,b2 :", multiple_linear_regression.coef_)

#predict

multiple_linear_regression.predict(np.array([[10,35],[5,35]]))
```

3-) Polynomial Regression
---
```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_regression.csv", sep=";")

x = df.araba_max_hiz.values.reshape(-1,1) #.values arraya çevirir, .reshape(15,1) olarak gösterir
y = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.show()

#lineare regression ? y = b0 + b1*x
#multiple linear regression ? y = b0 + b1*x1  + b2*x2

#polynomial_regression ? y = b0 + b1*x + b2*x^2 + b3*x^3...+ b4*x^4

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polynomial_regression = PolynomialFeatures(degree = 2) #degree = n eger degreeye 3 yazsaydık x_polynomial kübünü alana kadar yazacakti

x_polynomial = polynomial_regression.fit_transform(x) #.fit_tranform uygula ve çevir

#fit

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial, y)

#

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2, color = "green")
plt.show()
```

4-) Decision Tree Regression
---
```
"""
 Desicion Tree : 
     CART : Classification and regression tree
"""
 
 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
df = pd.read_csv("decision_tree_regression_dataset.csv", sep = ";", header=None)


x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)


tree_reg.predict([[5.5]])
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

#visualize


plt.scatter(x,y, color = "red")
plt.plot(x_,y_head, color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
```

5-) Random Forest Regression
---
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random_forest_regression_dataset.csv", sep = ";", header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42 )#n_estimator kaç tane agac kullanicagimizi yazariz

rf.fit(x,y)

print(rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x), 0.01).reshape(-1, 1)

y_head = rf.predict(x_)

plt.scatter(x, y, color = "red")
plt.plot(x_, y_head, color="green")

plt.show()
"""
!!Degerlendirme Yapma!!
y_head = rf.predict(x)

from sklearn.metrics import r2_score

print("r_score : ", r2_score(y,y_head))
"""
```