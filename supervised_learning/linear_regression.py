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

