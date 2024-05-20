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


