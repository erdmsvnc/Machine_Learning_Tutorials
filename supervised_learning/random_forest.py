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
