# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:04:59 2018

@author: user
"""

from sklearn.datasets import load_iris
import pandas as pd

# %%
df = pd.read_csv("One_year_compiled.csv")

df.rename(columns={"pCut::Motor_Torque" : "pCut_Motor_Torque"},inplace=True)
df.rename(columns={"pCut::CTRL_Position_controller::Lag_error" : "pCut_CTRL_Position_controller_Lag_error"},inplace=True)
df.rename(columns={"pCut::CTRL_Position_controller::Actual_position" : "pCut_CTRL_Position_controller_Actual_position"},inplace=True)
df.rename(columns={"pCut::CTRL_Position_controller::Actual_speed" : "pCut_CTRL_Position_controller_Actual_speed"},inplace=True)
df.rename(columns={"pSvolFilm::CTRL_Position_controller::Actual_position" : "pSvolFilm_CTRL_Position_controller_Actual_position"},inplace=True)
df.rename(columns={"pSvolFilm::CTRL_Position_controller::Actual_speed" : "pSvolFilm_CTRL_Position_controller_Actual_speed"},inplace=True)
df.rename(columns={"pSvolFilm::CTRL_Position_controller::Lag_error" : "pSvolFilm_CTRL_Position_controller_Lag_error"},inplace=True)
df.rename(columns={"pSpintor::VAX_speed" : "pSpintor_VAX_speed"},inplace=True)

data = df.pCut_CTRL_Position_controller_Lag_error
pSpintor_VAX_speed = df.pSpintor_VAX_speed
y = df.data = df.pCut_Motor_Torque


df2 = pd.DataFrame(data,columns = pSpintor_VAX_speed)
df2["sinif"] = y

x = data

#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, whiten= True )  # whitten = normalize
pca.fit(x)

x_pca = pca.transform(x)

print("variance ratio: ", pca.explained_variance_ratio_)

print("sum: ",sum(pca.explained_variance_ratio_))

#%% 2D

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df2.p1[df2.sinif == each],df2.p2[df2.sinif == each],color = color[each],label = df.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
























