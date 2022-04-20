# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:37:13 2022

@author: Noel
"""

# Modelos de regresión lineal (modelos con datos simulados)

# y = a + b*x
# X : 100 valores destribuidos segun una N(1.5, 2.5)

# Ye = 5 + 1.9 * x + e
# e: estará distribuido segun una N(0, 0.8) 

import pandas as pd
import numpy as np

x = 1.5 + 2.5 * np.random.randn(100)
res = 0 + 0.8 * np.random.randn(100)

y_pred = 5 + 1.9 * x
y_act = 5 + 1.9 * x + res

x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()

data = pd.DataFrame(
    {
         "x":x_list,
         "y_actual":y_act_list,
         "y_prediccion":y_pred_list
    }    
)

print(data.head())