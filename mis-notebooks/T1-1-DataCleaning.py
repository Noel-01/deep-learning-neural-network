# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:13:25 2022

@author: Noel
"""

import pandas as pd

mainpath = "D:/CursosExternos/ML-DataScience/datasets/"
filename = "titanic/titanic3.csv"
fullpath = mainpath + filename

data = pd.read_csv(fullpath)

print(data.head())

# works with a text file as well
data2 = pd.read_csv(mainpath + "customer-churn-model/Customer Churn Model.txt")
print(data2.head())
print(data2.columns.values)


# function open
data3 = open(mainpath + "customer-churn-model/Customer Churn Model.txt", 'r')

cols = data3.readline().strip().split(",")
main_dict = {}

for col in cols:
    main_dict[col] = []
    
print(main_dict)


n_cols = len(cols)
counter = 0

for line in data3:
    values = line.strip().split(",")
    for i in range(n_cols):
        main_dict[cols[i]].append(values[i])
    counter +=1
        
print("El data set tiene %d filas y %d columnas"%(counter, n_cols))
        
df3 = pd.DataFrame(main_dict)
print(df3.head())


#LECTURA DESDE UNA URL

medals_url = "http://winterolympicsmedals.com/medals.csv"
medals_data = pd.read_csv(medals_url)

print(medals_data.head())


# dibuja el grafico de la media de dos variables
filename2 = "customer-churn-model/Customer Churn Model.txt"
data4 = pd.read_csv(mainpath + filename2)
data4.plot(kind="scatter", x="Day Mins", y="Day Charge")








