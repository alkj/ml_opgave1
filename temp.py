# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import pandas as pd

file_path = '/home/alexander/Documents/DTU/SEMESTER 5/machine learning/opgave 1/BreastTissue.xls'
breast_data = pd.read_excel(file_path, index_col=0, sheet_name="Data")
attributeNames = np.asanyarray(breast_data.columns)
tissue_names = np.array(breast_data.Class)
breast_data = breast_data.drop(['Class'], axis=1)

data = np.array(breast_data.get_values(), dtype=np.float64)

X_c = data1[:, :].copy()
y_c = data1[:, -1].copy()
#print(data1)
X_r = data1[:, 1:].copy()
y_r = data1[:, 0].copy()

k_coding = np.zeros((106, 6))

for i in range(len(tissue_names)):
    if tissue_names[i]=='car':
        k_coding[i][0]=1
    elif tissue_names[i]=='fad':
        k_coding[i][1]=1
    elif tissue_names[i]=='mas':
        k_coding[i][2]=1
    elif tissue_names[i]=='gla':
        k_coding[i][3]=1
    elif tissue_names[i]=='con':
        k_coding[i][4]=1
    elif tissue_names[i]=='adi':
        k_coding[i][5]=1
    else:
        raise(ValueError.with_traceback)
        
#print(k_coding)

conc = np.concatenate((data, k_coding), axis=1)

print(conc)
"""

## exercise 1.5.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load file in
file_path = '/home/alexander/Documents/DTU/SEMESTER 5/machine learning/opgave 1/BreastTissue.xls'
breast_data = pd.read_excel(file_path, sheet_name = "Data", index_col=0)

# Show the attributes
attributeNames = np.array(breast_data.columns)

# Isolate the class types
tissue_names = np.array(breast_data.Class)
breast_data = breast_data.drop(['Class'], axis=1)

# Lav 1-out-of-K coding
tiss = []
[tiss.append(elem) for elem in tissue_names if elem not in tiss]

K = np.zeros((len(breast_data), len(tiss)))

for i in range(len(tissue_names)):
    K[i][tiss.index(tissue_names[i])] = 1

# Slå 1-out-of-K sammen data
data = np.concatenate((breast_data, K), axis=1)
#del K, i, tiss, tissue_names, file_path


[(plt.figure(), plt.plot(row)) for row in data.T[0:9]]

pd.plotting.autocorrelation_plot(data)

[(plt.figure(), pd.plotting.autocorrelation_plot(row)) for row in data.T[0:9]]



#plt.plot(X_c[:, 0], X_c[:, 1], 'o')
#plt.plot(X_c[:, 0], X_c[:, 2], 'o')
#show()

#plt.plot(y_r, X_r[:, 3])
#show()





#print(attributeNames)
#print(table)
#df = pd.DataFrame(data)
#pd.plotting.scatter_matrix(df)
#pd.plotting.autocorrelation_plot(df)



