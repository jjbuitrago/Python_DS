########################################################
########### Fundamentals of Machine Learning ###########
########################################################

import os
path = r'''C:\Users\Juan Jose\Documents\Programacion\Python Data Science\Applied Machine Learning in Python\Data Base'''
os.chdir(path)

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
# This is color map
from matplotlib import cm
# This is to split the data in test and train set
from sklearn.model_selection import train_test_split
# This is to plot in 3-D
from mpl_toolkits.mplot3d import Axes3D
# To perform K-Neighboors classifier
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('fruit_data_with_colors.txt')
# print(fruits.head().to_string())
# print(fruits.shape)

# Split the data in train set & test set
'''X = fruits[['mass','width','height','color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0) #random_state is like the 'seed'

## Examining the Data
# Scatter Matrix
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

# 3-D feature scatterplot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()
'''
## K-nearest neighboors. Work by memorizing the examples and classify new examples
# For this example, only use 'mass','width','height'
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
# print(lookup_fruit_name)

X = fruits[['mass','width','height']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0) #random_state is like the 'seed'

# Create an instance of a clas object
knn = KNeighborsClassifier(n_neighbors = 5)

# Train the classifier
knn.fit(X_train,y_train)

# See how accurate it is applying it to the test set
print('Algorith accuracy:',knn.score(X_test,y_test))

# Classify inividual instances of fruits by entering hypothetical data
fruit_prediction = knn.predict([[20,4.3,5.5]])
print('It is a',lookup_fruit_name[fruit_prediction[0]])

from adspy_shared_utilities import plot_fruit_knn
#plot_fruit_knn(X_train, y_train, 10, 'uniform')

# To see how sensitive accuracy is to 'k'
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);
plt.show()









print()
