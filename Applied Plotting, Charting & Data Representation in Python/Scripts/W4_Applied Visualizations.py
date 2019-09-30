##################################################################
################### Applied Visualizations #######################
##################################################################

import os
path = r'''C:\Users\Juan Jose\Documents\Programacion\Python Data Science\Applied Plotting, Charting & Data Representation in Python\Data Sets'''
os.chdir(path)

import matplotlib as mpl
# Import pyplot. This is a procedural language for making graphics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
'''
################ Plotting with Pandas ################
# Really good for fast and easy plottinf of Series and Data Frames
# Predefine styles provided:
#print(plt.style.available)
plt.style.use('seaborn-colorblind')

### pandas DataFrame.plot ###
np.random.seed(123)
df = pd.DataFrame({'A': np.random.randn(365).cumsum(0),
'B': np.random.randn(365).cumsum(0)+ 20,
'C': np.random.randn(365).cumsum(0)-20},
index = pd.date_range('1/1/2018', periods = 365))

print(df.head().to_string())
# Let's see how it looks visually
df.plot()

# Choose which columns to plot and what type pf plot
df.plot('A','B',kind = 'scatter')

# Also use .scatter option. Color 'c' and size 's' will vary base on B
ax = df.plot.scatter('A','C',c ='B', s = df['B'], colormap = 'viridis')

# This also returns and axis
ax.set_aspect('equal')

# Graph boxplots
df.plot.box()

# Histograms
df.plot.hist(alpha=0.7)

# Kernell density plots. Derive a smooth continous function from a given sample
df.plot.kde()

### pandas.tools.plotting ###
iris = pd.read_csv('iris.csv')
#print(iris.head().to_string())

# Create a Scatter matrix (each column vs each column):
pd.tools.plotting.scatter_matrix(iris)

# Tool for creating parallel coordinates plots  for visualize high dimension multivariate data
plt.figure()
pd.tools.plotting.parallel_coordinates(iris,'Name')
plt.show()

################ Seaborn ################
np.random.seed(1234)
v1 = pd.Series(np.random.normal(0,10,1000),name = 'v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name = 'v2')

plt.figure()
plt.hist(v1,alpha=0.5,bins=np.arange(-50,150,5),label='v1')
plt.hist(v2,alpha=0.5,bins=np.arange(-50,150,5),label='v2')
plt.legend()

# Let's visualize in another way
plt.figure()
plt.hist([v1,v2],histtype = 'barstacked', normed = True)
# Create a combination of both
v3 = np.concatenate((v1,v2))
sns.kdeplot(v3)

# Seaborn provides a function for this types of plots
plt.figure()
sns.distplot(v3, hist_kws = {'color':'Teal'}, kde_kws={'color': 'Navy'})

# Seaborn jointplot
plt.figure()
sns.jointplot(v1,v2,alpha=0.5)

# Some functions return matplotlib objects
grid = sns.jointplot(v1,v2,alpha=0.5)
grid.ax_joint.set_aspect('equal')

# Hexan plots show the number of observation in hexagnal bins (works well with relaive large data sets)
sns.jointplot(v1,v2,kind='hex')

# Usefull for kernel density estimation in a 2-D environment
sns.set_style('white')
sns.jointplot(v1,v2,kind='kde', space=0) # space between the joint and marginal axes (default = 0.2)

# Load iris to see categorical data grpahs
iris = pd.read_csv('iris.csv')

# Seaborn scatter plott matrix
sns.pairplot(iris, hue = 'Name', diag_kind='kde') #diag_kind='kde' instead of histograms

# Violin plot: informative version of a boxplot
# Plot a violent plot next to a swarm plot(scatter plott for categorical data)
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.swarmplot('Name', 'PetalLength',data = iris)
plt.subplot(1,2,2)
sns.violinplot('Name', 'PetalLength',data = iris)

plt.show()
'''
################# Assignment 4
