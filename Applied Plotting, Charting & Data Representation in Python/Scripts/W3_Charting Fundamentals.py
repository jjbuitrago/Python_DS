##################################################################
################### Charting Fundamentals ########################
##################################################################

import os
path = r'''C:\Users\Juan Jose\Documents\Programacion\Python Data Science\Applied Plotting, Charting & Data Representation in Python\Data Sets'''
os.chdir(path)
import matplotlib as mpl
# Import pyplot. This is a procedural language for making graphics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import mplleaflet
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
import matplotlib.animation as animation
from random import shuffle

########### Subplots ###########
lin_data = np.arange(8)
exp_data = lin_data**2
'''
# plt.subplot(nrows,ncole,plot_number)
plt.figure()
# First subplot
plt.subplot(1,2,1)
plt.plot(lin_data,'-o')

# Second subplot

plt.subplot(1,2,2)
plt.plot(exp_data,'-o')

# To get the same axis use share=ax1
plt.figure()
ax1 = plt.subplot(1,2,1)
plt.plot(lin_data,'-o')

ax2 = plt.subplot(1,2,2,sharey = ax1)
plt.plot(exp_data,'-x')

# Create a grid of sublots
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,sharex=True,sharey=True)
ax5.plot(lin_data,'-o')

# To turn the labels back on, iterate through the axis objects and put them back on
for ax in plt.gcf().get_axes():
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(True)

plt.show()

############ Histograms #############
# Sample normal distribution
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex=True)
axs = [ax1,ax2,ax3,ax4]

for n in range(0,len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0.0, scale = 1.0, size = sample_size)
    axs[n].hist(sample, bins=100)
    axs[n].set_title('n={}'.format(sample_size))
plt.show()

# Use gridspec to custommize location of a plot
plt.figure()
gspec = gridspec.GridSpec(3,3)
top_histogram = plt.subplot(gspec[0,1:])
side_histogram = plt.subplot(gspec[1:,0])
lower_right = plt.subplot(gspec[1:,1:])

Y = np.random.normal(loc=0.0, scale = 1.0, size = 10000)
X = np.random.random(size = 10000)
lower_right.scatter(X,Y)
top_histogram.hist(X,bins=100, normed = True)
s = side_histogram.hist(Y,bins=100,orientation='horizontal',normed=True)

#To invert axis
side_histogram.invert_xaxis()

# Change axis limits
for ax in [top_histogram,lower_right]:
    ax.set_xlim(0,1)
for ax in [side_histogram,lower_right]:
    ax.set_ylim(-5,5)

plt.show()

############ Box Plot #############
normal_sample = np.random.normal(loc=0.0,scale=1.0, size = 10000)
random_sample = np.random.random(size=10000)
gamma_sample = np.random.gamma(2,size=10000)

# Create a Data Frame
df = pd.DataFrame({'normal':normal_sample,'random': random_sample,'gamma':gamma_sample})

# Describe to see some summary statistics
#print(df.describe())

#box plot:
plt.figure()
plt.boxplot([df['normal'],df['random'],df['gamma']],whis='range')

# Inset axes:
ax2 = mpl_il.inset_axes(plt.gca(), width='60%',height='40%',loc=2)
ax2.hist(df['gamma'],bins=100)
ax2.margins(x=0.5)
ax2.yaxis.tick_right()
plt.show()

########## Heatmaps: 2-D histograms ############
plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)

plt.hist2d(X,Y,bins=100)
plt.colorbar()

plt.show()

########## Animation ##########
n = 100
x = np.random.randn(n)

# Puts a frame over a frame until curr =n
def update(curr):
    if curr ==n:
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4,4,0.5)
    plt.hist(x[:curr],bins=bins)
    plt.axis([-4,4,0,30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr),[3,27])

fig = plt.figure()
a = animation.FuncAnimation(fig,update,interval=100) #interval ir miliseconds between frame
plt.show()

############ Practice Assignment ############

# generate 4 random variables from the random, gamma, exponential, and uniform distributions
x1 = np.random.normal(-2.5, 1, 10000)
x2 = np.random.gamma(2, 1.5, 10000)
x3 = np.random.exponential(2, 10000)+7
x4 = np.random.uniform(14,20, 10000)

# Puts a frame over a frame until curr =n
def update(curr):

    if curr ==n:
        a.event_source.stop()
    for i in range(len(ax)):
        ax[i].cla()
        ax[i].hist(x[i][:curr], normed = True, bins = bins[i], alpha=0.25)
        ax[i].set_title(title[i])
        ax[i].set_ylabel('Frequency')
        ax[i].set_xlabel('Value')
    plt.tight_layout()

fig = plt.figure()

gspec = gridspec.GridSpec(2,2)
normal_hist = plt.subplot(gspec[0,0])
gamma_hist = plt.subplot(gspec[0,1])
exp_hist = plt.subplot(gspec[1,0])
uni_hist = plt.subplot(gspec[1,1])

########## Animation ##########
n = 3000
#plt.figure()

bins1 = np.arange(-7.5, 2.5, 0.2)
bins2 = np.arange(0, 10, 0.2)
bins3 = np.arange(7, 17, 0.2)
bins4 = np.arange(12, 22, 0.2)
bins = [bins1, bins2, bins3, bins4]

ax = [normal_hist,gamma_hist,exp_hist,uni_hist]
x=[x1,x2,x3,x4]
title = ['x1 Normal', 'x2 Gamma', 'x3 Exponential', 'x4 Uniform']
print(title[1])

a = animation.FuncAnimation(fig,update,interval=10)

plt.figure()
gspec_1 = gridspec.GridSpec(2,2)
normal_hist = plt.subplot(gspec_1[0,0])
gamma_hist = plt.subplot(gspec_1[0,1])
exp_hist = plt.subplot(gspec_1[1,0])
uni_hist = plt.subplot(gspec_1[1,1])
normal_hist.hist(x1,bins=bins[0],normed=True, alpha=0.25)
gamma_hist.hist(x2, normed=True, bins=bins[1], alpha=0.5)
exp_hist.hist(x3, normed=True, bins=bins[2], alpha=0.5)
uni_hist.hist(x4, normed=True, bins=bins[3], alpha=0.5)

plt.show()
'''
############## Assignment 3: Building a Custom Visualization ###############
np.random.seed(12345)
from matplotlib import cm
df = pd.DataFrame([np.random.normal(32000,200000,3650),
                   np.random.normal(43000,100000,3650),
                   np.random.normal(43500,140000,3650),
                   np.random.normal(48000,70000,3650)],
                  index=[1992,1993,1994,1995])

#print(np.std(df.iloc[0])*1.96)

# Get the mean of each distribution
y = (df.mean(axis = 1)) - 36000 + 3515.919393
print(df.mean(axis = 1))
plt.figure()
# Define colors with option brg
colors = cm.brg(y / float(max(y)))
# PLot the means and then close then to get colorbar
plot = plt.scatter(y, y, c = y, cmap = 'brg')
plt.clf()
plt.colorbar(plot)
# Plot hitogram
plt.bar(list(df.index), list(df.mean(axis = 1)), width = 0.5, yerr = ([150000, 90000, 120000, 55000]/np.sqrt(3650)*1.96), color = colors);
plt.xticks(list(df.index), list(df.index))
plt.axhline(y=36000, color='black', linestyle='-')
#plt.yticks(list(plt.yticks()[0]) + [36000]);

plt.show()
