##################################################################
############ Principles of Information Visualization #############
##################################################################

import os
path = r'''C:\Users\Juan Jose\Documents\Programacion\Python Data Science\Applied Plotting, Charting & Data Representation in Python\Data Sets'''
os.chdir(path)

import matplotlib as mpl
# Import pyplot. This is a procedural language for making graphics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import mplleaflet

######### Weeek 1: Principles of Information Visualization #########

############## Tools for Thinking about Design ##############
# Alberto Cairo wheel: There is a tradeoff between some features, e.g., lightness and density
# https://www.google.com.mx/search?q=alberto+Cairo+trade+off+circle&safe=strict&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjQ39eT94XcAhUL34MKHZK2DFUQ_AUICigB&biw=1536&bih=732#imgrc=2H5FxUk57wU6hM:

############# Graphical heuristics: Data-ink ratio (Edward Tufte) #############
# Remove non esential features from the grph to make it lighter

############# Chart Junk #############
# Artistic decorations on statistical graphs are like weeds in our data graphics.  Three types of chart junk:
# 1. Unintended optical art
# 2. The grid
# 3. Non-data creative graphics
 ###########Lie Factor and Spark Lines ###########
 # Use them for grpahs with trends
 # Lie Factor: Reall view perspective

 ######## The Truthful Art #########
 # Book about information graphic
 # 1. There are ways of displaying data that are more truthful tahn others
 #

############## Week 2: Basic Charing #################
############## Matplotlib Architecture ###############
# Has Backend Layer, Artist Layer and Scripting Layer (pyplot)

########### Basic Plotting with Matplotlib ###########
# Import pyplot. This is a procedural language for making graphics
#help(plt.plot)
'''
# Let's graph a point
plt.plot(3,2)
# Add a marker
plt.plot(3,2,".")

#Changing axis. plt.figure() cretes a  figure
plt.figure()
plt.plot(3,2,"o")
ax = plt.gca()
ax.axis([0,6,0,10]) # min 6 max for x - min & max for y

############# Scatterplots ##############
# Makes a x vs y graph. Uses numpy arrays
x = np.array([1,2,3,4,5,6,7,8])
y = x

plt.figure()
#plt.scatter(x,y)

# It's posible to change the color of each obs
colors = ['green']*(len(x)-1)
colors.append('yellow')
plt.scatter(x,y,s=100,c=colors) # s = separation

# zip function generates tuples combinations
zip_gen = zip([1,2,3,4,5],[6,7,8,9,10])
print(list(zip_gen))

# to unpack still use zip
x,y = zip(*zip([1,2,3,4,5],[6,7,8,9,10])) # inside is *zip(x,y) because are duples to unzip
print(x)
#It's also posible to slice the data and join in a single Plotting
plt.figure()
plt.scatter(x[:2],y[:2],s=100,c = "blue",label = 'Tall')
plt.scatter(x[2:],y[2:],s=100,c = "red",label = 'Short') #Labeles are useful when building leyeds

# Adding x-label, y-label and a title
plt.xlabel('The number of times the child kicked a ball')
plt.ylabel('The grade of the student')
plt.title('Relationship between ball kicking and grades')

# Setting the leyend
plt.legend(loc =4, frameon = True, title = 'Legend') #'loc' puts it in the left corner

######### Line PLots ##########3
# Connects points witha line
lin_data = np.array([1,2,3,4,5,6,7,8,])
quad_data = lin_data**2
plt.figure()
plt.plot(lin_data,'-o',quad_data,'-o')

# Paint the space between both Line
plt.gca().fill_between(range(len(lin_data)), # alpha is a transparency factor
lin_data,quad_data,facecolor = 'blue', alpha = 0.25) #good for std or error plots

# Let's try working with dates adding it to the axis
plt.figure()
# Dates in a range with a type
observation_dates = np.arange('2017-01-01', '2017-01-09', dtype='datetime64[D]')
# Map them with pandas and put them in a list
observation_dates = list(map(pd.to_datetime, observation_dates)) # convert the map to a list to get rid of the error
plt.plot(observation_dates, lin_data, '-o',  observation_dates, quad_data, '-o')

# To rotate axis labels, iterate over each label using get()
xaxis = plt.gca().xaxis
for item in xaxis.get_ticklabels():
    item.set_rotation(45)

# adjust the sublot cause the image is out of bounds
plt.subplots_adjust(bottom = 0.25)

# It's possible to use LaTex language with '$ $'
plt.title("Quadratic ($x^2$) vs. Linear ($x$) performance")

########## Bar Charts ############
plt.figure()
lin_data = np.array([1,2,3,4,5,6,7,8,])
quad_data = lin_data**2
xvals = range(len(lin_data))
plt.bar(xvals,lin_data,width =0.3)

# To add a new column just put the Data
new_xvals = []
for item in xvals:
    new_xvals.append(item+0.3)

plt.bar(new_xvals,quad_data,width = 0.3, color ='red')

# Plot error bars
from random import randint
lin_error = [randint(0,15) for x in range(len(lin_data))]
plt.bar(xvals,lin_data,width = 0.3, yerr = lin_error)

# We can do stacked bar charts with 'bottom' param
plt.figure()
plt.bar(xvals,lin_data,width=0.3,color='g')
plt.bar(xvals,quad_data,width=0.3,bottom = lin_data,color='red')

# Pivot into an horizontal bar graph with 'barh'
plt.figure()
plt.barh(xvals,lin_data,height = 0.3, color = 'blue', alpha=0.25)


########## Dejunkifying a Plot ##########
languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]
plt.figure()
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')

# Give ticks names
plt.xticks(pos, languages, alpha=0.8)

# Title
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# use tick_params to remove ticks
plt.tick_params(bottom = False, left = False, labelbottom = 'on', labelleft = 'on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

#Change only one barh
bars[0].set_color('g')

# Ticks parameters:
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# Remove frameon
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Give each bar it's value
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%',
                 ha='center', color='w', fontsize=11)

############## Assignment 2 #############
def leaflet_plot_stations(binsize, hashid):
    df = pd.read_csv('BinSize_d{}.csv'.format(binsize))

    # Creates a new data frame according to param
    station_locations_by_hash = df[df['hash'] == hashid]

    # Creates to lists from the columns of df
    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    #Begins plot
    plt.figure(figsize=(8,8))
    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)
    #plt.show()
    # Maps it
    return mplleaflet.show()

leaflet_plot_stations(18,'37e494052ec73cae19435697b9afea9dc9776a5f270f67e19bf6a251')
'''
daf = pd.read_csv('DatosAss2.csv').sort(['ID','Date'])

#Remove leap days unziping Date in Year and Month-Date
daf['Year'], daf['Month-Date'] = zip(*daf['Date'].apply(lambda x: (x[:4], x[5:])))
#Keep if different than leap day
daf = daf[daf['Month-Date'] != '02-29']

a = daf[(daf['Year'] != '2015') & (daf['Element'] == 'TMIN')]

temp_min = daf[(daf['Element'] == 'TMIN') & (daf['Year'] != '2015')].groupby('Month-Date').aggregate({'Data_Value' : np.min})
temp_max = daf[(daf['Element'] == 'TMIN') & (daf['Year'] != '2015')].groupby('Month-Date').aggregate({'Data_Value' : np.max})

min_2015 = daf[(daf['Element'] == 'TMIN') & (daf['Year'] =='2015')].groupby('Month-Date').aggregate({'Data_Value' : np.min})
max_2015 = daf[(daf['Element'] == 'TMIN') & (daf['Year'] =='2015')].groupby('Month-Date').aggregate({'Data_Value' : np.max})

temp_min['Compare Min'] = min_2015['Data_Value']
temp_max['Compare Max'] = max_2015['Data_Value']

min_break = temp_min[temp_min['Compare Min'] < temp_min['Data_Value']]
del min_break['Data_Value']

max_break = temp_max[temp_max['Compare Max'] > temp_max['Data_Value']]
del max_break['Data_Value']

temp_min.columns=['Min', 'Compare Min']
temp_max.columns=['Max', 'Compare Max']
print(temp_min.to_string())

plt.figure()
plt.plot(temp_min['Min'], '-o', temp_max['Max'],'-o', markersize=1)
plt.scatter(min_break.index.values,min_break, color='black', marker='v', s=30,zorder=10)
plt.scatter(max_break.index.values,max_break, color = 'black', marker='^',s=30,zorder=10)
plt.title('Minimum and maximum temperatures 2006-2014')
plt.legend(loc=4)
#plt.scatter(temp_min['Compare Min'][temp_min['Compare Min'] < temp_min['Data_Value']], temp_max['Compare Max'][temp_min['Compare Max'] > temp_min['Data_Value']], markersize = 3)

#print(temp_min)
# Shade area between lines
plt.gca().fill_between(range(len(temp_min)),temp_min['Min'],temp_max['Max'],facecolor = 'blue', alpha = 0.25) #good for std or error plots

plt.xticks(list(temp_min.index[1::60]))

plt.show()









#print(daf.head().to_string())
#dates = pd.unique(daf['Date'])
#print(len(pd.unique(pd.unique(daf['ID']))))
