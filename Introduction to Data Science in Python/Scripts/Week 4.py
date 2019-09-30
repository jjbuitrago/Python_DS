##################################################################
####################### Statistical Analysis #####################
##################################################################

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
import math

import os
path = r'''C:\Users\Juan Jose\Documents\Programacion\Python Data Science\Introduction to Data Science in Python\Data Sets'''
os.chdir(path)


'''

############### Distributions #################

print(np.random.binomial(1000,0.5)/1000)

# Example: chance of a tornado two days in a row

chance = 0.01 #Suppose probs are 1% of a tornado any given day

tornado_events = np.random.binomial(1, chance,1000000)

two_days_in_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_row += 1

print('{} tornadoes back to back in {} years'. format(two_days_in_row,1000000/365))

# Uniform distribution
print(np.random.uniform(0,1))

# Normal distribution
print(np.random.normal(0.75)) #Default values are mean = 0 and sd = 1

# Standard distribution for N normal distributions
dist = np.random.normal(0.75, size=1000)

print(np.sqrt(np.sum((np.mean(dist)-dist)**2)/len(dist)))

# Also, use bulit-n functions
print(np.std(dist))

# Measure distributoin kurtosis
print(stats.kurtosis(dist))

# Measure distribution skew
print(stats.skew(dist))

# Chi-squares distribution
chi_squared_2df = np.random.chisquare(2,size=10000) #Parameter is degrees of freedom
print(stats.skew(chi_squared_2df))

chi_squared_50df = np.random.chisquare(15,size=10000) #Parameter is degrees of freedom
print(stats.skew(chi_squared_50df))

#PLotting both Distributions
output = plt.hist([chi_squared_2df,chi_squared_50df], bins=50, histtype='step',
                  label=['2 degrees of freedom','50 degrees of freedom'])
plt.legend(loc='upper right')
plt.show()

############### Hypothesis Testing in Python #############
df = pd.read_csv('grades.csv')
print(df.head().to_string())

#Let's segment the data
early = df[df['assignment1_submission']<='2015-12-31']
late = df[df['assignment1_submission'] > '2015-12-31']

#print(early.mean())
#print(late.mean())

### Mean difference test (T-Test)
# For assignment 1:
print(stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade']))

# For assignment 2:
print(stats.ttest_ind(early['assignment2_grade'], late['assignment2_grade']))

# For assignment 3:
print(stats.ttest_ind(early['assignment3_grade'], late['assignment3_grade']))

'''
###########################################
############# Assigment 4 #################
###########################################

############ Get list of university towns ############
with open('university_towns.txt') as file:
    # Crea una lista en donde se va a poner cada linea del archivo
    names = []
    for line in file:
        # Añade la linea al archivo
            names.append(line[:-1])
    # Crea una lista en donde se van a añadir las duplas
    towns = []
    for line in names:
        # If finds '[edit]' in the last 6 letter, it's a state. This var will be rewritten until it finds another one
        if line[-6:]=='[edit]':
            state = line[:-6]
        elif '(' in line:
            # If finds '(' in line, it's a town and will take from the start of the line until '(' index (-1 to not take account the space)
            town = line[:line.index('(')-1]
            # Appens as a duple state and town
            towns.append([state,town])
        else:
            # If doesn't fine either '[edit]' nor '(' take the entire line as town and appends the duple state and town
            town =  line
            #In both cases, state oesn't change until it finds a new one
            towns.append([state,town])
    #Creates data frame and give names to both columns
    college_df = pd.DataFrame(towns, columns=['State', 'RegionName'])
    college_df['dummy'] = 1
    college_df = college_df.set_index(['State','RegionName'])

# Printing examples
#print(college_df[college_df['State']=='Alabama'])

############   Get recession start ############
# First load the data
gdp = pd.read_excel('gdplev.xls', skiprows=7)
# Keep quarters and chained gdp since 2000
gdp = gdp.loc[212:].reset_index()
gdp = gdp[['Unnamed: 4','Unnamed: 6']]
# Rename columns
gdp.columns = ['QUARTER','GDP']

# Empty list to store recession beginnnings
quarters = []
# '-2' to study the two foregoing periods
for i in range(len(gdp)-2):
    # Checks recession condition
    if (gdp.iloc[i][1] > gdp.iloc[i+1][1]) & (gdp.iloc[i+1][1] > gdp.iloc[i+2][1]):
        # Append if true
        quarters.append(gdp.iloc[i][0])

# Get index of recession start
start = gdp[gdp['QUARTER']==quarters[0]].index.values.astype(int)[0]

############ Get recession end ############
# Empty list
end = []
# Get recession end (same method)
for i in range(start,len(gdp)-2):
    if (gdp.iloc[i][1] < gdp.iloc[i+1][1]) & (gdp.iloc[i+1][1] < gdp.iloc[i+2][1]):
        end.append(gdp.iloc[i][0])

# Get recession end index
ended = gdp[gdp['QUARTER']==end[0]].index.values.astype(int)[0]

######### Get recession bottom ############
#start at the beginning of recession
min = start
for i in range(start, ended):
    if gdp.iloc[min][1] > gdp.iloc[i][1]:
        #if next obs is less, change min to that obs
        min = i


############ Convert Housing Data to Quarters ############
# Load data
houses = pd.read_csv('City_Zhvi_AllHomes.csv').sort(['State', 'RegionName'])
#Keep columns
houses = houses[['RegionID','RegionName','State','Metro','CountyName','SizeRank','2000-01','2000-02','2000-03','2000-04','2000-05','2000-06','2000-07','2000-08','2000-09','2000-10','2000-11','2000-12','2001-01','2001-02','2001-03','2001-04','2001-05'
,'2001-06','2001-07','2001-08','2001-09','2001-10','2001-11','2001-12','2002-01','2002-02','2002-03','2002-04','2002-05','2002-06','2002-07','2002-08','2002-09','2002-10','2002-11','2002-12','2003-01','2003-02','2003-03','2003-04','2003-05','2003-06',
'2003-07','2003-08','2003-09','2003-10','2003-11','2003-12','2004-01','2004-02','2004-03','2004-04','2004-05','2004-06','2004-07','2004-08','2004-09','2004-10','2004-11','2004-12','2005-01','2005-02','2005-03','2005-04','2005-05','2005-06','2005-07',
'2005-08','2005-09','2005-10','2005-11','2005-12','2006-01','2006-02','2006-03','2006-04','2006-05','2006-06','2006-07','2006-08','2006-09','2006-10','2006-11','2006-12','2007-01','2007-02','2007-03','2007-04','2007-05','2007-06','2007-07','2007-08',
'2007-09','2007-10','2007-11','2007-12','2008-01','2008-02','2008-03','2008-04','2008-05','2008-06','2008-07','2008-08','2008-09','2008-10','2008-11','2008-12','2009-01','2009-02','2009-03','2009-04','2009-05','2009-06','2009-07','2009-08','2009-09',
'2009-10','2009-11','2009-12','2010-01','2010-02','2010-03','2010-04','2010-05','2010-06','2010-07','2010-08','2010-09','2010-10','2010-11','2010-12','2011-01','2011-02','2011-03','2011-04','2011-05','2011-06','2011-07','2011-08','2011-09','2011-10',
'2011-11','2011-12','2012-01','2012-02','2012-03','2012-04','2012-05','2012-06','2012-07','2012-08','2012-09','2012-10','2012-11','2012-12','2013-01','2013-02','2013-03','2013-04','2013-05','2013-06','2013-07','2013-08','2013-09','2013-10','2013-11',
'2013-12','2014-01','2014-02','2014-03','2014-04','2014-05','2014-06','2014-07','2014-08','2014-09','2014-10','2014-11','2014-12','2015-01','2015-02','2015-03','2015-04','2015-05','2015-06','2015-07','2015-08','2015-09','2015-10','2015-11','2015-12',
'2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08']]

#Index where gdp starts
index = 6
for year in range(2000,2016):
    for quarter in range(1,5):
        name = str(year) + 'q' + str(quarter)
        #Sum over the columns
        houses.loc[:,name] = houses.iloc[:,index:index+3].sum(axis=1)
        #Divide by 3
        houses.loc[:,name] = houses.loc[:,name]/3
        index += 3
# This ones manually
houses.loc[:,'2016q1'] = houses.iloc[:,198:201].sum(axis=1)
houses.loc[:,'2016q2'] = houses.iloc[:,201:204].sum(axis=1)
houses.loc[:,'2016q3'] = houses.iloc[:,204:206].sum(axis=1)
houses.loc[:,'2016q1'] = houses.loc[:,'2016q1']/3
houses.loc[:,'2016q2'] = houses.loc[:,'2016q2']/3
houses.loc[:,'2016q3'] = houses.loc[:,'2016q3']/2

#Keep these columns
houses = houses[['RegionName','State',quarters[0],end[0]]]


############ Run t-test ############

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska',
'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine',
 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico',
 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana',
 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico',
 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

# Change State for houses-df
houses.replace({'State':states},inplace=True)
houses = houses.set_index(['State','RegionName'])

# Merge both data frames by State and RegionName
merge = pd.merge(houses,college_df, how = 'outer', left_index = True, right_index = True)
#creates new data showing the decline or growth of housing prices
merge['diff'] = (merge['2008q2']-merge['2009q2'])/merge['2008q2']
#Replace nan for 0's
merge['dummy'] = merge['dummy'].fillna(0)

# Crete two data frames
u_towns = merge.copy()
u_towns = u_towns.drop(u_towns[u_towns['dummy']==0].index)
non_u_towns = merge.copy()
non_u_towns = non_u_towns.drop(non_u_towns[non_u_towns['dummy']==1].index)

# Perform t-Test
statistic, p_value = stats.ttest_ind(non_u_towns.dropna()['diff'],u_towns.dropna()['diff'])

# Create tuple:
better = ""
if p_value < 0.01:
    better = 'University Town'
else:
    better = "Non University Towns"

tuple = (p_value < 0.01, p_value, better)

#### Answers ####
print ('Recession started in ' + str(quarters[0]) + " and ended in " + str(end[0]) + ". Recession bottom was in: " + str(gdp.iloc[min][0]) )
print(tuple)

#print(stats.ttest_ind(u_towns['diff'],non_u_towns['diff']))

#Printing conditioning a column
#print(merge[['diff','dummy']][merge['dummy']==0].to_string)

#print(houses.head().to_string())
#print(college_df.head().to_string())
#print(merge.to_string())

# states printing examples
#for key in states.keys():
    #print(key)

#for key,value in states.items():
    #print(key)
    #print(value)
