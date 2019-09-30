##################################################################
###################### Advanced Python Pandas ####################
##################################################################

import pandas as pd
import numpy as np
'''
################ Merging Dataframes ##################
df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df["Date"] = ['December 1', 'January 1', 'mid_may']

df['Delivered'] = True

# If there are only a few items to add. Input non values ourselfs
df["Feedback"] = ['Positive', None, 'Negative']
adf = df.reset_index()
adf['Date'] = pd.Series({0:'December', 2:'mid_may'})

# Merge two different data frames
# First create two data frames
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')

# Merging two data frames
#print(pd.merge(staff_df,student_df, how='outer', left_index=True,right_index=True)) #'outer' for union
#print(pd.merge(staff_df,student_df, how='inner', left_index=True,right_index=True)) #'inner' for intersection


# Get a list of staff and get their student info
#print(pd.merge(staff_df,student_df,how='left', left_index=True,right_index=True))

# Right joint for the other variable
#print(pd.merge(staff_df,student_df,how='right', left_index=True,right_index=True))

#Use columns to join them:
staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
print(pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name'))

#What happens when there are conflicts between data frames  (location is different between data frames)
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])

# _x is the left frame info and _y is the right frame info
print(pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name'))

# Multi index if for some key value is the same but for other is not
staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])

print(pd.merge(staff_df,student_df,how='inner' , left_on = ['First Name', 'Last Name'], right_on = ['First Name', 'Last Name']))

################## Pandas Idioms ###################
# Method chainning: every method on an object returns a reference on that object
df = pd.read_csv('census.csv')
df = (df.where(df['SUMLEV']==50).dropna().set_index(['STNAME','CTYNAME']).rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))

# apply() function takes the axis and the function which to operate
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

print(df.apply(min_max, axis=1).head())

# With lambdas
rows = ['POPESTIMATE2010',
'POPESTIMATE2011',
'POPESTIMATE2012',
'POPESTIMATE2013',
'POPESTIMATE2014',
'POPESTIMATE2015',]

print(df.apply(lambda x: np.max(x[rows]), axis=1))

################# Group by function #################
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg) + ' for processing.')

# provide a groupby function in one or more columns and segment the Data
# first, set the index of the column to group:
df = df.set_index('STNAME')
def fun(item):
    if item[0] < 'M':
        return 0
    if item[0] < 'Q':
        return 1
    return 2

for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group))

print(df.groupby(level=0)['POPESTIMATE2010', 'POPESTIMATE2011'].agg({'avg':np.average}))

############### Scales ###############
df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
#print(df['Grades'].astype('category').head())

# To change te order:
grades = df['Grades']. astype('category',
categories = ['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
ordered = True)print(df.pivot_table(values ='(kW)', index='YEAR',columns='Make',aggfunc = [np.mean,np.min])) #aggfunc = [,] for more than one function
'''
############## Assignmnent ###################
    ## Energy Data Base: Read excel file with no header
energy = pd.read_excel('Energy Indicators.xls', header = None)
#Drop the first column
energy = energy.drop(energy.columns[0:2],axis=1)
#Replace missing values
energy.replace('...', np.nan,inplace = True)
#Rename columns
energy.rename(columns={2: 'Country',3:'Energy Supply',
4:'Energy Supply per Capita',5:'% Renewable'}, inplace=True)
#Multiply energy Supply columns by 1000000
energy['Energy Supply'] *= 1000000
#Replace country names
repl = {"Republic of Korea": "South Korea",
"United States of America20": "United States",
"United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
"China, Hong Kong Special Administrative Region3": "Hong Kong", 'Bolivia (Plurinational State of)': 'Bolivia',
'Switzerland17' : 'Switzerland', 'China2' : 'China', 'Japan10':'Japan', 'Italy9' : 'Italy',
'Iran (Islamic Republic of)' :'Iran', 'Australia1':'Australia', 'Spain16':'Spain', 'France6':'France'}

energy.replace({"Country": repl},inplace = True)
#print
#print(energy)

##GDP Data Base - Read Data Base
GDP = pd.read_csv('world_bank.csv',skiprows=4)
repl={
"Hong Kong SAR, China": "Hong Kong",
'Iran, Islamic Rep.':'Iran'
}
GDP.replace({'Data Source':repl}, inplace = True)
#Keep columns I want
GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
#Chane columns names
GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']

#Print
#print(GDP[100:150])

## ScimEn Data Base - read file
ScimEn = pd.read_excel('scimagojr-3.xlsx')

#Print
#print(ScimEn['Country'])

############# Question 1 ####################
union = pd.merge(ScimEn,pd.merge(energy,GDP,how='outer'),how='outer')
old_len = len(union)
merge = union.drop(union[union['Rank']>15].index)[np.isfinite(union['Rank'])]
merge = merge.set_index('Country')
#print(merge.to_string())

############## Question 2 ##################
print('Lost items: ' + str(old_len-len(merge)))

############### Question 3 ###################
merge['avg']=merge[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].mean(axis=1)
