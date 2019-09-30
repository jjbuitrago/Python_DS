##################################################################
############### Basic Data Processing using Pandas ###############
##################################################################

######## Series Data Structure ########

import pandas as pd
import numpy as np
'''
#Can be created from lists, where indexes starts at 0
animals = ['Tiger', 'Bear', 'dog']
print(pd.Series(animals))

#When creating a series with different data types, pandas change the type
animals = ['Tiger', 'Bear', None]
print(pd.Series(animals))
numbs = [1,2,None]
print(pd.Series(numbs)) #NaN := Not a Number != None

#series can be created form dictionaries, where index are the keys

s = pd.Series({'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'})
print (s)
print(s.index)

#¿What happens if the list objects are not aligned with the dictionary?
#Pandas will ignor from the dictionary all keys that are not in the index
g = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
print(g)

######## Querying a Series ########
# Use iloc and loc to get index value or key value
print(s.iloc[3])
print(s.loc['Golf']) # This is important if you are using an array of integers

# Iteration over series:
s = pd.Series([100.00, 120.00, 101.00, 3.00])
print (np.sum(s))

#For  bigger data
s = pd.Series(np.random.randint(0,1000,10000))
print(np.sum(s))

###When indices are not unique. Also, use append i two series:
original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'],
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)
print (all_countries)
print(all_countries.loc['Cricket'])

########### The DataFrame Data Structure ###########
# Primary object in pandas. It's a 2-d object

# Create a data frame using a set of Series

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1,purchase_2,purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

# Also use iloc and iloc
print(df.loc['Store 2'])
print(df.iloc[1])
print(type(df.loc['Store 1']))

# Get an entry with row and column indices
print(df.loc['Store 1', 'Cost'])

#Also use transpose as in arrays:
print(df.T)

#Getting some columns
print(df.loc[:,['Name', 'Cost']])

#Drop data using index or row label (doesn't change data, gives copy)
print(df.drop('Store 1'))

#To change it, create a copy
df_copy = df.copy()
df_copy = df.drop('Store 1')
print(df_copy)

#Another way is to use 'del'
del df_copy['Name']
print(df_copy)

#Adding a new column is pretty easy
df['Location'] = None
print(df)

########## DataFrame Indexing and Loading ##########
df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
#print(df.head())

#Iterate through columns
print(df.columns)
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)

print(df.head())

################ Querying a DataFrame ##############
# Boolean masking: is an array of booleans that's aligned to our data DataFrame
print(df['Gold']>0) # Creates a boolean column of countries with at least 1 gold medal

# To mask, use 'where'
only_gold = df.where(df['Gold']>0)
print(only_gold.head())

# And use dropan to delete the one's with NanN
only_gold = only_gold.dropna()
print(only_gold.head())

############ Indexing Dataframes #############
# Let's say we want to index by the number of gold medals won
df['Country'] = df.index # Creates a new column named Country
df = df.set_index('Gold') # Sets the index column
print(df.head())

df = df.reset_index() #Promotes index into a column and generates default index
print(df.head())

# Multilevel indexing
df = pd.read_csv('census.csv')

#Possible values for SUMLEV
print(df['SUMLEV'].unique())

#Let's keep county Data
df = df[df['SUMLEV']==50]
columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]

df = df.set_index(['STNAME', 'CTYNAME']) #Multilevel index
print(df.head())
#Since there is multilevel index, use it
print(df.loc[[('Michigan','Barbour County'), ('Michigan','Wayne')]])
