import numpy as np
import datetime as dt
import time as tm
import csv
'''
def numbs(x,y,z =None, flag = False):
    if(flag):
        return ("Flag is true")
    if(z==None):
        return x+y
    else:
        return x+y+z
print (numbs(1,2,flag=True))
print (numbs(1,2,4,flag=False))

##################################################################
######################## Built in Types ##########################
##################################################################

##Tuple := sequence of variables which itself s inmutable
tuple = (1,'2',3,'b')
print (type(tuple))

#List := sequence that is mutable
list = [3,'d',True]
print (type(list))

list.append(3.4) #append to add objects
print (list)

#Both of them are iterable with a for
for item in list:
    print(item)

#Also with a while
i=0
while(i!=len(tuple)):
    print(tuple[i])
    i+=1

#"+" concatenates to lists and "*" repeats
x = [1,2]+[3,4]
print (x)
y = [1]*3
print(y)

#Command "in" returns a boolean
print(1 in[1,2,3])

#Slicing:
x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters

#index can be negative
print(x[-4:-2])

#Leaving the parameter empty reference the start or the end
print(x[:3])
print(x[3:])

#Split operations
firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)

#Dictionaries: labeled collections without order. Built tihe "{key: value}"
x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
print (x['Christopher Brooks']) # Retrieve a value by using the indexing operator

#Iterate over all keys:
for name in x:
    print(x[name])

#Iterate over values:
for email in x.values():
    print(email)

#Iterate over Both
for name, email in x.items():
    print(name)
    print(email)

#Unpacking a collection in different variables (error occuers if number of variables is different):
x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x

print (fname + " " + lname + " " +email)

##################################################################
############################ Strings #############################
##################################################################

# Works with Unicode
# Dynamic typing:
#print ("Juan " +2) #Doesn't Work
print ("Juan " + str(2)) #Works!!

# Format function: replace brackets

sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'Chris'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))

##################################################################
################# Reading and Writing .csv files #################
##################################################################

#%precision 2

with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))

#Get the columns
print (mpg[0].keys())

#Get some average
print (sum(float(d['cty']) for d in mpg)/len(mpg))
print (sum(float(d['hwy']) for d in mpg)/len(mpg))

#Know the average city mpg has gropued by the numbers of cylinders a car has
cyl = set(d['cyl'] for d in mpg)
print(cyl)

#Empty list to store results:
CityMpgByCyl = []

for c in cyl:
    summpg = 0
    CylTypeCount = 0
    for d in mpg:
        if d["cyl"] ==c:
            summpg += float (d['cty'])
            CylTypeCount +=1
    CityMpgByCyl.append((c, summpg/CylTypeCount))

CityMpgByCyl.sort(key=lambda x: x[0])
print(CityMpgByCyl)

# Suppose we want to find the hiaghest highway mpg for the different vehicle classes
vehicleClass = set(d['class'] for d in mpg)

HwyMpgByClass = []
for t in vehicleClass:
    sumMpg = 0
    vClassCount = 0
    for d in mpg:
        if d['class']==t:
            sumMpg+=float(d['hwy'])
            vClassCount+=1
    HwyMpgByClass.append((t,sumMpg/vClassCount))


HwyMpgByClass.sort(key=lambda x: x[1])
print(HwyMpgByClass)

##################################################################
########################## Dates & Times #########################
##################################################################

print (tm.time())

# Create a time stamp
dtnow = dt.datetime.fromtimestamp(tm.time())
print(dtnow.day)

##################################################################
######################## Objects & map() #########################
##################################################################

class Person:
    dept = "School of Information" #Class variables
    # Methods
    def set_name(self,new_name):
        self.name = new_name
    def set_loc(self,new_loc):
        self.loc = new_loc

#map looks like map(function,iterable,...)
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
print (cheapest)

##################################################################
################## lambda and list comprehension #################
##################################################################

#lambda functions are like anonymous functions (are simple and shorter)

my_func = lambda a, b, c : a + b # func_name = lamba [parameters]:expresion

print(my_func(1,2,3))

# List comprehensions are like global sequences
my_list = []
for number in range(0,1000):
    if number % 2 ==0:
        my_list.append(number)
print(my_list)

#In list comprehension (has performance benefits):
my_list_2 = [number for number in range(0,1000) if number % 2 ==0]
print(my_list_2)

##################################################################
################## The Numerical Python Library ##################
##################################################################

# Let us work with arrays and matrices in Python

# creating arrays
mylist = [1,2,3]
x = np.array(mylist)
y = np.array([4, 5, 6])
print(mylist)
print (x)

#Multidimensions
m = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(m)

#Check dimensions with shape
print(m.shape)

#Create an array with a start, an end and spaces
n = np.arange(0,30,2)

#Reshape to create a matrix
print(n.reshape(3,5))

#linspace is similar, but we tell how many numbers we want
o = np.linspace(1,4,9)
print(o)
h = o.reshape(3,3)
print(h)

#Shortcuts
print(np.ones((3,2))) #table of 1's
print(np.zeros((3,3))) #table of 0's
print(np.eye(3))
print(np.diag((3,3)))

# Operations
# Element wise are straight forward
print(x+y)
print(x*y)
print(x**2)
print(x.dot(y))

z = np.array([y,y**2])
print(z.T) #transpose
print (z.dtype)

# Math functions
a = np.array([-4, -2, 1, 3, 5])
print(a.sum())
print(a.max())
print(a.min())
print(a.mean())
print(a.std())
print(a.argmax()) # Index of max argument

# Index and Slicing
s = np.arange(13)**2
#print(s[0:3])

# two dimensions
r = np.arange(36)
r.resize((6,6))
#print(r)

#Get a slice of the 3th row form columns 3 to 6
print(r[3,3:6])

#Every 2nd element of the last row
print(r[-1,::2]) #rember that thirs parameter is steps

print(r[r>30]) # Conditional array
r[r>30] = 666 # Change if entry is greater than 30


#Copying data
r2 = r[:3,:3]
r2[:] = 0 #This will also change r
print (r)
r_copy = r.copy() #Creates a copy. Is independent from the original

# Itearting over arrays
test = np.random.randint(0,10,(4,3))
print (test)

for i in test:
    print(i) #Option 1

for i in range(len(test)):
    print(test[i]) #option 2

#Iterate in two arrays
test2 = test**2
for i,j in zip(test,test2):
    print(i, '+',j,'=',j+i)
