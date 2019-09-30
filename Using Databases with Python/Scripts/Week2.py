###################################################
##########Using Databases with Python##############
###################################################
import os
path = r'''C:\Users\Juan Jose\Documents\Programacion\Python Data Science\Using Databases with Python\Data Base'''
os.chdir(path)

# Structured Query Language: is the language we use to issue commands to the database
# CRUD: Create, Read, Update & Delete

# Difference between Data Base Administrator and Data Base Developer
import sqlite3
conn = sqlite3.connect('emaildb.sqlite')

cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS Counts') ##This executes in the SQL command prompt
cur.execute('CREATE TABLE Counts (email TEXT, count INTEGER)')

fname = input('Enter file name: ')
if(len(fname)<1): fname = 'mbox-short.txt'


fh = open(fname)
for line in fh:
    if not line.startswith('From: '): continue #'continue' continues with the next iteration of the loop
    pieces = line.split()
    email=pieces[1]
    cur.execute('SELECT count FROM Counts WHERE email = ?', (email,))
    row = cur.fetchone()
    if row is None: #It's the first time the email appears
        cur.execute('INSERT INTO Counts (email,count) VALUES (?,1)', (email,))
    else: #Not the first time that email appears
        cur.execute('UPDATE Counts SET count = count + 1 WHERE email = ?',(email,))
    conn.commit() #Forces everything to be written to disk

## Sorting the data
sqlstr = 'SELECT email,count FROM Counts ORDER BY count DESC LIMIT 10' #'DESC' is descendent; 'LIMIT 10' are the first 10

for row in cur.execute(sqlstr):
    print(str(row[0]),'appears',row[1])

cur.close()













print()
