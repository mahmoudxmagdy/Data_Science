
# coding: utf-8

# In[2]:

x=2
y=5
z=(x*x)+(y*y)/5
print(z)
print('Hello \nThis is me \nI am going to be a \t Data scientist')


# In[3]:

#Lists
Magdy=['Mahmoud','Magdy','Amin']
Magdy.append(['Ali','AIX'])
Magdy[3][1]
Magdy[0]='Abdel Samee3'
del(Magdy[1])
y=Magdy[1].split()
print(Magdy)
magdy2=Magdy[:]
magdy2


# In[4]:

#Tuples
Mtuple=(1,2,3,'Magdy',['B','C','X'])
Mtuple[0]


# In[5]:

#Sets
Set1={'I','am','Magdy','123','20'}
Set1.add('Gamed')
print(Set1)
Set1.remove('I')
print(Set1)
"aim" in Set1


# In[6]:

#Dictionaries

Dic={'m1':'Mahmoud', 'm2':'Magdy', 'm3':'Amin'}
Dic['m2']
Dic['m4']='Data'
del(Dic['m1'])
print(Dic)
'm2' in Dic
Dic.values()


# In[7]:

#IF statement
x1=20
y1=21
if x1==18:
    print('you can enter gate1')
elif x1<=17:
    print ('you can enter gate2')
elif y1==20 or x1>18:
    print ("Go HOME")
else:
    print('You have no place here')
    
print('Move on')


# In[8]:

#Loops
##For loop
r=[0,1,2,3,4]
r2=[]
x2=0
for i in r:
    x2=r[i]+1
    print(x2)
    r2.append(x2)
print(r2)


##While loop
x3=[0,1,2,3,4,5]
y3=[]

i2=0
while x3[i2]<5:
    y3.append(x3[i2])
    print(y3)
    i2=i2+1
    
print("Final shape of y3 = ",y3)


# In[9]:

#Functions
l1=[3,5,1,2,89,53,21]

l=len(r2) #length
print(l) 

s=sum(r2) #sum 
print(s)

l1=sorted(l1) #Sort a list
print(l1)

def f1(inp): # built function
    oup=inp+1;
    return oup
print(f1(s))

#loop in function
stuff=[10,7,8.5,9]
def prstuff(stuff):
    for i,s in enumerate(stuff):
        print("No.",i," Rate:",s)
prstuff(stuff)

#Variadic parameter
artists=["Mahmoud",'magdy','amin','ali']
def artist(*names):
    for name in names:
        print(name)
artist(artists)


# In[10]:

def addDC(y):
    global xy
    xy=y+"DC"
    return(xy)
x="AaaC"
z=addDC(x)
print(z)
print(xy)
type(z)



# In[11]:

#Classes

class cir():
    def __init__(self,radius,color):
        self.radius=radius
        self.color=color
    def add_radius(self,r):
        self.radius=self.radius+r

ct=cir(10,"Red")
print(ct.radius)
ct.add_radius(2)
print(ct.radius)
#dir(ct.color)
    


# In[12]:

##Data Analysis with Python

#file1=open("/resources/data/trial.txt","w")
#print(file1.name)
#file1.close

l1=["I love dogs\n","I don't like Cats\n","I may own a dog or Hippo but not a cat\n"]
with open ("/resources/data/trial.txt","w") as file1:
    for line in l1:
        file1.write(line)
print(file1.closed)

with open("/resources/data/trial.txt","r") as file1:       
    print(file1.read())
print(file1.closed)
    
    


# In[13]:

#dataframes and laoding data in pandas

import pandas as pd
csv_path="/resources/data/samples/olympic-medals/medals.csv"
df=pd.read_csv(csv_path)
df.head()

yeardf=df[['Year']]
#yeardf

df.iloc[0,2] #indexing iloc is used instead of ix as ix is not installed here
z=df.iloc[0:3,0:5]
z

df['Year'].unique() # shows the values of that column with no duplications
df['Year']>="2000" # gives boolean values of the condition corresponding to each row of column years

df1=df[df['Year']>="1990"] # put in new DF the old dataframe with only values after year 1990
df1.head()



# In[14]:

## Arrays

#1D array

import numpy as np
a=np.array([0,1,2,3,4,5,6,7,8,9])
a
type(a)
a.size
a.shape

# adding array a and b and put in array c
b=np.array([0,1,2,3,4,5,6,7,8,9])
c=a+b
c

##adding array using loops
u=np.array([0,1])
v=np.array([1,2])
uv=u+v
uv

uxv=2*v # multiplying arrays
uxv

udv=np.dot(u,v) #dot product of u and v
udv

max_b=b.max()
max_b

mean_b=b.mean()
mean_b

piarr=np.array([0,np.pi/2,np.pi]) # using pi=3.14 in the array
piarr

sinarr=np.sin(piarr) # convert pi to sin
sinarr

seq=np.linspace(-2,2,num=5) # creating seuence array
seq

#plot the pi array
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

piarr=np.linspace(0,2*np.pi,num=100)
sinarr=np.sin(piarr)
plt.plot(piarr,sinarr) ## Ploting sin curve

piarr.std()


# In[17]:

#2D Array

a=[[11,12,13],[21,22,23],[31,32,33]]
b=np.array([[11,11,11],[22,22,22],[33,33,33]])
A=np.array(a)
A.ndim
A.shape
A.size
A[0:3,1]
Adot=np.dot(A,b)
Adot


# In[24]:

#Data Analysis with Pyhton
import pandas as pd

filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df1=pd.read_csv(filename, names = headers)
df1.head()


# In[34]:

##cleaning data
import numpy as np

df1.replace("?", np.nan, inplace=True)
df1.head()

#check missing data
missing_data=df1.isnull()
missing_data
# check the number of lost values
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print(" ")
    # now we can see the number of missing values in each column


# In[49]:

# Replace missing data

#replace with mean
avg1=df1['normalized-losses'].astype(float).mean(axis=0)
df1['normalized-losses'].replace(np.nan,avg1,inplace=True)

avg2=df1['bore'].astype('float').mean(axis=0)
df1['bore'].replace(np.nan,avg2,inplace=True)

avg3=df1['stroke'].astype('float').mean(axis=0)
df1['stroke'].replace(np.nan,avg3,inplace=True)

avg4=df1['horsepower'].astype('float').mean(axis=0)
df1['horsepower'].replace(np.nan,avg4,inplace=True)

avg5=df1['peak-rpm'].astype('float').mean(axis=0)
df1['peak-rpm'].replace(np.nan,avg5,inplace=True)

#replace with frequency
df1['num-of-doors'].value_counts() #count values of that column
df1['num-of-doors'].value_counts().idxmax() #calculate the most common value

#replace the missing 'num-of-doors' values by the most frequent 
df1["num-of-doors"].replace(np.nan, "four", inplace = True)

#Drop whole rows with NAN in price column
df1.dropna(subset=['price'], axis=0, inplace=True)
df.reset_index(drop = True, inplace = True) # reset index as we dropped rows
df1.head()


# In[55]:

#corrceting formats
df1.dtypes #check data types in each column

#convert data types to correct formats
df1[['bore','stroke','price','peak-rpm']]=df1[['bore','stroke','price','peak-rpm']].astype('float')
df1[['normalized-losses']]=df1[['normalized-losses']].astype('int')

## NOW we have the data cleaned and in proper format

df1.dtypes


# In[57]:

df1.head(10)


# In[60]:

## Data Strandardization , to put the data in a standard format

#convert mpg to L/100km
df1['city-L/100km']=235/df1['city-mpg']
df1['highway-L/100km']=235/df1['highway-mpg']
df1.head()




# In[61]:

#Data Normalization, Transforming the values of several varliables in similar range

# dividing the values of length and width on their max values and replacing them with the original values
df1['length'] = df1['length']/df1['length'].max()
df1['width'] = df1['width']/df1['width'].max()
df1['height']=df1['height']/df1['height'].max()
df1[["length","width","height"]].head()


# In[71]:

# Data Binning, is to classify continuous values into limited number of bins, 
#Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis. 

df1["horsepower"]=df1["horsepower"].astype(float, copy=True) #convert data into correct format
binwidth = (max(df1["horsepower"])-min(df1["horsepower"]))/4 # we want 4 binwidth of similar size
bins=np.arange(min(df1['horsepower']), max(df1['horsepower']), binwidth) # We build a bin array, with a minimum value to a maximum value, with bandwidth calculated above. The bins will be values used to determine when one bin ends and another begins.
bins
group_names=['High','Medium','Low'] # We set group names
df1['horsepower-binned']=pd.cut(df1['horsepower'], bins, labels=group_names, include_lowest=True)
df1[['horsepower','horsepower-binned']].head(10)


# In[73]:

# Bins Visualization

get_ipython().magic(u'matplotlib inline')
import matplotlib as plt
from matplotlib import pyplot
ax=(0,1,2)
plt.pyplot.hist(df1['horsepower'],bins=3) #Draw histogram for horepower of 3 bins

#add labels

plt.pyplot.xlabel('horsepower')
plt.pyplot.ylabel('counts')
plt.pyplot.title('Horsepower Bins')


# In[81]:

## Indicator Variables or Dummy variables
#An indicator variable (or dummy variable) is a numerical variable used to label categories. 
#They are called 'dummies' because the numbers themselves don't have inherent meaning. as fule-type; desiel and gas to 1 and 2

dummy_var1= pd.get_dummies(df1["fuel-type"]) #assign dummy variables to fule_type column and put in new df
dummy_var1.head(10)

dummy_var1.rename(columns={'fuel-type-diesel':'gas','fuel-type-diesel':'diesel'}, inplace=True) #rename column names
dummy_var1.head()

df1=pd.concat([df1,dummy_var1],axis=1) #insert dummy_var1 to df1
df1.drop('fuel-type', axis=1, inplace=True)


# In[91]:

#do the same but for aspirations column

dummyvar2=pd.get_dummies(df1['aspiration'])
dummyvar2.head()

dummyvar2.rename(columns={'asporation-type':'std','aspiration-type':'turbo'}, inplace=True)
dummyvar2.head()

df1=pd.concat([df1,dummyvar2], axis=1)
df1.head()


# In[97]:

df1=pd.concat([df1,dummyvar2], axis=1)
df1.head()


# In[98]:

# Put the clean data in new csv file

df1.to_csv('magdy_clean_data.csv')


# In[ ]:



