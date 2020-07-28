# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:13:15 2019

@author: Harsh Anand
"""

import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
key = pd.read_csv("key.csv")
station = pd.read_csv("station.csv")
weather = pd.read_csv("weather.csv")
sample_df = pd.read_csv("sample_df.csv")

train = train[(train.units != 0)]
train = train.reset_index(drop=True)
#take 1% of data using frac and random_state is used for same type of data
# train = train.sample(frac = 0.01,random_state = 1)
'''
length = len(train)*.01         #take 1% of train data
train = train.loc[0:length,:] '''



#merge the key column of station_nbr into train with match store_nbr from both table***************************************
list1=[]
for item in train['store_nbr']:
    #print(item)
    for index,item_key in enumerate(key['store_nbr']):
        if item == item_key:
            list1.append(key['station_nbr'][index]) #store the specific index value of station_nbr where item & item_key match
list1 = pd.DataFrame(list1) #convert list into dataframe
train['station_nbr'] = list1 #append the list1 into new column of train dataset


#################################################################################################################################################################
#take the specific column with specific value
'''
train = train[train['store_nbr']==1]  
train = train[train['item_nbr']==9]  '''

# select the unit which is not 0 in column 
'''train = train[(train.units != 0)]'''

#merge the weather columns into train with match station_nbr & date from both table***************************************
tmax=[]
tmin=[]
tavg=[]
depart=[]
dewpoint=[]
wetbulb=[]
heat=[]
cool=[]
sunrise=[]
sunset=[]
codesum=[]
snowfall=[]
preciptotal=[]
stnpressure=[]
sealevel=[]
resultspeed=[]
resultdir=[]
avgspeed=[]

for item_station,item_date in zip(train['station_nbr'],train['date']):
    #print(item_station,item_date)
    for index,weather_date in enumerate(weather['date']):
        #print(weather_station,weather_date)

        if item_date == weather_date and item_station == weather['station_nbr'][index]:
            #compare both date and station_nbr and we check station_nbr at specific index where date are matched
            # and append into specific list
            tmax.append(weather['tmax'][index])
            tmin.append(weather['tmin'][index])
            tavg.append(weather['tavg'][index])
            depart.append(weather['depart'][index])
            dewpoint.append(weather['dewpoint'][index])
            wetbulb.append(weather['wetbulb'][index])
            heat.append(weather['heat'][index])
            cool.append(weather['cool'][index])
            sunrise.append(weather['sunrise'][index])
            sunset.append(weather['sunset'][index])
            codesum.append(weather['codesum'][index])
            snowfall.append(weather['snowfall'][index])
            preciptotal.append(weather['preciptotal'][index])
            stnpressure.append(weather['stnpressure'][index])
            sealevel.append(weather['sealevel'][index])
            resultspeed.append(weather['resultspeed'][index])
            resultdir.append(weather['resultdir'][index])
            avgspeed.append(weather['avgspeed'][index])
       
            
            
tmax = pd.DataFrame(tmax) #convert list into dataframe
tmin = pd.DataFrame(tmin)
tavg = pd.DataFrame(tavg)
depart = pd.DataFrame(depart)
dewpoint = pd.DataFrame(dewpoint)
wetbulb = pd.DataFrame(wetbulb)
heat = pd.DataFrame(heat)
cool = pd.DataFrame(cool)
sunrise = pd.DataFrame(sunrise)
sunset = pd.DataFrame(sunset)
codesum = pd.DataFrame(codesum)
snowfall = pd.DataFrame(snowfall)
preciptotal = pd.DataFrame(preciptotal)
stnpressure = pd.DataFrame(stnpressure)
sealevel = pd.DataFrame(sealevel)
resultspeed = pd.DataFrame(resultspeed)
resultdir = pd.DataFrame(resultdir)
avgspeed = pd.DataFrame(avgspeed)


train['tmax'] = tmax #append the list into new column of train dataset
train['tmin'] = tmin
train['tavg'] = tavg
train['depart'] = depart
train['dewpoint'] = dewpoint
train['wetbulb'] = wetbulb
train['heat'] = heat
train['cool'] = cool
train['sunrise'] = sunrise
train['sunset'] = sunset
train['codesum'] = codesum
train['snowfall'] = snowfall
train['preciptotal'] = preciptotal
train['stnpressure'] = stnpressure
train['sealevel'] = sealevel
train['resultspeed'] = resultspeed
train['resultdir'] = resultdir
train['avgspeed'] = avgspeed

'''train.to_csv('File Name.csv',index=False) #save the dataset into csv file '''
# ***********************************************************************************************



import pickle


import pandas as pd
file = pd.read_csv("File Name.csv")


month=[]
season=[]
year = []
for item in file['date']:
    m_onth=[]
    m_onth = item.split("-")
    year.append(m_onth[0])
    if (int(m_onth[1])==12 or int(m_onth[1])==1 or int(m_onth[1])==2):
        season.append("winter")
    elif (3 <= int(m_onth[1]) < 6):
        season.append("spring")
    elif (6 <= int(m_onth[1]) < 9):
        season.append("summer")
    elif (9 <= int(m_onth[1]) < 12):
        season.append("fall")

        
    if int(m_onth[1])==1:
        month.append("january")
    elif int(m_onth[1])==2:
        month.append("february")
    elif int(m_onth[1])==3:
        month.append("march")
    elif int(m_onth[1])==4:
        month.append("april")
    elif int(m_onth[1])==5:
        month.append("may")
    elif int(m_onth[1])==6:
        month.append("june")
    elif int(m_onth[1])==7:
        month.append("july")
    elif int(m_onth[1])==8:
        month.append("august")
    elif int(m_onth[1])==9:
        month.append("september")
    elif int(m_onth[1])==10:
        month.append("october")
    elif int(m_onth[1])==11:
        month.append("november")
    elif int(m_onth[1])==12:
        month.append("december")

year = pd.DataFrame(year)
file['year'] = year
month = pd.DataFrame(month)
file['month'] = month
season = pd.DataFrame(season)
file['season'] = season
        
file = file.drop(['depart','tmax','tmin','dewpoint','wetbulb','heat','cool','sunrise','resultdir','resultspeed','sunset','codesum'],axis=1)

#*************************************************************************************************
'''
item1_fall=0
item1_winter=0
item1_spring=0
item1_summer=0
i=1
while i<=1:
    for item,season,unit in zip(file['item_nbr'],file['season'],file['units']):
        #print(item,season,unit)
        if item == i and season == "fall":
            item1_fall+=unit
        elif item == i and season == "winter":
            item1_winter+=unit
        elif item == i and season == "spring":
            item1_spring+=unit
        elif item == i and season == "summer":
            item1_summer+=unit
            
            
            
    i+=1
 '''   


# ********************* plotting *******************************************************************
''' plot the graph for top 10 units a/c season'''
unit = file.nlargest(10, ['units']) 

season = unit['season']
unit_s = unit['units']

import matplotlib.pyplot as plt
plt.bar(season, unit_s)
plt.xlabel('Season', fontsize=15)
plt.ylabel('No. of Units', fontsize=15)
plt.title('Top 10 units')
plt.show()


''' plot the graph for top 10 units a/c store, item'''
item =unit['store_nbr'].astype(str) + "," + unit['item_nbr'].astype(str)

import matplotlib.pyplot as plt
plt.bar(item, unit_s)
plt.xlabel('store_nbr , item_nbr', fontsize=15)
plt.ylabel('No. of Units', fontsize=15)
plt.title('Top 10 units sold for Store, Item')
plt.show()

#pie chart for top 5 units at specific store_nbr & item_nbr
unit = file.nlargest(5, ['units']) 
item ="store_nbr :" + unit['store_nbr'].astype(str) + "," + "Item_nbr :" + unit['item_nbr'].astype(str)
unit_s = unit['units']

plt.figure(figsize=(6,6))
plt.pie(unit_s, labels=item, startangle=90, autopct='%.1f%%')
plt.title('Top 5 units sold at specific Store_nbr, Item_nbr')
plt.show()

#*************************************************************************************************
top_10 = file.iloc[:,[1,2,3,13]]
top_10.to_csv('top_10.csv',index=False)
features = file.iloc[:,[1,2,13]]
labels = file.iloc[:,3].values

'''
#store the specific item_nbr for specific store_nbr
item = []
for i in range(1,46):
    li=[]
    li = list(features['item_nbr'][features['store_nbr']==i].unique())
    item.append(li)
'''

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()   
'''
features = pd.get_dummies(features.astype(str))

features= pd.get_dummies(features,columns=['store_nbr'] )
features = features.drop('store_nbr_1',axis=1)'''

# Encode labels in column 'season'. 
features.iloc[:,2:] = label_encoder.fit_transform(features.iloc[:,2:]) 
#features= pd.get_dummies(features)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=0) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
    
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

with open('feature_scaling.pkl','wb') as f:
    pickle.dump(sc,f)



from sklearn.ensemble import RandomForestRegressor
#n_estimator --> no of decision tree to create from random data
regressor = RandomForestRegressor(n_estimators=55, random_state=0)  
regressor.fit(features_train, labels_train)  #accuracy = 0.11533


with open('picklefile.pkl','wb') as f:
    pickle.dump(regressor,f)

''' 
 # or  ******************************************************
from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(features_train, labels_train)  #accuracy = 0.11321
# **********************************************************
from sklearn.preprocessing import PolynomialFeatures
poly_object = PolynomialFeatures(degree = 5)
features_poly_t = poly_object.fit_transform(features_train)#converting features into degree 5
features_poly_te = poly_object.fit_transform(features_test)

from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(features_poly_t, labels_train)
labels_pred = lin_reg_2.predict(features_poly_te)
lin_reg_2.predict(poly_object.transform([[38,9,3]])) #accuracy = 0.10321
# ***********************************************************
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train) #accuracy = 0.007455


# ************************************************************
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors = 5, p = 2) #When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
regressor.fit(features_train, labels_train)  #accuracy = 0.10804


# *************************************************************
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
lm = LinearRegression ()
lm_lasso = Lasso() 
lm_ridge =  Ridge() 
lm_elastic = ElasticNet() 


#Fit a model on the train data


lm.fit(features_train, labels_train)
lm_lasso.fit(features_train, labels_train)
lm_ridge.fit(features_train, labels_train)
lm_elastic.fit(features_train, labels_train)

#Predict on test and training data

predict_test_lm =	lm.predict(features_test ) 
predict_test_lm = predict_test_lm.astype(int)   #0.007455
predict_test_lasso = lm_lasso.predict (features_test) 
predict_test_lasso = predict_test_lasso.astype(int)  #0.007371
predict_test_ridge = lm_ridge.predict (features_test)
predict_test_ridge = predict_test_ridge.astype(int) #0.007455
predict_test_elastic = lm_elastic.predict(features_test)
predict_test_elastic = predict_test_elastic.astype(int) #0.007961
from sklearn.metrics import accuracy_score
accuracy_score(labels_test,predict_test_lm)
accuracy_score(labels_test,predict_test_lasso)
accuracy_score(labels_test,predict_test_ridge)
accuracy_score(labels_test,predict_test_elastic)



#*************************************************************
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(features_train, labels_train)

#*************************************************************
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
gnb = GaussianNB()
gnb.fit(features_train,labels_train)
labels_pred = gnb.predict(features_test)  #accuracy-->0.09073


# ***************************************************************************

'''

labels_pred = regressor.predict(features_test)
labels_pred = labels_pred.astype(int)


'''
from sklearn.metrics import accuracy_score, confusion_matrix
cn = confusion_matrix(labels_test,labels_pred) 
accuracy_score(labels_test,labels_pred) 
'''

#d = pd.Series(labels).value_counts()    


import numpy as np
input_data = [1,8,3]

input_data = np.array(input_data)
input_data = input_data.reshape(-1,3)

#input_data[:,2:] = label_encoder.transform(input_data[:,2:]) 
input_data = sc.transform(input_data)



input_pred = regressor.predict(input_data)
input_pred = input_pred.astype(int)

print("units :",input_pred[0])


