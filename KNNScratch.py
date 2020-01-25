# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:52:19 2019

@author: kiran
"""


import numpy as np
import math
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

f = open("HandWrittenLetters.txt", "r")

line = f.readline()
arr = np.array([line.split(",")])
line = f.readline()

while line:    
    newarr = np.array(line.split(","))
    arr = np.append(arr,[newarr],axis=0)
    line = f.readline()

f.close()

f1=open("testDataX.txt", "r")
line1 = f1.readline()
tarr1 = np.array([line1.split(",")])
line1 = f1.readline()

while line1:    
    testarr = np.array(line1.split(","))
    tarr1 = np.append(tarr1,[testarr],axis=0)
    line1 = f1.readline()

f1.close()

tarr1 = np.array(np.transpose(tarr1)) 


#Subroutine 1
temp_arr = np.array([arr[:,0]])
def pickDataClass(arr1):
    for j in arr1:
        for i in range((arr.shape[1])):
            if int(arr[0][i]) == j:
                value = i               
                temp =np.array(arr[:,value])
                global temp_arr
                temp_arr = np.append(temp_arr,[temp],axis=0)
    temp_arr = np.array(np.delete(temp_arr,0,0))   
    temp_arr = np.array(np.transpose(temp_arr)) 
    global test_data_xy
    test_data_xy = np.array([temp_arr[:,0]])
    global train_data_xy
    train_data_xy = np.array([temp_arr[:,0]])


     

#Subroutine 4  
def letter_To_digit_Convert(data):
    class_data=[]
    for i in range(len(data)):
        class_data.append(ord(data[i].lower())-96)
        #print(ord(data[i].lower())-96)  
    pickDataClass(class_data)
    

#letter_To_digit_Convert("ABCDE")

#Subroutine 2
def splitData2TestTrain(number_per_class,test_instance):
    data = test_instance.split(":")
    k=0
    while(k < temp_arr.shape[1]):
        for i in range(k,int(data[1])+k):
            test_data = np.array(temp_arr[:,i])
            global test_data_xy
            test_data_xy = np.append(test_data_xy,[test_data],axis=0)
    
        for j in range(k+int(data[1]),number_per_class+k):
            train_data = np.array(temp_arr[:,j])
            global train_data_xy
            train_data_xy = np.append(train_data_xy,[train_data],axis=0)
            
        k = k+number_per_class
        
    test_data_xy = np.array(np.delete(test_data_xy,0,0))
    train_data_xy = np.array(np.delete(train_data_xy,0,0))
    
    test_data_xy = test_data_xy.astype(np.int)
    train_data_xy = train_data_xy.astype(np.int)
    
    global y_test
    y_test = np.array(test_data_xy[:,0])
    #test_data_xy = np.array(np.delete(test_data_xy,0,1))
    
    global y_train
    y_train = np.array(train_data_xy[:,0])
    #train_data_xy = np.array(np.delete(train_data_xy,0,1))
    global X_train
    X_train= train_data_xy[:,1:]
    global X_test
    X_test= test_data_xy[:,1:]
    


'''
#Preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


label_encoderX = LabelEncoder()
X_train[:,0] = label_encoderX.fit_transform(X_train[:,0])
X_test[:,0] = label_encoderX.transform(X_test[:,0])

label_encoderX1 = LabelEncoder()
X_train[:,2] = label_encoderX1.fit_transform(X_train[:,2])
X_test[:,2] = label_encoderX1.transform(X_test[:,2])

label_encoderX2 = LabelEncoder()
X_train[:,3] = label_encoderX2.fit_transform(X_train[:,3])
X_test[:,3] = label_encoderX2.transform(X_test[:,3])

label_encoderX3 = LabelEncoder()
X_train[:,4] = label_encoderX3.fit_transform(X_train[:,4])
X_test[:,4] = label_encoderX3.transform(X_test[:,4])

label_encoderX5 = LabelEncoder()
X_train[:,1] = label_encoderX5.fit_transform(X_train[:,1])
X_test[:,1] = label_encoderX5.transform(X_test[:,1])

label_encoderY = LabelEncoder()
y_train = label_encoderY.fit_transform(y_train)
y_test = label_encoderY.transform(y_test)

from sklearn.compose import ColumnTransformer

columntransformer = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X_train = np.array(columntransformer.fit_transform(X_train),dtype=np.float)
X_test = np.array(columntransformer.transform(X_test),dtype=np.float)
X_train = X_train[:,1:]  # Dummuy value trap
X_test = X_test[:,1:]
'''

#Build the Model


def calculateEuclideanDistance(X_test1):
    rows, cols = (X_train.shape[0], 3)  # rows of Train Data
    global distance
    distance=[[0 for i in range(cols)] for j in range(rows)] # 0- distance, 1 - class label ,2 train_sample index

    distance_index=0
    training_sample = 0
    #for i in X_test1: # i defines row count of X_test data
    for j in X_train: # j define row count of X_train
        k=0
        for k in range(0,X_train.shape[1]): #Get the count of columns
            # Caluclate Eucludian Distance for each point
            distance[distance_index][0] = distance[distance_index][0] + pow((X_test1[k]-j[k]),2)          
        distance[distance_index][0] = math.sqrt(distance[distance_index][0])
        distance[distance_index][1] = y_train[distance_index]
        distance[distance_index][2] = training_sample
        training_sample += 1
        distance_index = distance_index+1
            #print('Distance Index is ', distance_index)    


def sortFirst(val): 
    return val[0] 

def sortSecond(val): 
    return val[1]
 
#------------------------------------------------------------------------------
#For Majority Voting
def majorityVoting():
    majority_voting = [[-1 for i in range(2)] for j in range(5)]
    for temp in range(5):
        majority_voting[temp][1] = 0 

    for knn in range(5): # 5 represent number of Neighbors. This can be change according to the classification.
        value = distance[knn][1]
        for imp in majority_voting:
            if imp[0] == value:
                imp[1] += 1 
                break
            else:
                majority_voting[knn][0] = value
                majority_voting[knn][1] = 1
                break
            
    majority_voting.sort(key=sortSecond,reverse=True)
    print('Test data belong to class ',majority_voting[0][0] ,'Class Label ', label_encoderY.inverse_transform([majority_voting[0][0]]))  
 

#majorityVoting()    
#------------------------------------------------------------------------------
#Distance Weighted Voting


def distanceWeightedVoting(X_test1):
    row, col = (X_train.shape[0], 2)
    global distance_weighted
    distance_weighted=[[0 for i in range(col)] for j in range(row)]
    index=0    
    #for i in X_test: # i defines row count of X_test data
    for j in X_train: # j define row count of X_train
        k=0
        for k in range(7):            
            distance_weighted[index][0] = distance_weighted[index][0] + pow((X_test1[k]-j[k]),2)  #Distance Weighted for each train data point w.r.t test point        
        distance_weighted[index][0] = 1/(distance_weighted[index][0])
        distance_weighted[index][1] = y_train[index]
        index = index+1

#distanceWeightedVoting()       
knnlist=[]
def classifyTheData():
    nearest_neighbour =   [[0 for i in range(3)] for j in range(5)]  # Taking the nearest Neighbor
    for knn in range(5):  # Get the similar class value  
        nearest_neighbour[knn][0] =  distance[knn][0]   
        nearest_neighbour[knn][1] =  distance[knn][1] 
        nearest_neighbour[knn][2] =  distance[knn][2]    
        
    total_weight = 0;    
    final_class =-1
    for i in nearest_neighbour:
        weight=0   
        value = i[1]  # get the class Label
        for j in range(5):
            if value==nearest_neighbour[j][1]:
                index_value = nearest_neighbour[j][2]
                weight += distance_weighted[index_value][0] 
                if total_weight < weight:
                    total_weight = weight
                    #final_class = distance_weighted[index_value][1]
                    final_class = value
                     
    knnlist.append(final_class)
    #print(' Test Data Belongs to ',final_class)                               
 
#classifyTheData()
    
# COnfusion Matrix

def main():
    pickDataClass([1,2,3,4,6])
    splitData2TestTrain(39,'1:10')  # 1:10 means 10 test instance and rest will be training data.
    for i in X_test:
        calculateEuclideanDistance(i)
        distance.sort(key=sortFirst)
        distanceWeightedVoting(i)
        classifyTheData()
    print(knnlist)
    print('Accuracy Score ',accuracy_score(y_test, knnlist))
    

if __name__ == "__main__":
    main()
    
    
"""
This is just Backup
onehotencoder1 = OneHotEncoder(categorical_features = [0])
y = onehotencoder1.fit_transform(y).toarray()
"""
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
"""    
