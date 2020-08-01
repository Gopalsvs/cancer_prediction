import pandas as pd
import numpy as np 
import matplotlib as plt
from sklearn .model_selection import train_test_split as tts
from sklearn import preprocessing as prepro
from sklearn import metrics
from sklearn import linear_model

		#reading data
data=pd.read_csv("data.csv")

#analysing data
#print data.info()
#print data.describe()
y=data['diagnosis']
X=data.drop(["diagnosis","id","Unnamed: 32"],axis=1)
#print data.shape
#for i in data.columns:
#   print(i)
y=[1 if i == 'M' else 0 for i in y]

		#splitting data train set and test set
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.2,random_state=42,stratify=y)

		#normalisation
scaler=prepro.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
X_train_scaled=X_train_scaled.T
X_test_scaled=X_test_scaled.T
y_train=np.array(y_train)
y_test=np.array(y_test)
#print X_test_scaled.shape
#print X_train_scaled.shape
#print y_test.shape
#print y_train.shape

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def intializewb(dimension):
    w=np.full((dimension,1),0.01)
    b=0
    return w,b

def cost_gradient(w,b,X_train,y_train):
    z=np.dot(w.T,X_train)+b
    h=sigmoid(z)
    loss=-y_train*np.log(h)-(1-y_train)*np.log(1-h)
    #loss=-y_train*np.log(h)-(1-y_train)*np.log(1-h)
    cost=(np.sum(loss))/X_train.shape[1]
    derivative_weight=(np.dot(X_train,(h-y_train).T))/X_train.shape[1]
    derivative_bias=np.sum(h-y_train)/X_train.shape[1]
    gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost,gradients

def updateparameters(X_train,y_train,learningrate,numiterator,w,b):
    cost_list=[]
    cost_list1=[]
    index=[]
    for i in range(numiterator):
        cost,gradients=cost_gradient(w,b,X_train,y_train)
        cost_list.append(cost)
        w=w-(learningrate)*gradients["derivative_weight"]
        b=b-(learningrate)*gradients["derivative_bias"]
        if i%10==0:
            cost_list1.append(cost)
            index.append(i)
    parameters={"weight":w,"bias":b}
    return parameters,cost_list

def predict(w,b,X_test):
    z=np.dot(w.T,X_test)+b
    prediction=sigmoid(z)
    for i in range(z.shape[1]):
        if(prediction[0,i]>=0.5):
            prediction[0,i]=1
        if(prediction[0,i]<=0.5):
            prediction[0,i]=0
    return prediction

def misclassificationerror(prediction,y_test):
    errorlist=[]
    prediction.tolist()
    y_test.tolist()
    for i in range(prediction.shape[1]):
        if(prediction[i]==1 & y_test[i]==0):
            errorlist.append(1)
        elif(prediction[i]==0 & y_test[i]==1):
            errorlist.append(1)
        else:
            errorlist.append(0)
    return sum(errorlist)
        
def logistic_regression(X_train,y_train,X_test,y_test,learningrate,numiterator):
    dimn=X_train.shape[0]
    w,b=intializewb(dimn)
    parameters,cost_list=updateparameters(X_train,y_train,learningrate,numiterator,w,b)
    X_test_prediction=predict(parameters["weight"],parameters["bias"],X_test)
    X_train_prediction=predict(parameters["weight"],parameters["bias"],X_train)
    #print "X_train accuracy:",misclassificationerror(X_train_prediction,y_train)/X_train.shape[1]
    #print "X_test accuracy:",misclassificationerror(X_test_prediction,y_test)/X_test.shape[1]
    print "X_train accuracy:",metrics.accuracy_score(X_train_prediction.T,y_train)
    print "X_test accuracy:",metrics.accuracy_score(X_test_prediction.T,y_test)
    #print X_train_prediction.shape
    #print X_test_prediction.shape
logistic_regression(X_train_scaled,y_train,X_test_scaled,y_test,1,100)

		#in built libraries
logistic_regression_inbuilt=linear_model.LogisticRegression(random_state=42,max_iter=150)
clf=logistic_regression_inbuilt.fit(X_train_scaled.T,y_train)
#print "X_train accuracy with inbuilt:",logistic_regression_inbuilt.fit(X_train_scaled.T,y_train.T).score(X_test_scaled.T,y_train.T)
#print "X_test accuracy with inbuilt:",logistic_regression_inbuilt.fit(X_train_scaled.T,y_train.T).score(X_test_scaled.T,y_test.T)
print "X_train accuracy with inbuilt:",clf.score(X_train_scaled.T,y_train.T)
print "X_test accuracy with inbuilt:",clf.score(X_test_scaled.T,y_test.T)
