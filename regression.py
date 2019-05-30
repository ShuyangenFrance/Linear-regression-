#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:33:15 2019

@author: xiangshuyang
"""
import numpy as np
import matplotlib.pyplot as plt
class regression: 
    
    def  __init__(self,X,Y,tau,a,regressiontype):
        self.X=X
        self.Y=Y
        self.tau=tau
        self.a= a
        self.type=regressiontype
        
    def normalize(self, X,Y):
         Xmean= np.mean(X,0) # mean of X and Y
         Ymean=np.mean(Y,0)
         Xvar= np.var(X,0)
         X=(X-Xmean)/np.where(Xvar==0,1,Xvar)# variation of X
         Y=Y-Ymean
         return X,Y 
        
   ########standard regression############
    def standard_weight(self,X, Y): 
        return (X.T*X).I*(X.T*Y)
     
    ##########lwlr regression############
    def lwlr_weight(self,X,Y,chosenpoint,tau):
        m,n=np.shape(X)
        w=np.matrix(np.eye(m))
        for i in range(m):
            w[i,i]= np.exp(-(X[i,:] -chosenpoint)*(X[i,:] -chosenpoint).T/(2*tau**2) )
        return (X.T*X).I*(X.T*(w*Y))


    ##########rigid regression
    
    def rigid_weight(self,X,Y,a):
        m,n=np.shape(X)
        return (X.T*X+a*np.matrix(np.eye(n))).I*(X.T*Y)
    
        
    #####foward stagewise######## 
    def stagewise_weight(self,X,Y,iternum,epsilon):
        m,n=np.shape(X)
        weight = np.zeros((n,1))
        weighttemp= weight.copy()
        weighttest=weight.copy()
        for k in range(iternum):
            errorset = 10**(10)
            for j in range(n):
                for c in [-1,1]: # I use hthe pearson correlation here 
                    weighttest= weight.copy()
                    weighttest[j]+= epsilon * c
                    errortest=self.error(X*weighttest,Y)
                    if errortest< errorset:
                        weighttemp=weighttest
                        errorset= errortest
            weight=weighttemp
        return weight
    
    def return_y(self,X,weight):
        return X*weight 
            
    def error(self,y,Y):
        y=np.matrix(y)
        return (Y-y).T*(Y-y)

dataMat = []; labelMat = []
numFeat = len(open('regresstest.txt').readline().split('\t'))-1
fr = open('regresstest.txt')
for line in fr.readlines():
    lineArr =[]
    curLine= line.strip().split('\t')
    for i in range(numFeat):
        lineArr.append(float(curLine[i]))
    dataMat.append(lineArr)
    labelMat.append(float(curLine[-1]))
X=np.matrix(dataMat)
Y=np.matrix(labelMat).T

g=regression(X,Y,1,2,'stagewise')
X,Y=g.normalize(X,Y)
weight=g.stagewise_weight(X,Y,500,0.01)
y=g.return_y(X,weight)
print(y)
#theta=rigid_weight(X,Y,2)

plt.scatter(X[:,1].flatten().A[0],Y.flatten().A[0],color='red')
#e=error(y,Y)
#print(e)
#plt.plot(X[:,1],y,color='black')

