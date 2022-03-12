# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = pd.read_csv('iris.csv') 
data.drop('Id',axis=1,inplace=True)
print(data) 
data.describe()

x=data.iloc[:,1:-1]
y=data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)

x_train.head(2)
x_test.head(2)

model=LogisticRegression()
model.fit(x_train, y_train)
prediction=model.predict(x_test)
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(prediction,y_test))

fig = data[data.Species=='Iris-setosa'].plot(kind='hist',x='SepalLengthCm',y='SepalWidthCm',color='red', label='Setosa')
data[data.Species=='Iris-versicolor'].plot(kind='hist',x='SepalLengthCm',y='SepalWidthCm',color='purple', label='versicolor',ax=fig)
data[data.Species=='Iris-virginica'].plot(kind='hist',x='SepalLengthCm',y='SepalWidthCm',color='black', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(8,4)
plt.show()

fig = data[data.Species=='Iris-setosa'].plot.hist(x='PetalLengthCm',y='PetalWidthCm',color='red', label='Setosa')
data[data.Species=='Iris-versicolor'].plot.hist(x='PetalLengthCm',y='PetalWidthCm',color='purple', label='versicolor',ax=fig)
data[data.Species=='Iris-virginica'].plot.hist(x='PetalLengthCm',y='PetalWidthCm',color='black', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(8,4)
plt.show()

data.hist(color='yellow', edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
