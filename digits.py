import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#read the file
data = pd.read_csv('train.csv').to_numpy()
clf = DecisionTreeClassifier()

# traning dataset
x_train=data[0:21000,1:]
y_train=data[0:21000]

#fit the model
clf.fit(x_train,y_train)

# testing data
x_test = data[21000: ,1:]
y_test = data[21000: ,0:]

 p = clf.predict(x_test)

count=0
for i in range(0,21000):
    count+=1 if p[i] == y_test[i] else 0
print("Accuracy = ", (count/21000)*1000)   

print(confusion_matrix(p,y_test))

#print(accuracy_score(p,y_test))
