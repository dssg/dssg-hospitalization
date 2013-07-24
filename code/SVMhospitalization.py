mport numpy as np
import numpy.ma as ma
import pylab as pl
from sklearn import datasets, svm
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
##http://scikit-learn.org/stable/_downloads/plot_iris_exercise1.py

name = 'flat_data_v1.csv'
data = pd.read_csv(name)
data = data.fillna(0)
nn = data.columns
yy = nn[6: ]
yy = yy.insert(0, nn[0])
yy = yy.insert(1, 'age')
xx = yy
newdata = data[xx]
for i in range(len(xx)):
    newdata[xx[i]] = np.array(newdata[xx[i]]).astype(np.float)
    
XXdata = np.array(newdata) ##also include age
Y = np.array(newdata[xx[0]])           
Y_1ind = np.where(Y == 1)[0]
Y_0ind = np.where(Y == 0)[0]
n_sample1 = len(Y_1ind)
n_sample0 = len(Y_0ind)
Data = XXdata
#split the data into train and test##90% in train and rest 10% in test
np.random.seed(0)
order_1 = np.random.permutation(n_sample1)
order_0 = np.random.permutation(n_sample0)
Data_1 = Data[Y_1ind[order_1], :]
Data_0 = Data[Y_0ind[order_0], :]
Data_train = (np.concatenate([Data_1[:.9*n_sample1, :],Data_0[:.9*n_sample0, :]]))
Data_test = (np.concatenate([Data_1[.9*n_sample1: , :],Data_0[.9*n_sample0: , :]]))
X_train = Data_train[:,1:]
X_train = (X_train - X_train.mean(axis = 0)) / (X_train.std(axis = 0, ddof = 1)+1e-15)
X_test = Data_test[:,1:]
X_test = (X_test - X_test.mean(axis = 0)) / (X_test.std(axis = 0, ddof = 1) + 1e-15)
Y_train = Data_train[:,0]
Y_test = Data_test[:,0]
###fit the model and plot
classifier = svm.SVC(C=100000, cache_size=200, class_weight={1:3}, coef0=0.0, degree=3,
                     gamma=1e-05, kernel='rbf', max_iter=-1, probability=False,
                     shrinking=True, tol=0.001, verbose=False)
y_ = classifier.fit(X_train, Y_train).predict(X_test)
cm_each = confusion_matrix(y_,Y_test) #confusion matrix
#cm.append(cm_each)
print cm_each
accuracy_each = (cm_each[0][0].astype(np.float)+cm_each[1][1].astype(np.float))/np.sum(cm_each).astype(np.float)
print accuracy_each
#accuracy.append(accuracy_each)
overlap_each = cm_each[1][1].astype(np.float)/(cm_each[1][0].astype(np.float)+cm_each[0][1].astype(np.float)+cm_each[1][1].astype(np.float))
print overlap_each
#overlap.append(overlap_each) 

sensitivity = []
specificity = []

for x in range(len(cm)):
    cm_small = cm[x]
    sensitivity_small = (cm_small[1][1].astype(np.float))/(cm_small[0][1].astype(np.float)+cm_small[1][1].astype(np.float))
    specificity_small = 1-(cm_small[1][0].astype(np.float))/(cm_small[0][0].astype(np.float)+cm_small[1][0].astype(np.float))
    sensitivity.append(sensitivity_small)
    specificity.append(specificity_small)


