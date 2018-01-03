#! /usr/bin/python3

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

# Loading the dataset 
iris = datasets.load_iris()

# Put the data and the targets together so they can be randomly 
# shuffled without losing their relationship. 
full_dataset = np.column_stack((iris.data,iris.target))
np.random.shuffle(full_dataset)

# Training set
X_train = full_dataset[:120,0:4]
y_train = full_dataset[:120,4]
# Test set
X_test = full_dataset[120:,0:4]
y_test = full_dataset[120:,4]

# Set up the classifier and train it on the training set
logRegClasif = LogisticRegression(verbose=1,max_iter=500,solver='lbfgs')
logRegClasif.fit(X_train,y_train)

# Test the accuracy of the model using the test set 
accuracy = logRegClasif.score(X_test,y_test) 

accuracy_percentage = round((accuracy*100),2)
print("Model accuracy: " + str(accuracy_percentage) + "%")




#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#le.fit(iris.target_names)
#target_names = le.inverse_transform(iris.target)
