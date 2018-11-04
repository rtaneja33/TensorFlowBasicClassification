## Rohan Taneja, Nov. 4th


## Basic Notes: 
# "Machine Learning is about learning some properties of a 
# data set and then testing those properties against 
# another data set"

from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris() ## multivariate dataset to quantify variation if iris flowers
digits = datasets.load_digits() ## multivariate dataset of handwritten digits 

## digits.target = target data set with ground truth ([0, 1, 3,2, etc])
## similar to labels in tensor flow

## digits.data gives access to features used to classify data samples

clf = svm.SVC(gamma=0.001, C=100.)
## svm implements support vector classification
## an estimator for classification implements fit(x,y) and predict()

## clf for classifier instance, call fit here to learn from model
## digits.data[:-1] uses all images except for last from data set
clf.fit(digits.data[:-1], digits.target[:-1])  
clf.predict(digits.data[-1:])


##MODEL PERSISTENCE 
## pickle module implements a algorithm for serializing an object strucutre
## in other words, we can save a model in sci-kit learn

# import pickle

# s= pickle.dumps(clf)
# clf2 = pickle.loads(s)
# X = iris.data
# clf2.predict(X[0:1])

