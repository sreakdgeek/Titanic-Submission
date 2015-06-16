import numpy as np
import decimal
from sklearn import linear_model
from numpy import genfromtxt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing as pr
from sklearn.metrics import f1_score

np.set_printoptions(threshold=np.nan)


train = np.genfromtxt('/home/nivvi80/hackathon/new_train.csv', dtype=np.float, delimiter=',', skip_header=1)
Y = train[:, 0]
X = train[:, 1:train.shape[1]]

X = pr.normalize(X, axis=1, norm='l1')

# shuffle and split training and test sets
print "Start training..."
random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=0)
print "Training done. Ready to classify..."
# Learn to predict each class against the other
classifier = linear_model.LogisticRegression(C=1e5)
print "Created the classifier. Ready to fit."

print X_train
print y_train
print X_test
print y_test
y_score = classifier.fit(X_train, y_train).predict(X_test)
print "After fitting the data. Ready to plot."
print sum(y_score)
print sum(y_test)
print y_score - y_test
print sum(y_score - y_test)
f1score = f1_score(y_test, y_score, average='binary')
print f1score

print "Start testing..."
test = np.genfromtxt('/home/nivvi80/hackathon/test.csv', dtype=np.float, delimiter=',', skip_header=1)
print np.isnan(test)

test = pr.normalize(test, axis=1, norm='l1')
test_score = classifier.fit(X, Y).predict(test)
test_score_int = test_score.astype(np.int)
np.savetxt('/home/nivvi80/hackathon/submission.csv', test_score_int, fmt='%d', delimiter=",")
