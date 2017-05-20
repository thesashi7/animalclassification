
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn import linear_model

#############################################################################
#    Classification of dog and cat species using vocalization
#    This is a logistic regression based model
#
#############################################################################


train_x = np.genfromtxt('/home/sashi/Documents/Spring2017/CS599/project/train-data.csv',
                          delimiter=',')
train_y = np.genfromtxt('/home/sashi/Documents/Spring2017/CS599/project/test-data.csv',
                          delimiter=',')

X_train = train_x[:.9 * train_x.shape[0]]
y_train = train_y[:.9 * train_y.shape[0]]
X_test = train_x[.9 * train_x.shape[0]:]
y_test = train_y[.9 * train_y.shape[0]:]

#print X_train.shape
#print X_test.shape
#print train_y.shape
#print y_test.shape

logistic = linear_model.LogisticRegression()

logistic.fit(X_train, y_train)
predict = logistic.predict(X_test)
print("SC Loss: "+str(log_loss(y_test,predict)))
print("Accuracy: "+str(accuracy_score(y_test, predict)))

