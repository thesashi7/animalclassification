
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#############################################################################
#    Classification of dog and cat species vocalization using logistic
#    regression model
#
#############################################################################


# Change File path to where you have your data
train_x = np.genfromtxt('/home/sashi/Documents/Spring2017/CS599/project/train-x.csv',
                          delimiter=',')
train_y = np.genfromtxt('/home/sashi/Documents/Spring2017/CS599/project/train-y.csv',
                          delimiter=',')

X_train,X_test,y_train,y_test = train_test_split(train_x,(train_y[:, np.newaxis]),
                                                 test_size=0.33, random_state=42)
#print X_train.shape
#print X_test.shape

logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)
predict = logistic.predict(X_test)
print("Loss: "+str(log_loss(y_test,predict)))
print("Accuracy: "+str(accuracy_score(y_test, predict)))

fig, ax = plt.subplots()
ax.scatter(y_test, predict)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()