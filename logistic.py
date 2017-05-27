
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preparedata import FeatureLoader
#############################################################################
#    Classification of dog and cat species vocalization using logistic
#    regression model
#
#############################################################################

# Load data from CSV file. Edit this to point to the features file
#data, target = FeatureLoader("data/features.csv").loadLibrosaCSV()
data, target = FeatureLoader().loadFeatures(["data/train-x.csv","data/train-y.csv"])

# Split the data into two parts: training data and testing data
train_data, test_data, train_target, test_target = train_test_split(
                 data, (target[:, np.newaxis]), test_size=0.3, random_state=42)

print train_target.shape
print train_data.shape
print test_data.shape

#train_target = np.ravel(train_target)
print train_target.shape

logistic = linear_model.LogisticRegression()
logistic.fit(train_data, train_target)
predict = logistic.predict(test_data)
print("ACC: "+str(logistic.score(test_data,test_target)))
