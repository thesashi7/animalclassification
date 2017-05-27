from sklearn import svm
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from preparedata import FeatureLoader



# Load data from CSV file. Edit this to point to the features file
data, target = FeatureLoader("data/features.csv").loadLibrosaCSV()

# Split the data into two parts: training data and testing data
train_data, test_data, train_target, test_target = train_test_split(
                 data, (target[:, np.newaxis]), test_size=0.2,random_state=42)

svmModel = svm.SVC()
svmModel.fit(train_data, train_target)
predict = svmModel.predict(test_data)

print("Loss: "+str(log_loss(test_target,predict)))
print("Accuracy: "+str(accuracy_score(test_target, predict)))