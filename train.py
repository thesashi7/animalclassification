from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from preparedata import FeatureLoader
from neuralnet import DenseNeuralNetwork
from preparedata import FeatureLoader
from fextractor import *

# Load data from CSV file. Edit this to point to the features file
#data, target = FeatureLoader().loadFeatures(["data/features.csv"])
data, target = FeatureLoader().loadFeatures(["data/train-x-py.csv","data/train-y-py.csv"])
# Split the data into two parts: training data and testing data
train_data, test_data, train_target, test_target = train_test_split(
                 data, (target[:, np.newaxis]), test_size=0.3,random_state=42)
train_target = np_utils.to_categorical(train_target, 2)
test_target = np_utils.to_categorical(test_target, 2)


dnn = DenseNeuralNetwork()
dnn.learn(train_data,train_target)
dnn.test(test_data,test_target)
dnn.save("pyAudio")