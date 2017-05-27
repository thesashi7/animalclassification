from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adamax, Adam, SGD
from keras.utils import to_categorical
import keras.initializers
import numpy as np
import numpy
from sklearn.model_selection import train_test_split
from preparedata import FeatureLoader
from keras.models import model_from_json

class DenseNeuralNetwork:

    def __init__(self):
        self.model = None
        self.drp_rate = 0.1


    def learn(self,train_data,train_target, drp_rate=0.1, epochs=150, batch_size=40,
              num_class=2):

        # Trying to get consistent results but this is just the first step
        #   It looks like keras doesn't currenlty allow seeds to be initialized
        #   So every time there is new seed and random new weights so the results can be
        #       different on each run
        numpy.random.seed(0)

        self. model = Sequential()

        num_features = train_data.shape[1]
        # numpy.randomfrom keras.utils import to_categorical.seed(0)
        # Adding first dense layer with acitivation relu and dropout
        self.model.add(Dense(128, input_dim=num_features))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(drp_rate))
        # Adding second dense layer with acitivation relu and dropout
        self.model.add(Dense(156))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(drp_rate))

        self.model.add(Dense(156))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(drp_rate))

        self.model.add(Dense(228))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(drp_rate))

        self.model.add(Dense(228))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(drp_rate))

        self.model.add(Dense(128))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(drp_rate))

        # Adding final output layer with softmax
        self.model.add(Dense(units=num_class))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        self.model.fit(train_data, train_target, batch_size=batch_size, epochs=epochs)



    def test(self,test_data,test_target):
        score = self.model.evaluate(test_data, test_target, batch_size=10)
        print('\n')
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def save(self,name=""):
        f_name = "model-dnn"
        name = f_name+"-"+name
        model_json = self.model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(name+".h5")
        print("Saved model to disk")


    def load(self,name=""):
        f_name = "model-dnn"
        name = f_name +"-"+name
        # load json and create model
        json_file = open(name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(name+".h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
        self.model = loaded_model

    def predict(self,test_data):
        prediction = self.model.predict(test_data)
        return prediction



# Load data from CSV file. Edit this to point to the features file
#data, target = FeatureLoader("data/features.csv").loadLibrosaCSV()
fl = FeatureLoader()
fl.filepath = "data/train-x.csv"
data =  fl.loadPyCSV()
fl.filepath = "data/train-y.csv"
target =  fl.loadPyCSV()

#exit(1)
# Split the data into two parts: training data and testing data
train_data, test_data, train_target, test_target = train_test_split(
                 data, (target[:, np.newaxis]), test_size=0.3,random_state=42)
train_target = np_utils.to_categorical(train_target, 2)
test_target = np_utils.to_categorical(test_target, 2)



# serialize model to JSON
"""model_json = model.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model3.h5")
print("Saved model to disk")"""