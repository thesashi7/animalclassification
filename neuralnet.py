
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

# Load data from CSV file. Edit this to point to the features file
data, target = FeatureLoader("data/features.csv").loadLibrosaCSV()

#exit(1)
# Split the data into two parts: training data and testing data
train_data, test_data, train_target, test_target = train_test_split(
                 data, (target[:, np.newaxis]), test_size=0.3,random_state=42)
train_target = np_utils.to_categorical(train_target, 2)
test_target = np_utils.to_categorical(test_target, 2)

# Trying to get consistent results but this is just the first step
#   It looks like keras doesn't currenlty allow seeds to be initialized
#   So every time there is new seed and random new weights so the results can be
#       different on each run
numpy.random.seed(0)

model = Sequential()
drp_rate = 0.1
num_features = 193
#numpy.randomfrom keras.utils import to_categorical.seed(0)
#Adding first dense layer with acitivation relu and dropout
model.add(Dense(128, input_dim=num_features))
model.add(Activation('relu'))
model.add(Dropout(drp_rate))
#Adding second dense layer with acitivation relu and dropout
model.add(Dense(156))
model.add(Activation('sigmoid'))
model.add(Dropout(drp_rate))

model.add(Dense(156))
model.add(Activation('sigmoid'))
model.add(Dropout(drp_rate))

model.add(Dense(228))
model.add(Activation('tanh'))
model.add(Dropout(drp_rate))

model.add(Dense(228))
model.add(Activation('tanh'))
model.add(Dropout(drp_rate))

model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(drp_rate))



#Adding final output layer with softmax
model.add(Dense(units=2))
model.add(Activation('softmax'))

#compiling model with Adadelta
model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
model.fit(train_data, train_target, batch_size = 40, epochs=150)

score = model.evaluate(test_data, test_target, batch_size=10)
print('\n')
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# serialize model to JSON
"""model_json = model.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model3.h5")
print("Saved model to disk")"""