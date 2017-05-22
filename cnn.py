# from keras.models import Sequential
from keras.utils import np_utils
# from keras.layers.core import Dense, Activation, Dropout

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, Activation, convolutional, pooling, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adamax, Adam, SGD
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from preparedata import FeatureLoader


# Load data from CSV file. Edit this to point to the features file
data, target = FeatureLoader("data/features.csv").loadLibrosaCSV()

# Split the data into two parts: training data and testing data
train_data, test_data, train_target, test_target = train_test_split(
                 data, (target[:, np.newaxis]), test_size=0.2,random_state=42)
train_target = np_utils.to_categorical(train_target, 2)
test_target = np_utils.to_categorical(test_target, 2)
train_data = train_data.reshape(train_data.shape[0],1,193,1)
test_data = test_data.reshape(test_data.shape[0],1,193,1)


model = Sequential()

d_rate = 0.5
#Adding zero padding filter as the first input layer

#Adding first convolution layer with dropout and maxpooling with activation relu
model.add(
    convolutional.Conv2D(filters=32, kernel_size=(3,3),strides=(1, 1),padding='same',activation='relu'
                         ,input_shape=(1,193,1)))
#model.add(Dropout(0.2))
model.add(pooling.MaxPooling2D(pool_size=(2, 2),padding='same'))

#Adding second convolution layer with activation relu
model.add(convolutional.Conv2D(48,(5,5),padding='same'))
model.add(Activation('relu'))



#Adding third convolution layer with activation relu and dropout
model.add(convolutional.Conv2D(48,(5,5),padding='same'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))


#Adding fourth convolution layer with activation relu and dropout
model.add(convolutional.Conv2D(64,(5,5),padding='same'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))


#Adding fifth convolution layer with activation relu and dropout
model.add(convolutional.Conv2D(64,(5,5),padding='same'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))


#model.add(pooling.MaxPooling2D(pool_size=(2,2),padding='same'))
#model.add(convolutional.Conv2D(48,(5,5),padding='same'))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
#model.add(pooling.MaxPooling2D(pool_size=(2,2),padding='same'))

# Adding flatten layer and fully connected layer or deep neural network
model.add(Flatten())
#Adding first dense layer with acitivation relu and dropout
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(d_rate))
#Adding second dense layer with acitivation relu and dropout
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(d_rate))

model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(d_rate))



#Adding final output layer with softmax
model.add(Dense(units=2))
model.add(Activation('softmax'))

#compiling model with Adadelta
model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
model.fit(train_data, train_target, batch_size = 40, epochs=100)

score = model.evaluate(test_data, test_target, batch_size=10)
print('\n')
print('Test loss:', score[0])
print('Test accuracy:', score[1])