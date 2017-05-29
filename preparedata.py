from __future__ import print_function

import csv
import numpy as np
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split

#
# Labels for cat and dog
cat = 1
dog  = 0

##############################################################################
# FeatureLoader class to load csv audio features
#   Loads audio features for species classification extracted from mainly
#        two different extractors; PyAudio and Librosa
#
##############################################################################
class FeatureLoader:

    def __init__(self,newFilepath=[]):
        self.filepath = newFilepath


    ####################################################################################
    # Function to load features.csv file
    # These csv files must contain audio features extracted using librosa
    #   along with the label at the end colum
    #
    def loadLibrosaCSV(self):
        file = open(self.filepath, 'r')  # Open file
        csv_file = csv.reader(file)  # Create CSV reader
        data = list()  # Create empty Data list
        target = list()  # Create empty target list

        for row in csv_file:  # Rows contain target split into two
            if (len(row) == 0):  # Skip empty rows
                continue
            data.append(row[:len(row) - 1])
            if (row[len(row) - 1] == 'cat'):
                target.append(cat)
            else:
                target.append(dog)
                # target.append(row[len(row) - 1])

        return np.asarray(data, dtype=np.float64), np.asarray(target,
                                                                  dtype=np.int64)  # Return tuple containing

    ####################################################################################
    # Function to load train-x.csv or train-y.csv
    # These csv files must contain audio features extracted using pyaudioanalysis
    #   or the target labels
    #
    def loadPyCSV(self):
        return np.genfromtxt(self.filepath, delimiter=',')

    ###################################################################################
    #@files : Two files containing audio features for classification
    #         Ideally one file is for only features and the other is for only label
    #
    def loadFeatures(self, files=[]):
        data = []
        target = []
        if(len(files)==1):
            self.filepath = files[0]
            data, target = self.loadLibrosaCSV()
        elif(len(files)==2):
            self.filepath = files[0]
            data = self.loadPyCSV()
            self.filepath = files[1]
            target = self.loadPyCSV()
        return data,target


#######################################################################################
#
# FeatureWriter class to write features to a CSV file
#   Especially supports combining features of two different classes like cat,dog
#
#       Like:   Input  = cat.csv, dog.csv
#               Output = train-x.csv,train-y.csv
#   In addition provides the option of creating different training and test dat for input features
#   See writeTrainAndTestFromTwoPy(...) for more details
#
class FeatureWriter:

    def __init__(self, newFeatures=[]):
        self.features = newFeatures

    ################################################################################
    # @file_name: Name of the csv file that you want to write the featues to
    #
    def write_csv(self, file_name="test-data"):
        with open(file_name + ".csv", 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            # print(type(features[0]))
            if (isinstance(self.features[0], np.ndarray)):
                for fv in self.features:
                    print(fv)
                    writer.writerow(fv)
            else:  # only one feature vector
                writer.writerow(self.features)

    #################################################################################
    # This is to combine training data of cats and dogs audio features extracted
    #       using PyAudioAnalysis or cat,dog features without label
    #
    # @field1 : cat audio feature csv file
    # @field1 : dog audio feature csv file
    #
    def writeFromTwoPy(self, file1,file2):
        cat_data = np.genfromtxt(file1,delimiter=',')
        dog_data = np.genfromtxt(file2,delimiter=',')
        print(cat_data.shape)
        print(dog_data.shape)
        cat_label = np.ones([cat_data.shape[0],1])
        dog_label = np.zeros([dog_data.shape[0],1])
        cat_data = np.append(cat_data, cat_label, 1)
        dog_data = np.append(dog_data, dog_label, 1)
        train_data = np.concatenate((cat_data,dog_data))
        np.random.shuffle(train_data)
        print(train_data.shape)
        train_y = np.empty([train_data.shape[0],1])
        i=0
        while i < train_data.shape[0]:
            train_y[i]=[train_data[i][train_data.shape[1]-1]]
            i+=1
        train_data = np.delete(train_data, np.s_[train_data.shape[1]-1:train_data.shape[1]], axis=1)
        print(train_data.shape)
        self.features = train_data
        self.write_csv( "data/train-x")
        self.features = train_y
        self.write_csv("data/train-y")

    #################################################################################
    # This is to combine training data of cats and dogs audio features extracted
    #       using PyAudioAnalysis
    # In addition creates separate training and testing csv files
    #
    #   Example:   Input  = cat.csv, dog.csv
    #               Output = train-x.csv, train-y.csv, test-x.csv, test-y.csv
    #
    # @field1 : cat audio feature csv file
    # @field1 : dog audio feature csv file
    #
    def writeTrainAndTestFromTwoPy(self,file1,file2, name="high"):
        cat_data = np.genfromtxt(file1, delimiter=',')
        dog_data = np.genfromtxt(file2, delimiter=',')
        print(cat_data.shape)
        print(dog_data.shape)
        cat_label = np.ones([cat_data.shape[0],1])
        dog_label = np.zeros([dog_data.shape[0],1])
        cat_data = np.append(cat_data, cat_label, 1)
        dog_data = np.append(dog_data, dog_label, 1)
        data = np.concatenate((cat_data, dog_data))
        np.random.shuffle(data)
        target = np.empty([data.shape[0], 1])
        i = 0
        while i < data.shape[0]:
            target[i] = data[i][data.shape[1] - 1]
            i += 1
        data = np.delete(data, np.s_[data.shape[1] - 1: data.shape[1]], axis=1)
        # Split the data into two parts: training data and testing data
        train_data, test_data, train_target, test_target = train_test_split(
            data, (target[:, np.newaxis]), test_size=0.2, random_state=42)
        test_target = test_target.reshape((test_target.shape[0],1))
        train_target = train_target.reshape((train_target.shape[0], 1))

        self.features = train_data
        self.write_csv("data/"+name+"-train-x")
        self.features = train_target
        self.write_csv("data/"+name+"-train-y")
        self.features = test_data
        self.write_csv("data/"+name+"-test-x")
        self.features = test_target
        self.write_csv("data/"+name+"-test-y")


#combineTrainData()
#createTrainAndTestData()