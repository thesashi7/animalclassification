import csv
import numpy as np

class FeatureLoader:

    def __init__(self,newFilepath):
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
                target.append(1)
            else:
                target.append(0)
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


class FeatureWriter:

    def __init__(self, newFeatures):
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
                    writer.writerow(fv)
            else:  # only one feature vector
                writer.writerow(self.features)

    #################################################################################
    # This is to combine training data of cats and dogs audio features extracted
    #       using PyAudioAnalysis
    #
    # @field1 : cat audio feature csv file
    # @field1 : dog audio feature csv file
    #
    def writeFromTwoPy(self, file1,file2):
        cat_data = np.genfromtxt(file1,delimiter=',')
        dog_data = np.genfromtxt(file2,delimiter=',')
        print cat_data.shape
        print dog_data.shape
        cat_label = np.ones([cat_data.shape[0],1])
        dog_label = np.zeros([dog_data.shape[0],1])
        cat_data = np.append(cat_data, cat_label, 1)
        dog_data = np.append(dog_data, dog_label, 1)
        train_data = np.concatenate((cat_data,dog_data))
        np.random.shuffle(train_data)
        print train_data.shape
        train_y = np.empty([train_data.shape[0],1])
        i=0
        while i < train_data.shape[0]:
            train_y[i]=[train_data[i][68]]
            i+=1
        train_data = np.delete(train_data, np.s_[68:69], axis=1)
        print train_data.shape
        self.features = train_data
        self.write_csv( "data/train-x")
        self.features = train_y
        self.write_csv("data/train-y")

    #################################################################################
    # This is to combine training data of cats and dogs audio features extracted
    #       using PyAudioAnalysis
    # In addition creates separate training and testing data and target csv files
    #
    # @field1 : cat audio feature csv file
    # @field1 : dog audio feature csv file
    #
    def writeTrainAndTestFromTwoPy(self,file1,file2):
        cat_data = np.genfromtxt(file1, delimiter=',')
        dog_data = np.genfromtxt(file2, delimiter=',')
        print cat_data.shape
        print dog_data.shape
        cat_label = np.ones([cat_data.shape[0],1])
        dog_label = np.zeros([dog_data.shape[0],1])
        cat_data = np.append(cat_data, cat_label, 1)
        dog_data = np.append(dog_data, dog_label, 1)
        train_data = np.concatenate((cat_data[:109],dog_data[:203]))
        test_data = np.concatenate((cat_data[109:],dog_data[203:]))
        print train_data.shape
        print test_data.shape
        #np.random.shuffle(train_data)
        train_y = np.empty([train_data.shape[0],1])
        test_y = np.empty([test_data.shape[0],1])
        i=0
        while i < train_data.shape[0]:
            train_y[i]=[train_data[i][68]]
            i+=1
        i=0
        while i < test_data.shape[0]:
            test_y[i]=[test_data[i][68]]
            i+=1
        train_data = np.delete(train_data, np.s_[68:69], axis=1)
        test_data = np.delete(test_data, np.s_[68:69], axis=1)
        print train_data.shape
        print test_data.shape
        print test_y.shape
        print train_y.shape
        self.features = train_data
        self.write_csv("data/train-x")
        self.features = train_y
        self.write_csv("data/train-y")
        self.features = test_data
        self.write_csv("data/test-x")
        self.features = test_y
        self.write_csv("data/test-y")

#combineTrainData()
#createTrainAndTestData()