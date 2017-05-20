import csv
import numpy as np



##################################################################
# @features: feature vectors
# @file_name: Name of the csv file that you want to write the featues to
#
def write_csv(features, file_name="test-data"):
    with open(file_name + ".csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        # print(type(features[0]))
        if (isinstance(features[0], np.ndarray)):
            for fv in features:
                writer.writerow(fv)
        else:  # only one feature vector
            writer.writerow(features)

# This is to combine training data of cats and dogs as initially they are separate
def combineTrainData():
    cat_data = np.genfromtxt('/home/sashi/Documents/Spring2017/CS599/project/fex/cat-feat-2.csv',delimiter=',')
    dog_data = np.genfromtxt('/home/sashi/Documents/Spring2017/CS599/project/fex/dog-feat.csv',delimiter=',')
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

    write_csv(train_data, "/home/sashi/Documents/Spring2017/CS599/project/train-data")
    write_csv(train_y, "/home/sashi/Documents/Spring2017/CS599/project/test-data")


combineTrainData()