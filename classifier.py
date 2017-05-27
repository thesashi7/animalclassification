from neuralnet import DenseNeuralNetwork
from cnn import ConvolutionNeuralNetwork
from fextractor import LibrosaExtractor
from fextractor import PyExtractor
from fextractor import FeatureExtractorManager
import numpy as np


#####################################################################
# Classifier to classify Species base on Vocalizations
#   Currently only support Dog and Cat
#
class SpeciesAudioClassifier:

    def __init__(self,classifier_type="dnn"):
        if classifier_type=="dnn":
            self.classifier = DenseNeuralNetwork()
        elif classifier_type=="cnn":
            self.classifier = ConvolutionNeuralNetwork()



    def classify(self, raw_audio, feature_extractor="librosa",features=""):

        extractor = FeatureExtractorManager()
        onlyMFCC = False
        if features=="mfcc":
            onlyMFCC = True
            # Need to used this to get model trained with only mfcc
            features= "-"+features
        if feature_extractor == "librosa":
            #load model trained with librosa features
            self.classifier.load("librosa"+features)
        else:
            #load model trained with PyAudio featuers
            self.classifier.load("pyAudio"+features)

        features = extractor.extractFromFile(feature_extractor,raw_audio,onlyMFCC)

        print features

        prediction = self.classifier.predict(features)
        prediction = prediction.argmax(1)
        return prediction

#
# Labels for cat and dog
cat = 1
dog  = 0
label= {cat:"Cat",dog:"Dog"}
audio_file="raw-test/cat1.wav"
clf = SpeciesAudioClassifier()
classific = clf.classify(audio_file,"librosa")
print "\nYou are a "+label[classific[0]]


