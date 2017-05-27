import sys
import csv
import json
from pyAudioAnalysis.audioFeatureExtraction import *
from pyAudioAnalysis.audioVisualization import *
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from preparedata import FeatureWriter
import librosa
import numpy as np

class FeatureExtractorManager:

    def __init__(self):
        self.extractors = {
          "librosa": LibrosaExtractor(),
          "pyAudio":PyExtractor()
        }


    def extractFromFile(self,extractor_name, new_file_path, onlyMFCC=False):
      features = []
      self.extractors[extractor_name].file_path  = new_file_path
      if onlyMFCC==True:
        features = self.extractors[extractor_name].extractMFCCFile()
      else:
        features  = self.extractors[extractor_name].extractFile()
      return features

    def extractFromFolder(self,extractor_name, new_folder_path,onlyMFCC=False):
      features = []
      self.extractors[extractor_name].folder_path = new_folder_path
      if onlyMFCC==True:
        features = self.extractors[extractor_name].extractMFCCFolder()
      else:
        features  = self.extractors[extractor_name].extractFolder()
      return features



class FeatureExtractor:


    def extractFile(self): raise NotImplementedError

    def extractFolder(self): raise NotImplementedError

    def extractMFCCFile(self): raise NotImplementedError

    def extractMFCCFolder(self): raise NotImplementedError


class LibrosaExtractor(FeatureExtractor):


    def __init__(self, new_file_path="",new_folder_path=""):
        self.file_path = new_file_path
        self.folder_path = new_folder_path

    def extract_feature(self, seconds=None):
      mfccs = []
      chroma = []
      mel = []
      contrast = []
      tonnetz = []

      X, sample_rate = librosa.load(self.file_path)
      start = 0
      if seconds is None:
        duration = len(X)
      else:
        duration = sample_rate * seconds
      while start < len(X):
        end = min(start + duration, len(X))

        stft = np.abs(librosa.stft(X[start:end]))
        mfccs.append(np.mean(librosa.feature.mfcc(y=X[start:end], sr=sample_rate, n_mfcc=40).T, axis=0))
        chroma.append(np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0))
        mel.append(np.mean(librosa.feature.melspectrogram(X[start:end], sr=sample_rate).T, axis=0))
        contrast.append(np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0))
        tonnetz.append(
          np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X[start:end]), sr=sample_rate).T, axis=0))

        start = end

      return mfccs, chroma, mel, contrast, tonnetz


    def merge_features(self, mfccs, chroma, mel, contrast, tonnetz):
      features = []
      for i in range(len(mfccs)):
        ext_features = np.hstack([mfccs[i], chroma[i], mel[i], contrast[i], tonnetz[i]])
        features.append(ext_features)
      return np.array(features)


    def label_audio_files(self, dir_name, label, file_ext='*.wav'):
      features, labels = np.empty((0, 193)), np.empty(0)
      for fn in glob.glob(os.path.join(dir_name, file_ext)):
        # label = os.path.basename(os.path.dirname(fn))
        print("Processing %s" % fn)
        mfccs, chroma, mel, contrast, tonnetz = self.extract_feature(fn)
        # mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn, 10)
        for i in range(len(mfccs)):
          ext_features = np.hstack([mfccs[i], chroma[i], mel[i], contrast[i], tonnetz[i]])
          features = np.vstack([features, ext_features])
          labels = np.hstack([labels, label])

      # header
      header = ["mfcc" for n in mfccs[0]]
      header += ["chroma_stft" for n in chroma[0]]
      header += ["melspectrogram" for n in mel[0]]
      header += ["spectral_contrast" for n in contrast[0]]
      header += ["tonnetz" for n in tonnetz[0]]

      return np.array(features), np.array(labels), np.array(header)


    def write_header(self,header, dir_name, file_name='header.csv'):
      fn = os.path.join(dir_name, file_name)
      with open(fn, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
      print("Header saved in %s" % fn)

    def write_features(self,features, labels, dir_name, file_name='features.csv'):
      fn = os.path.join(dir_name, file_name)
      with open(fn, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(features.shape[0]):
          writer.writerow(np.hstack((features[i], labels[i])))
      print("Features saved in %s" % fn)


    def load_features(self, dir_name, file_name='features.csv'):
      features, labels = [], [];
      with open(os.path.join(dir_name, file_name), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
          features.append(row[:-1])
          labels.append(row[-1])
      return np.array(features, dtype=np.float64), np.array(labels)


    def save_label(self, label_list, dir_name, file_name='labels.csv'):
      with open(os.path.join(dir_name, file_name), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(label_list)


    def load_label(self, dir_name, file_name='labels.csv'):
      label_list = []
      with open(os.path.join(dir_name, file_name), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
          label_list = np.hstack([label_list, row])
      label_numbers = [n for n in range(len(label_list))]
      label_dict = dict(zip(label_list, label_numbers))
      return label_list, label_dict


    def one_hot_encode(self, labels, num_classes=None):
      if num_classes is None:
        num_classes = np.max(labels) + 1

      one_hot_list = np.zeros((len(labels), num_classes), dtype=np.int)
      one_hot_list[np.arange(len(labels)), labels] = 1
      return np.array(one_hot_list)


    def one_hot_decode(self, one_hot_list):
      return np.array([n.argmax() for n in one_hot_list])


    def extractFile(self):
      mfccs, chroma, mel, contrast, tonnetz = self.extract_feature()
      return self.merge_features(mfccs, chroma, mel, contrast, tonnetz)


    def extractFolder(self,onlyMFCC=False):
      types = ('*.wav', '*.aif', '*.aiff', '*.mp3', '*.au')
      wavFilesList = []
      features = []
      for files in types:
        wavFilesList.extend(glob.glob(os.path.join(self.folder_path, files)))

      # wavFilesList = sorted(wavFilesList)
      print wavFilesList
      for files in wavFilesList:
        self.file_path = files
        if onlyMFCC==False:
          features.append(self.extractFile())
        else:
          features.append(self.extractMFCCFile())
      return features


    def extractMFCCFile(self,seconds=None):
      mfccs = []
      X, sample_rate = librosa.load(self.file_path)
      start = 0
      if seconds is None:
        duration = len(X)
      else:
        duration = sample_rate * seconds
      while start < len(X):
        end = min(start + duration, len(X))
        mfccs.append(np.mean(librosa.feature.mfcc(y=X[start:end], sr=sample_rate, n_mfcc=40).T, axis=0))
        start = end

      return mfccs


    def extractMFCCFolder(self):
      return self.extractFolder(onlyMFCC=True)



#######################################################################################################
#######################################################################################################
#
# FeatureExtractor uses pyAudioAnalysis library to extract audio features.
# Currently we will need to clone the library locally, it is not found as a python package yet.
# Dependecies pYAudioAnalysis, numpy, matplotlib, scipy, sklearn, hmmlearn, simplejson, eyed3, pydub
# Please visit the following links for more details about pyAudioAnalysis:
#		https://github.com/tyiannak/pyAudioAnalysis
#		http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610
#
#    The following audio features are evaluated and collected in pyAudioAnalysis
#    1			Zero Crossing Rate
#    2 			Energy
#    3 			Entropy of Energy
#    4 			Spectral Centroid
#    5 			Spectral Spread
#    6 			Spectral Entropy
#    7 			Spectral Flux
#    8 			Spectral Rolloff
#    9-21 	MFCCs
#    22-33 	Chroma Vector
#    34 		Chroma Deviation
#
class PyExtractor(FeatureExtractor):



  ##Parameter Tuning for feature extraction guys
  # Default is fine for now but tune it for fun [:)
  # 	BE A TUNER MY FRIEND
  ###################################################################
  #
  # Steps are shorter than the Window length or equal
  #@new_file_path: Full path of your data folder where wav files exist
  #@new_mtWin  : Mid-Term Window size usually between 1 to 10 seconds (In Seconds)
  #@new_mtStep :Mid-Term Step size (In Seconds)
  #@new_stWin  : Short-Term window size usually between 20 to 100 ms (In Seconds)
  #@new_stStep : Short-Term step size (In Seconds)
  #@new_computeBEAT : Boolean Value to compute BEAT of audio (This is for Music Audio)
  #
  def __init__(self, new_file_path="", new_folder_path="", new_mtWin=10, new_mtStep=1, new_stWin=.02, new_stStep=.01, new_computeBEAT=False):
    self.file_path = new_file_path
    self.folder_path = new_folder_path
    self.mtWin = new_mtWin
    self.mtStep = new_mtStep
    self.stWin = new_stWin
    self.stStep = new_stStep
    self.computeBEAT = new_computeBEAT

  #################################################################
  #@return feature vectors
  # Multidimenstional array for more than one wave file
  #
  def extractFolder(self):
    return dirWavFeatureExtraction(self.file_path, self.mtWin, self.mtStep, self.stWin, self.stStep, self.computeBEAT)[0]


  def extractFile(self):

    ft = mtFeatureExtractionToFile(self.file_path, self.mtWin, self.mtStep, self.stWin, self.stStep,"py-feat-temp", True, True, True)
    ft = ft.mean(axis=0)
    print ft.shape
    ft = ft.reshape((1, ft.shape[0]))
    return ft

  def visualize(self,label="NA",reduction_method="pca"):
    visualizeFeaturesFolder(self.folder_path,reduction_method,label)

  ##################################################################
  #@features: feature vectors
  #@file_name: Name of the csv file that you want to write the featues to
  #
  def write_csv(self,features,file_name="test-data"):
    with open(file_name+".csv", 'w') as csvfile:
      writer = csv.writer(csvfile,delimiter=',',quoting = csv.QUOTE_NONE)
      #print(type(features[0]))
      if(isinstance(features[0],numpy.ndarray)):
        for fv in features:
          writer.writerow(fv)
      else: #only one feature vector
        writer.writerow(features)


##
#(class)FeatureExtractor ENDS
##

#print "Main Running"
#folder_path = "/home/sashi/Documents/Spring2017/CS599/project/dog/"
#file_path =  folder_path+"21013_44k.wav"
#fl = []
#feM = FeatureExtractorManager()
#features = feM.extractFromFolder("librosa", folder_path,onlyMFCC=True)
#features = feM.extractFromFile("librosa", file_path,onlyMFCC=True)

#exit()
#fl.append(features)
#print features
#print fl
#fl = np.array(fl)
#print fl
#print fl.shape
#print fl.reshape((1,40))
#exit()
#features = np.array(features)
#features = features.reshape((features.shape[0],40))
#fw = FeatureWriter(features)
#fw.write_csv("mfcc-dog-feat")
#fw = FeatureWriter()
#fw.writeFromTwoPy("mfcc-cat-feat.csv","mfcc-dog-feat.csv")