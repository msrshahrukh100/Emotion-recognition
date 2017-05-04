import sounddevice as sd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from scipy.io.wavfile import write
# from svmutil import *
from sklearn import svm

dir_path = os.path.dirname(os.path.realpath(__file__))


def listOfFeatures2Matrix(features):
    '''
    listOfFeatures2Matrix(features)

    This function takes a list of feature matrices as argument and returns a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces
    '''

    X = np.array([])
    Y = np.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * np.ones((len(f), 1))
        else:
            X = np.vstack((X, f))
            Y = np.append(Y, i * np.ones((len(f), 1)))
    return (X, Y)


class FeatureExtractor :

	def __init__(self, duration) :
		self.duration = duration
		self.sound = []
		self.fs = 16000

	def record(self) :
		sd.default.samplerate = self.fs
		sd.default.latency = 'high'
		sd.default.channels = 1
		print "Recording ....."
		self.sound = sd.rec(int(self.duration * self.fs),  blocking=True)
		print "Recording Finished !"
		
		np.save('input.npy',self.sound)
		write('test.wav', self.fs, self.sound)
		print self.sound

	def play(self) :
		sd.play(self.sound)
		sd.wait()

	def find_mfcc(self,filename):
		(rate,sig) = wav.read(filename)
		mfcc_feat = mfcc(sig,rate)
		# print "MFCC Features : "
		# print mfcc_feat
		return mfcc_feat

	def find_energy_features(self,filename):
		(rate,sig) = wav.read(filename)
		fbank_feat = logfbank(sig,rate)
		# print "Mel-filterbank energy features : "
		# print fbank_feat[1:3,:]
		return fbank_feat[1:3,:]

	def read_data(self, directory) :
		features = []
		labels = []
		x = 0
		r = FeatureExtractor(2)
		for i in os.listdir(directory) :
			if os.path.isdir(os.path.join(dir_path,directory,i)) :
				# print os.path.join(dir_path,directory,i)
				# listOfDirs.append(os.path.join(dir_path,directory,i))
				foldername = i.split('.')
				label = foldername[0]
				subfolderpath = os.path.join(dir_path,directory,i)
				# print label
				# print subfolderpath
				for j in os.listdir(subfolderpath) :
					if os.path.isfile(os.path.join(subfolderpath,j)) :
						file_path = os.path.join(subfolderpath,j)
						print file_path
						feature = []
						feature.append(r.find_mfcc(file_path))
						feature.append(r.find_energy_features(file_path))
						# print feature
						features.append(feature)
						labels.append(label)
		features = np.asarray(features)
		np.save("features.npy",features)
		labels = np.asarray(labels,dtype=object)
		np.save("labels.npy",labels)
		

	def train_svm(self):

		features = np.load("features.npy")
		labels = np.load("labels.npy")

		# prob = svm_problem(labels,features)
		# param = svm_parameter('-t 0 -c 4 -b 1')
		# m = svm_train(prob,param)
		linearfeatures = listOfFeatures2Matrix(features)
		clf = svm.SVC(kernel = 'linear',  probability = True)
		clf.fit(linearfeatures,labels)

		# svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

		# prob = svm_problem(labels, features)

		# param = svm_parameter()
		# param.kernel_type = LINEAR
		# param.C = 10

		# m=svm_train(prob, param)
		# print features
		# print labels



	def detect_emotion(self, directory) :
		for i in os.listdir(directory) :
			if os.path.isdir(os.path.join(dir_path,directory,i)) :
				label = i
				subfolderpath = os.path.join(dir_path,directory,i)
				count = 0
				total = 0
				# print label
				# print subfolderpath
				for j in os.listdir(subfolderpath) :
					if os.path.isfile(os.path.join(subfolderpath,j)) :
						file_path = os.path.join(subfolderpath,j)
						# print file_path
						# Result, P, classNames = aT.fileClassification(file_path, "svmforemotions", "svm")
						print "Labelled Emotion is : " + label
						print "Detected emotion is : " + classNames[int(Result)]
						if label == classNames[int(Result)] :
							count+= 1
						total+= 1
				print "For Emotion " + label + " Correct : " + str(count) + " Total : " + str(total)
				print "Percentage : " + str(count * 100 / total)

