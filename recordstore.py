import sounddevice as sd
import numpy as np
import time
from python_speech_features import mfcc
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os
from pyAudioAnalysis import audioTrainTest as aT


dir_path = os.path.dirname(os.path.realpath(__file__))

class Recorder :

	def __init__(self, duration) :
		self.duration = duration
		self.sound = []
		self.mfccofinput = []
		self.fs = 16000

	def record(self) :
		sd.default.samplerate = self.fs
		sd.default.latency = 'high'
		sd.default.channels = 1
		self.sound = sd.rec(int(self.duration * self.fs),  blocking=True)
		# myrecording = sd.playrec(self.sound, self.fs, channels=2)
		# print (self.sound)
		np.save('input.npy',self.sound)
		write('test.wav', self.fs, self.sound)
		print self.sound

	def play(self) :
		sd.play(self.sound)
		sd.wait()

	def extract_features_from_input_voice(self) :
		[Fs, x] = audioBasicIO.readAudioFile("test.wav");
		F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
		np.save('features.npy',F)
		print F.shape
		# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel('ZCR'); 
		# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel('Energy'); plt.show()

	def train_svm(self, directory) :
		listOfDirs = []
		for i in os.listdir(directory) :
			if os.path.isdir(os.path.join(dir_path,directory,i)) :
				print os.path.join(dir_path,directory,i)
				listOfDirs.append(os.path.join(dir_path,directory,i))

		print listOfDirs

		aT.featureAndTrain(listOfDirs, 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmforemotions")

	def detect_emotion(self) :
		# audioAnalysisRecordAlsa.recordAnalyzeAudio(3, "record.wav", 3, "svmforemotions", "svm")
		Result, P, classNames = aT.fileClassification("neutral.wav", "svmforemotions", "svm")
		print Result
		print P
		print classNames






	# def find_mfcc_features(self) :
	# 	self.mfccofinput = mfcc(self.sound)
	# 	np.save('mfcc_features.npy',self.mfccofinput)




