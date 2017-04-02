import sounddevice as sd
import numpy as np
import time
from python_speech_features import mfcc

class Recorder :

	def __init__(self, duration) :
		self.duration = duration
		self.sound = []
		self.mfccofinput = []

	def record(self) :
		fs = 48000
		sd.default.samplerate = fs
		sd.default.latency = 'high'
		sd.default.channels = 2
		self.sound = sd.rec(self.duration * fs,  blocking=True)
		# print (self.sound)
		np.save('input.npy',self.sound)

	def play(self) :
		sd.play(self.sound)
		sd.wait()

	# def find_mfcc_features(self) :
	# 	self.mfccofinput = mfcc(self.sound)
	# 	np.save('mfcc_features.npy',self.mfccofinput)



