import os

import numpy as np
import scipy.io.wavfile
from scipy import signal

import config


class WAV:

	def __init__(self, root, fname):
		self.root = root
		self.fname = fname
		self.fnameStripped = self.fname[:-4]
		self.srate, self.data, self.length, self.frameNum = self.openFile()
		self.voiceRanges = self.getVoiceRanges()
		self.curFrame = 0

	def openFile(self):
		srate, data = scipy.io.wavfile.read(os.path.join(self.root, self.fname))
		length = len(data)
		frameNum = int(np.floor(length / config.STEP)) - 1
		return srate, data, length, frameNum

	def getMetaData(self):
		subdirs = os.path.relpath(self.root, self.fname).split(os.sep)
		dialect_region = subdirs[5]
		speaker = subdirs[6]
		recording = self.fnameStripped
		return dialect_region, speaker, recording

	def getVoiceRanges(self):
		with open(os.path.join(self.root, (self.fnameStripped + '.WRD'))) as f:
			ranges = [tuple(map(int, i.split(' ')[:-1])) for i in f]
		return ranges

	def getFrame(self, frameNum):
		if frameNum in range(0, self.frameNum):
			frameStart = frameNum * config.STEP
			frame = self.data[np.arange(frameStart, min(frameStart + config.FRAMESIZE, self.length))]
			return frame
		else:
			return None

	def getFrameLabel(self, frameNum):
		if frameNum in range(0, self.frameNum):
			frameStart = frameNum * config.STEP
			label = 0
			if any(start <= frameStart + config.FRAMESIZE / 2 <= end for (start, end) in self.voiceRanges):  # TODO: remove hardcode
				label = 1
			return label
		else:
			return None # ???

	def getNextFrame(self):
		if self.curFrame in range(0, self.frameNum):
			frame = self.getFrame(self.curFrame)
			label = self.getFrameLabel(self.curFrame)
			self.curFrame = self.curFrame + 1
			return frame, label
		return None # ???

	def getSpectro(self, frame):
		paddedFrame = np.zeros(config.NFFT_SIZE)
		paddedFrame[0:config.FRAMESIZE] = frame
		f, t, Sxx = signal.spectrogram(paddedFrame, self.srate, config.WINDOW, nperseg=config.FRAMESIZE, nfft=config.NFFT_SIZE, noverlap=config.NFFT_OVERLAP)
		return f, t, Sxx

	def getLabels(self):
		labels = np.zeros(self.length)
		for i in range(0, self.length):
			if any(start <= i <= end for (start, end) in self.voiceRanges):
				labels[i] = 1
		return labels