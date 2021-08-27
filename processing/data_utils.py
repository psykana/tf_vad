import os

import numpy as np
import scipy.io.wavfile
from scipy.fft import rfft

import config


class WAV:

	def __init__(self, root, fname):
		self.root = root
		self.fname = fname
		self.fnameStripped = self.fname[:-4]
		self.srate, self.data, self.length, self.frameNum = self.openFile()
		self.voiceRanges = self.getVoiceRanges()
		self.curFrame = -1

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
		if self.curFrame in range(-1, self.frameNum):
			self.curFrame = self.curFrame + 1
			frame = self.getFrame(self.curFrame)
			label = self.getFrameLabel(self.curFrame)
			return frame, label
		return None # ???

	def getPsd(self, frame):
		paddedFrame = np.zeros(config.NFFT_SIZE, dtype='float32')
		paddedFrame[0:config.FRAMESIZE] = frame
		bins = rfft(paddedFrame)
		psd = np.square(abs(bins))
		return psd

	def getLabels(self):
		labels = np.zeros(self.length)
		for i in range(0, self.length):
			if any(start <= i <= end for (start, end) in self.voiceRanges):
				labels[i] = 1
		return labels
