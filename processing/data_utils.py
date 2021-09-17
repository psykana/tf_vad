import os

import numpy as np
import scipy.io.wavfile
from scipy.fft import rfft
from scipy.signal.windows import hann

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
		psd = abs(bins)
		return psd

	def getLabels(self):
		labels = np.zeros(self.length)
		for i in range(0, self.length):
			if any(start <= i <= end for (start, end) in self.voiceRanges):
				labels[i] = 1
		return labels


def dumpModel(filename, model, path=None):
	window = hann(config.FRAMESIZE)
	if path == None:
			path = config.ROOT_DIR
	dimensions = " "
	with open(os.path.join(path, filename), "w") as file:
		file.write('#ifdef __cplusplus\n'
							'extern "C" {\n'
							'#endif\n\n')
		file.write('static const int N = 1024;\n'
								'static const int LAYERS = 4;\n'
							 'static const int numBins = 513;\n\n')
		for i in range(len(model.layers)):
			weights, biases = model.layers[i].get_weights()
			weights = np.transpose(weights)
			input_size, output_size = weights.shape
			dimensions = dimensions + str(input_size) + ", "
			if (input_size) > 1:
				file.write('static const float weights{num}[][{len}] = {{\n'.format(num=i, len=input_size))
			else:
				file.write('static const float weights{num}[] = {{\n'.format(num=i))
			np.savetxt(file, weights.astype('float'), fmt='%f,')
			file.write('}};\n'
										'static const unsigned int weights{num}_xlen = {data_xlen};\n'
                    'static const unsigned int weights{num}_ylen = {data_ylen};\n\n'
                    .format(num=i, data_xlen=input_size, data_ylen=output_size)
				)
			file.write('static const float biases{num}[] = {{\n'.format(num=i))
			np.savetxt(file, biases.astype('float'), fmt='\t%f,')
			file.write('}};\n'
									'static const unsigned int biases{num}_len = {len};\n\n'
									.format(num=i, len=len(biases)))
		file.write('static const float hannWindow[] = {\n')
		np.savetxt(file, window.astype('float'), fmt='\t%f,')
		file.write('}};\n'
							 'static const unsigned int hannWindow_len = {len};\n\n'
							 .format(len=len(biases)))
		file.write('static const float dimensions[] = {{{layer_dimensions}}};\n\n'.format(layer_dimensions=dimensions))
		file.write('#ifdef __cplusplus\n'
								'}\n'
								'#endif\n')