import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.signal.windows import hann

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_utils import WAV
import config

_CITATION = ""
_DESCRIPTION = ""

class DARPA_TIMIT(tfds.core.GeneratorBasedBuilder):

	VERSION = tfds.core.Version("1.0.0")

	def _info(self):
		return tfds.core.DatasetInfo(
			builder=self,
			description=_DESCRIPTION,
			features=tfds.features.FeaturesDict({
				"fft": tfds.features.Tensor(shape=(config.TENSOR_SHAPE,), dtype=tf.float32),
				"label": tfds.features.ClassLabel(num_classes=2),
			}),
			supervised_keys=("fft", "label"),
			citation=_CITATION,
		)

	def _split_generators(self, dl_manager):
		"""Returns SplitGenerators."""
		path = config.PREPROCESSED_DIR
		# There is no predefined train/val/test split for this dataset.
		return [
			tfds.core.SplitGenerator(
				name=tfds.Split.TRAIN,
				gen_kwargs={
					"path": os.path.join(path, "TRAIN")
				}
			),
			tfds.core.SplitGenerator(
				name=tfds.Split.TEST,
				gen_kwargs={
					"path": os.path.join(path, "TEST")
				}
			),
			tfds.core.SplitGenerator(
				name=tfds.Split.VALIDATION,
				gen_kwargs={
					"path": os.path.join(path, "VALIDATION")
				}
			),
		]

	def _generate_examples(self, path):
		"""Yields examples.
		Args:
		   path: Path of the downloaded and extracted directory
		Yields:
		   Next examples
		"""
		window = hann(config.FRAMESIZE)  # Hann window
		for root, _, file_name in tf.io.gfile.walk(path):
			for fname in file_name:
				if fname.endswith(".WAV"):
					wav = WAV(root, fname)
					dialect_region, speaker, fnameStripped = wav.getMetaData()
					metaData = '.'.join([dialect_region, speaker, fnameStripped])
					while wav.curFrame < wav.frameNum - 1:
						trackFrameNum = str(wav.curFrame).zfill(5)
						frame, label = wav.getNextFrame()
						frame = np.multiply(frame, window)
						psd = wav.getPsd(frame)
						psd.reshape((1, 513))
						key = '.'.join([metaData, trackFrameNum])
						example = {
							"fft": psd,
							"label": label,
						}
						yield key, example
