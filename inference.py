import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from processing.data_utils import WAV


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

wavList = [f for f in os.listdir(config.INFERENCE_DIR) if f.endswith('.WAV')]
if len(wavList) < 1:
  raise ValueError('No WAVs found')

# with tf.keras.utils.custom_object_scope({'Precision': tf.keras.metrics.Precision(), 'Recall': tf.keras.metrics.Recall()}):
model = tf.keras.models.load_model(config.INF_MODEL, custom_objects={'custom_f1': custom_f1})

for file in wavList:
  wav = WAV(config.INFERENCE_DIR, file)
  labels = wav.getLabels()

  predictions = np.zeros(wav.frameNum)
  while wav.curFrame < wav.frameNum-1:
    frame, label = wav.getNextFrame()
    t, f, Sxx = wav.getSpectro(frame)
    Sxx_tensor = Sxx[:, 0]
    Sxx_tensor = Sxx_tensor.reshape(1, config.TENSOR_SHAPE)
    prediction = model(Sxx_tensor)
    prediction = tf.cast(prediction, tf.float32)
    if prediction > config.ROUNDING_THRESHOLD:
      predictions[wav.curFrame] = 1
    else:
      predictions[wav.curFrame] = 0


  wav_norm = wav.data * 1.0 / (max(abs(wav.data)))
  plt.plot(wav_norm)
  plt.plot(labels, label="Labels")
  t = np.arange(0, wav.frameNum * config.OVERLAP, config.OVERLAP)
  plt.plot(t, predictions, label="predictions")
  plt.legend()
  plt.show()
