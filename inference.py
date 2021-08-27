import os
import time
start_time = time.time()

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal.windows import hann

from matplotlib.widgets import Slider

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
wavList = sorted(wavList)

# with tf.keras.utils.custom_object_scope({'Precision': tf.keras.metrics.Precision(), 'Recall': tf.keras.metrics.Recall()}):
model = tf.keras.models.load_model(config.INF_MODEL, custom_objects={'custom_f1': custom_f1})

print("Init: --- %s seconds ---" % (time.time() - start_time))
try:
    os.remove('predictions.txt')
except OSError:
    pass

window = hann(config.FRAMESIZE)
for file in wavList:
    start_time = time.time()
    print(file + ": ", end="")
    wav = WAV(config.INFERENCE_DIR, file)
    labels = wav.getLabels()
    predictions = np.zeros(wav.frameNum, dtype='float32')
    while wav.curFrame < wav.frameNum - 1:
        frame, label = wav.getNextFrame()
        frame = np.multiply(frame, window)
        psd = wav.getPsd(frame)
        psd = psd.reshape((1, config.TENSOR_SHAPE))
        predictions[wav.curFrame] = model(psd)

    print("--- %s seconds ---" % (time.time() - start_time))

    fig, ax = plt.subplots()
    wav_norm = wav.data * 1.0 / (max(abs(wav.data)))
    t = np.arange(0, wav.frameNum * config.OVERLAP, config.OVERLAP)
    line3, = plt.plot(wav_norm)
    line2, = plt.plot(labels, label="Labels")
    line1, = plt.plot(t, predictions, label="Predictions")

    plt.subplots_adjust(bottom=0.2)
    sliderax = plt.axes([0.25, 0.1, 0.65, 0.03])
    rounding_slider = Slider(
        ax=sliderax,
        label='Rounding threshold',
        valmin=0.0,
        valmax=1.0,
        valinit=config.ROUNDING_THRESHOLD
    )

    def update(val):
        length = len(predictions)
        res = np.zeros(length)
        for i in range(length):
            if predictions[i] > val:
                res[i] = 1
            else:
                res[i] = 0
        line1.set_ydata(res)
        fig.canvas.draw_idle()

    update(config.ROUNDING_THRESHOLD)

    rounding_slider.on_changed(update)

    plt.legend()
    plt.show()

    with open("predictions.txt", 'a') as out:
        np.savetxt(out, predictions, fmt='%.4e', encoding='bytes')
