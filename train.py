import datetime

import keras.backend as K
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

import config


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

ds_train = tfds.load('darpa_timit:1.0.0', split='train', as_supervised=True, batch_size=64)
ds_test = tfds.load('darpa_timit:1.0.0', split='test', as_supervised=True, batch_size=64)
ds_val = tfds.load('darpa_timit:1.0.0', split='validation', as_supervised=True, batch_size=64)
ds_train = ds_train.prefetch(128)

model = keras.Sequential(
    [
        layers.InputLayer(513),
        layers.Dense(513, activation="relu", name="layer1"),
        layers.Dense(513, activation="relu", name="layer2"),
        layers.Dense(513, activation="relu", name="layer3"),
        layers.Dense(1, activation="sigmoid", name="layer4"),
    ]
)

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
              metrics=[custom_f1])

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + timestamp
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#tf.debugging.set_log_device_placement(True)

history = model.fit(
    ds_train,
    epochs=config.EPOCHS,
    callbacks=[tensorboard_callback],
    shuffle=True,
    validation_data=ds_val
)

test_scores = model.evaluate(ds_test)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

model_dir = config.MODELS_DIR + "/" + timestamp
model.save(model_dir)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(model_dir + "/" + 'model.tflite', 'wb') as f:
    f.write(tflite_model)
