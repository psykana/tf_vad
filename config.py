import glob
import os

FRAMESIZE = 600
OVERLAP = 300
STEP = FRAMESIZE - OVERLAP

NFFT_SIZE = 1024
TENSOR_SHAPE = int(NFFT_SIZE / 2) + 1

EPOCHS = 10
LEARNING_RATE = 0.001
ROUNDING_THRESHOLD = 0.7

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data", "raw", "darpa-timit-acousticphonetic-continuous-speech")
PREPROCESSED_DIR = os.path.join(ROOT_DIR, "data", "preprocessed")
INFERENCE_DIR = os.path.join(ROOT_DIR, "data", "inference")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

latest_model = max(glob.glob(os.path.join(MODELS_DIR, '*/')), key=os.path.getmtime)
INF_MODEL = latest_model
