import os
import shutil

import numpy as np
import scipy.io.wavfile

import config


def copytree(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                ext = s.split('.', 1)
                if ext[1] == "WAV.wav":
                    shutil.copy2(s, d[:-4])
                elif ext[1] == "WRD":
                    shutil.copy2(s, d)


def pcm2float(sig, dtype='float32'):  # https://github.com/mgeier/python-audio
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


copytree(config.RAW_DATA_ROOT_DIR, config.PREPROCESSED_DIR)

for root, _, file_name in os.walk(config.PREPROCESSED_DIR):
    print(root)
    for fname in file_name:
        if fname.endswith(".WAV"):
            file = os.path.join(root, fname)
            srate, data = scipy.io.wavfile.read(file)
            data = pcm2float(data, dtype='float32')
            scipy.io.wavfile.write(file, srate, data)
