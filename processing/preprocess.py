import os
import shutil

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

copytree(config.RAW_DATA_ROOT_DIR, config.PREPROCESSED_DIR)
