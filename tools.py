__author__ = 'Jakob Abesser'

import numpy as np

def num_to_pat(num):
    cc = np.unpackbits(np.array(num, dtype='>i8').view(np.uint8)).astype(bool)[::-1][:48][:, np.newaxis]
    return np.reshape(cc, (3, 16)).astype(bool)