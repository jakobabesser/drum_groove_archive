__author__ = 'Jakob Abesser'

import numpy as np
import glob
import os
import pickle

if __name__ == '__main__':

    dir_data = '/Volumes/MINI/guitar_pro_data'

    Q = 16

    fn_list = glob.glob(os.path.join(dir_data, '*'))

    a = 2**np.arange(48)

    for f in fn_list:
        with open(f, 'rb') as f:
            score = pickle.load(f)
        drum_score = score['score'][1]
        drum_score[0, :] = np.round(drum_score[0, :]*16)
        drum_score[0, :] -= np.min(drum_score[0, :])
        drum_score = drum_score.astype(int)
        max_bin = np.max(drum_score[0, :])
        drum_score_bin = np.zeros((3, max_bin+1), dtype=bool)
        drum_score_bin[drum_score[1, :], drum_score[0,:]] = True

        width = drum_score_bin.shape[1]
        new_width = int(np.ceil(width / Q)*Q)
        drum_score_bin = np.hstack((drum_score_bin, np.zeros((3, new_width-width), dtype=bool)))
        d = np.reshape(drum_score_bin, (16*3, int(new_width/16)))
        print(d.shape)

        # extend it to width (multiple of 16)
        # reshape to width 48
        # convert to int
        # np.unique -> count -> if >1 -> save number -> use dict (num = key) count = val
        #

        a = 1
    pass
