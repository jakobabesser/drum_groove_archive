__author__ = 'Jakob Abesser'

import numpy as np
import glob
import os
import pickle

from tools import num_to_pat

if __name__ == '__main__':

    dir_data = '/Volumes/MINI/guitar_pro_data'
    dir_out = os.path.join(dir_data, '_all')
    Q = 16

    fn_all = os.path.join(dir_out, 'all_patterns_stacked')

    fn_list = glob.glob(os.path.join(dir_data, '*'))

    a = 2**np.arange(48)

    all_patterns = {}
    MIN = 8
    num_files = len(fn_list)
    all_patterns = []
    for f, fn in enumerate(fn_list):
        if f % 10 == 0:
            print('File {}/{}'.format(f+1, num_files))
        try:
            with open(fn, 'rb') as fh:
                score = pickle.load(fh)
            drum_score = score['score'][1]
            drum_score[0, :] = np.round(drum_score[0, :]*16)
            drum_score[0, :] -= np.min(drum_score[0, :])
            drum_score = drum_score.astype(int)
            max_bin = np.max(drum_score[0, :])
            drum_score_bin = np.zeros((3, max_bin+1), dtype=bool)
            drum_score_bin[drum_score[1, :], drum_score[0, :]] = True

            width = drum_score_bin.shape[1]
            new_width = int(np.ceil(width / Q)*Q)
            drum_score_bin = np.hstack((drum_score_bin, np.zeros((3, new_width-width), dtype=bool)))
            d = np.reshape(drum_score_bin, (16*3, int(new_width/16)))
            encoded = np.dot(a, d)
            all_patterns.append(encoded)
        except:
            pass

    all_patterns = np.concatenate(all_patterns)
    print('{} subsequent patterns '.format(len(all_patterns)))
    np.save(fn_all, all_patterns)
    with open(fn_all, 'wb+') as fh:
        pickle.dump(all_patterns, fh)

