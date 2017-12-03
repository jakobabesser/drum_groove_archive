__author__ = 'Jakob Abesser'

import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as pl


from tools import num_to_pat

if __name__ == '__main__':

    dir_data = '/Volumes/MINI/guitar_pro_data'
    dir_out = os.path.join(dir_data, '_all')
    Q = 16
    MIN_LEN = 8
    fn_all = os.path.join(dir_out, 'all_patterns')

    fn_stat = os.path.join(dir_out, 'pattern_statistic.txt')



    with open(fn_all, 'rb') as f:
        all_pats = pickle.load(f)

    pats = np.array(list(all_pats.keys()))
    freq = np.array(list(all_pats.values()))
    pats2D = [num_to_pat(np.array([_])) for _ in pats]
    len = np.array([np.sum(_) for _ in pats2D])

    mask = np.where(len > MIN_LEN)[0]
    pats = pats[mask]
    freq = freq[mask]
    pats2D = [pats2D[_] for _ in mask]
    all_idx = np.argsort(freq)[::-1]

    with open(fn_stat, 'w+') as f:
        for idx in all_idx:
            f.write('{},{}\n'.format(pats[idx], freq[idx]))

    OFFSET = 0
    X = 5
    Y = 8
    fs = 6
    pl.figure()
    yticklabels = ['BD', 'SD', 'HH']
    yticks = np.arange(3)
    xticks = np.arange(16)
    xticklabels = ['1', '', '.', '', '2', '', '.', '', '3', '', '.', '', '4', '', '.', '']
    for i in range(X*Y):
        idx = all_idx[i+OFFSET]
        pl.subplot(Y, X, i+1)
        pat = num_to_pat(np.array([pats[idx]]))
        pl.imshow(pat, interpolation="nearest", aspect="auto", cmap='Greys')
        pl.gca().invert_yaxis()
        pl.title('# = {}'.format(freq[idx]), fontsize=fs)
        pl.yticks(yticks, yticklabels, fontsize=fs)
        pl.xticks(xticks, xticklabels, fontsize=fs)
    pl.tight_layout()
    pl.savefig(os.path.join(dir_out, 'patterns.png'))

    # 48 bit

    # tst
    a = np.zeros((3, 16), dtype=bool)
    a[0, 0] = 1
    a[1, 8] = 1
    a[2, 8] = 1
    c = np.reshape(a, (48, 1))

    d = 2 ** np.arange(48)

    num = np.dot(d, c)

    pat = num_to_pat(num)

    print(np.sum(np.abs((a-pat))))


    a = 1