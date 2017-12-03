__author__ = 'Jakob Abesser'

import numpy as np
import glob
import os
import time
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tools import num_to_pat
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

if __name__ == '__main__':

    dir_data = '/Volumes/MINI/guitar_pro_data'
    dir_out = os.path.join(dir_data, '_all')
    Q = 16

    fn_all = os.path.join(dir_out, 'all_patterns_stacked')

    seq = np.load(fn_all+'.npy')
    # 8.8 mio -> 0.1 mio
    seq = seq[:10000]

    print(len(seq))

    extract = True
    fn_u = os.path.join(dir_out, 'useq')
    fn_uc = os.path.join(dir_out, 'useq_count')
    if extract:
        useq = np.unique(seq)
        print('{} unique patterns'.format(len(np.unique(seq))))
        useq_count = np.zeros_like(useq)
        num_useq = len(useq_count)
        for u in range(num_useq):
            if u % 50 == 0:
                print('{}/{}'.format(u, num_useq))
            useq_count[u] = np.sum(seq == useq[u])
        np.save(fn_u, useq)
        np.save(fn_uc, useq_count)
    else:
        useq = np.load(fn_u + '.npy')
        useq_count = np.load(fn_uc + '.npy')


    # raise Exception
    a = 1

    # integer encode
    t = time.time()
    label_encoder = LabelEncoder().fit(useq)
    print('Took {} s'.format(time.time() - t))
    t = time.time()
    seq_enc = label_encoder.transform(seq)
    print('Took {} s'.format(time.time() - t))

    vocab_size = len(useq) + 1
    print('Vocabulary size = {}'.format(vocab_size))




    embedding_size = 200
    input_length = 4 # bars (context)

    max_length = 4

    # ! convert long string to shorter strings
    # len_seg = 100 # bars
    # N = int(np.floor(len(seq) / len_seg))
    # seq_enc_seg = np.reshape(seq_enc[:N], (N, len_seg))

    seq_len = 10
    total_len = len(seq_enc)
    dataX = []
    dataY = []
    for i in range(0, total_len-seq_len, 1):
        seq_in = seq_enc[i:i + seq_len]
        seq_out = seq_enc[i + seq_len]
        dataX.append(seq_in)
        dataY.append(seq_out)
    n_patterns = len(dataX)

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_len, 1))
    # normalize
    X = X / float(vocab_size)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    print(y.shape)
    print(X.shape)


    print(np.sum(y, axis=0))
    # https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/



    DO_TRAIN = True
    n_patterns = len(dataX)

    fn_model = os.path.join(dir_out, 'model.h5')



    if DO_TRAIN:
        num_classes = y.shape[1]
        num = np.sum(y, axis=0)
        class_weight = {i: 1/num[i] for i in range(num_classes)}
        # print(class_weight)

        model = Sequential()
        model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        filepath = os.path.join(dir_out, "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
        model.fit(X, y, epochs=100, batch_size=64, callbacks=callbacks_list, class_weight=class_weight)
        print('save model')
        model.save(fn_model)
    else:
        print('load model')
        model = load_model(fn_model)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    # now finally generate some drums
    seed_enc = label_encoder.transform([seq[123]])

    pattern = seq_enc[3000:3100]

    # random seed
    pattern = dataX[1560]

    for i in range(40):
        # X = np.reshape(dataX, (n_patterns, seq_len, 1))
        x = np.reshape(pattern[-seq_len:], (1, seq_len, 1))
        x = x / float(vocab_size)
        print(pattern)
        prediction = model.predict(x, verbose=0)
        print(prediction[:3])
        index = np.argmax(prediction)
        # result = int_to_char[index]
        # seq_in = [int_to_char[value] for value in pattern]
        # sys.stdout.write(result)
        pattern = np.append(pattern, index)
        # print(index)
        # pattern = pattern[1:len(pattern)]

    pattern = label_encoder.inverse_transform(pattern)

    print(pattern)

    import matplotlib.pyplot as pl

    X = 5
    Y = 8
    fs = 6
    pl.figure()
    yticklabels = ['BD', 'SD', 'HH']
    yticks = np.arange(3)
    xticks = np.arange(16)
    xticklabels = ['1', '', '.', '', '2', '', '.', '', '3', '', '.', '', '4', '', '.', '']
    for i in range(X * Y):
        pl.subplot(Y, X, i + 1)
        pat = num_to_pat(np.array([pattern[i]]))
        pl.imshow(pat, interpolation="nearest", aspect="auto", cmap='Greys')
        pl.gca().invert_yaxis()
        pl.title('# {}'.format(i), fontsize=fs)
        pl.yticks(yticks, yticklabels, fontsize=fs)
        pl.xticks(xticks, xticklabels, fontsize=fs)
    pl.tight_layout()
    pl.savefig(os.path.join(dir_out, 'patterns_prediction.png'))



    print('Done :)')

    # TODOS
    # try embedding layer
    # try larger RNN (https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

    # problem: lernt fast nur 0 klasse