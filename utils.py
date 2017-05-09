from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os

MAX_LEN = 40

class TextLoader():
    def __init__(self):
        self.MAX_LEN = 40
        path = 'data/python/scikit_cleaned.txt'
        self.text = open(path).read()
        print('corpus length:', len(self.text))

        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # cut the text in semi-redundant sequences of MAX_LEN characters
        step = 3
        self.sentences = []
        self.next_chars = []
        for i in range(0, len(self.text) - MAX_LEN, step):
            self.sentences.append(self.text[i: i + MAX_LEN])
            self.next_chars.append(self.text[i + MAX_LEN])
        print('nb sequences:', len(self.sentences))

        print('Vectorization...')
        self.X = np.zeros((len(self.sentences), MAX_LEN, len(self.chars)), dtype=np.bool)
        self.y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                self.X[i, t, self.char_indices[char]] = 1
            self.y[i, self.char_indices[self.next_chars[i]]] = 1
