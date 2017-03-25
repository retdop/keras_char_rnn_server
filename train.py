from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

from sample import sample, sample_chars
from utils import TextLoader, MAX_LEN

txt = TextLoader()

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape = (None, len(txt.chars))))
# input_shape=(MAX_LEN, len(txt.chars))))
model.add(Dense(len(txt.chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(txt.X, txt.y, batch_size=128, epochs=1)

    start_index = random.randint(0, len(txt.text) - MAX_LEN - 1)
    sample_chars(prime = txt.text[start_index: start_index + MAX_LEN], n_chars = 400, diversity = 0.5, txt = txt, model = model)

    model.save('data/nietzsche/nietzsche_' + str(iteration) + '.h5')
    print('model saved to ' + 'data/nietzsche/nietzsche_' + str(iteration) + '.h5')

import gc; gc.collect()
