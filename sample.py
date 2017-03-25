from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

from utils import TextLoader, MAX_LEN

def main():
    txt = TextLoader()
    print('Loading model...')
    model = load_model('data/nietzsche/nietzsche_1.h5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def sample_chars(prime = 'The ', n_chars = 20, diversity = 0.5, txt = None, model = None):
    # print()
    # print('----- Sampling with diversity', diversity)
    sentence = prime
    generated = sentence

    # print('----- Generating with seed: "' + sentence + '"')
    # sys.stdout.write(generated)
    for i in range(n_chars):
        x = np.zeros((1, len(sentence), len(txt.chars)))
        for t, char in enumerate(sentence):
            x[0, t, txt.char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = txt.indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        # sys.stdout.write(next_char)
        # sys.stdout.flush()
    # print()
    return generated
if __name__ == '__main__':
    main()
