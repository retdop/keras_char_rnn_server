from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from sample import sample, sample_chars
from utils import TextLoader, MAX_LEN


def data():
    txt = TextLoader()
    return txt.X, txt.y, txt.X, txt.y

def model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM({{choice([128, 256])}}, input_dim = len(txt.chars)))
    model.add(Dense(len(txt.chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

    model.fit(txt.X, txt.y, batch_size=128, epochs=1)
    model.save('data/nietzsche/nietzsche.h5')
    print('model saved to ' + 'data/nietzsche/nietzsche.h5')

    score, acc = model.evaluate(txt.X, txt.y, verbose=0)
    print('Test categorical accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(txt.X, txt.y))
