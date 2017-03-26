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

from keras.callbacks import EarlyStopping

from sklearn.cross_validation import train_test_split

from sample import sample, sample_chars
from utils import TextLoader, MAX_LEN



def data():
    txt = TextLoader()
    x_train, x_test, y_train, y_test = train_test_split(txt.X, txt.y, test_size=0.1)
    return x_train, x_test, y_train, y_test

def model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(128, input_shape = (None, X_train.shape[2])))
    model.add(Dense(X_train.shape[2]))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    hist = model.fit(X_train, Y_train, validation_split=0.2, callbacks=[early_stopping], batch_size=256, epochs=50)
    print(hist.history)
    model.save('data/python/python.h5')
    print('model saved to ' + 'data/python/python.h5')

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test categorical accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = data()
    model(x_train, y_train, x_test, y_test)
    import gc; gc.collect()
