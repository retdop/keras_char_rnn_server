from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np
import random
import sys

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

from sklearn.cross_validation import train_test_split

from sample import sample, sample_chars
from utils import TextLoader, MAX_LEN


def data():
    txt = TextLoader()
    X_train, X_test, Y_train, Y_test = train_test_split(txt.X, txt.y, test_size=0.1)
    return X_train, X_test, Y_train, Y_test

def model(X_train, Y_train, X_test, Y_test):
    print('starting model')
    n_modules = {{choice([128, 256])}}
    lr = {{choice([0.01, 0.001, 0.1])}}
    batch_size = {{choice([128, 256, 512])}}
    print("hyperparams :", n_modules, lr, batch_size)
    model = Sequential()
    model.add(LSTM(n_modules, input_shape = (None, X_train.shape[2])))
    model.add(Dense(X_train.shape[2]))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    tensorboard = TensorBoard(log_dir='./logs')
    hist = model.fit(X_train, Y_train, batch_size=batch_size, callbacks=[early_stopping, tensorboard], validation_split=0.2)
    print(hist.history)
    model.save('data/python/python_' + str(n_modules) + '_' + str(lr) + '_' + str(batch_size) + '.h5')
    print('model saved to ' + 'data/python/python_' + str(n_modules) + '_' + str(lr) + '_' + str(batch_size) + '.h5')

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test categorical accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    X_train, X_test, Y_train, Y_test = data()
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    import gc; gc.collect()
