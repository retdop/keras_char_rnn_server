from __future__ import print_function

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import Dense, Activation
except:
    pass

try:
    from keras.layers import LSTM
except:
    pass

try:
    from keras.optimizers import RMSprop
except:
    pass

try:
    import numpy
except:
    pass

try:
    import random
except:
    pass

try:
    import sys
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, conditional
except:
    pass

try:
    from keras.callbacks import EarlyStopping
except:
    pass

try:
    from sklearn.cross_validation import train_test_split
except:
    pass

try:
    from sample import sample, sample_chars
except:
    pass

try:
    from utils import TextLoader, MAX_LEN
except:
    pass

try:
    import gc
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

txt = TextLoader()
x_train, x_test, y_train, y_test = train_test_split(txt.X, txt.y, test_size=0.1)


def keras_fmin_fnct(space):

    model = Sequential()
    model.add(LSTM(space['LSTM'], input_shape = (None, len(txt.chars))))
    model.add(Dense(len(txt.chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    hist = model.fit(X_train, Y_train, validation_split=0.2, callbacks=[early_stopping], batch_size=128, epochs=1, shuffle = True)
    print(hist.history)
    model.save('data/python/python.h5')
    print('model saved to ' + 'data/python/python.h5')

    score, acc = model.evaluate(txt.X, txt.y, verbose=0)
    print('Test categorical accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'LSTM': hp.choice('LSTM', [128, 256]),
    }
