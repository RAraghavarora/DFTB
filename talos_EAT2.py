# NN model
import sys
import os
import pdb
from os import path, mkdir, chdir
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import HeNormal
# from qml.representations import generate_coulomb_matrix

import logging
# import schnetpack as spk
from keras_tuner import RandomSearch


# monitor the learning rate


class LearningRateMonitor(Callback):
    # start of training
    def on_train_begin(self, logs={}):
        self.lrates = list()

    # end of each training epoch
    def on_epoch_end(self, epoch, logs={}):
        # get and store the learning rate
        lrate = float(backend.get_value(self.model.optimizer.lr))
        self.lrates.append(lrate)


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    Target = np.array(Target)

    X_train, X_val, X_test = (
        np.array(Repre[:n_train]),
        np.array(Repre[-n_test - n_val: -n_test]),
        np.array(Repre[-n_test:]),
    )
    Y_train, Y_val, Y_test = (
        np.array(Target[:n_train]),
        np.array(Target[-n_test - n_val: -n_test]),
        np.array(Target[-n_test:]),
    )

    # Data standardization
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def egap_model(hp):
    model = Sequential()
    initializer = HeNormal()
    act1 = hp.Choice('activation1',['relu','sigmoid','tanh','elu'])
    act2 = hp.Choice('activation3',['relu','sigmoid','tanh','elu'])

    model.add(
        Dense(
            4,
            input_dim=316,
            activation=act1,
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )

    model.add(
        Dense(
            32,
            activation=act1,
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(
        Dense(
            32,
            activation=act2,
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(Dense(1, activation='linear', kernel_initializer=initializer))
    # compile model
    opt = Adam(learning_rate=1e-5)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    # fit model

    return model


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):

    trainX, trainy, valX, valy, testX, testy, x_scaler, y_scaler = split_data(
        n_train, n_val, n_test, iX, iY
    )

    n_input = int(len(iX[0]))
    # n_output = int(len(iY[0]))
    n_output = int(1)

    p = {'activation1': ['relu','tanh','sigmoid','elu'],
         'activation3': ['relu','tanh','sigmoid','elu'],
         'batch_size': [16],
         'epochs': [5000],
        }

    t = ta.Scan(x=trainX,
            y=trainy,
            model=egap_model,
            params=p,
            experiment_name='1')

    return t
    # return (
    #     model,
    #     lrm.lrates,
    #     history.history['loss'],
    #     history.history['mae'],
    #     testX,
    #     testy,
    # )


def plotting_results(model, testX, testy):
    # applying nn model
    y_test = model.predict(testX)
    # y_test = y_scaler.inverse_transform(y_test)
    MAE_PROP = float(mean_absolute_error(testy, y_test))
    MSE_PROP = float(mean_squared_error(testy, y_test))
    STD_PROP = float(testy.std())

    out2 = open('errors_test.dat', 'w')
    out2.write(
        '{:>24}'.format(STD_PROP)
        + '{:>24}'.format(MAE_PROP)
        + '{:>24}'.format(MSE_PROP)
        + "\n"
    )
    out2.close()

    # writing ouput for comparing values
    dtest = np.array(testy - y_test)
    format_list1 = ['{:16f}' for item1 in testy[0]]
    s = ' '.join(format_list1)
    ctest = open('comp-test.dat', 'w')
    for ii in range(0, len(testy)):
        ctest.write(
            s.format(*testy[ii]) + s.format(*y_test[ii]) + s.format(*dtest[ii]) + '\n'
        )
    ctest.close()


# save model and architecture to single file


def save_nnmodel(model):
    model.save("model.h5")
    print("Saved model to disk")


# load model


def load_nnmodel(idir):
    model = load_model(idir + '/model.h5')
    print("Loaded model from disk")
    return model


def save_plot(n_train):
    f = open("comp-test.dat", 'r')
    lines = f.readlines()
    x = []
    y = []
    mini = float(lines[0].split()[1])
    maxi = float(lines[0].split()[1])
    for line in lines:
        x1, y1, z1 = line.split()
        x.append(float(x1))
        y.append(float(y1))
        if float(x1) < mini:
            mini = float(x1)
        if float(x1) > maxi:
            maxi = float(x1)

    plt.plot(x, y, '.')
    temp = np.arange(mini, maxi, 0.1)
    plt.plot(temp, temp)
    plt.xlabel("True EAT")
    plt.ylabel("Predicted EAT")
    plt.title('Results for training size of %s' % n_train)
    plt.savefig('Results.png')
    plt.close()


# prepare dataset
train_set = ['10000']
n_val = 1000
n_test = 10000
op = sys.argv[1]

# fit model and plot learning curves for a patience
patience = 500

current_dir = os.getcwd()

tuner = RandomSearch(
    egap_model,
    objective="val_mae",
    max_trials=30,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir2",
    project_name="act",
    seed=42
)
print(tuner.search_space_summary())

iX = np.load('/scratch/ws/1/medranos-DFTB/raghav/data/iX.npy')
iY = np.load('/scratch/ws/1/medranos-DFTB/raghav/data/iY.npy')

n_train = 5000
n_val = 2000
n_test = 2000

trainX, trainY, valX, valY, testX, testY = split_data(
    n_train, n_val, n_test, iX, iY
)

tuner.search(trainX, trainY, epochs=2000, validation_data=(valX, valY))
models = tuner.get_best_models(num_models=2)
tuner.results_summary()
print(models)
