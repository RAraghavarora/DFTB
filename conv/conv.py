# NN model
import sys
import os
import pandas as pd
from os import path, mkdir, chdir
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.layers import Dense, Input, Add
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras import backend
from tensorflow.keras.models import load_model

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


def complete_array(Aprop):
    Aprop2 = []
    for ii in range(len(Aprop)):
        n1 = len(Aprop[ii])
        if n1 == 23:
            Aprop2.append(Aprop[ii])
        else:
            n2 = 23 - n1
            Aprop2.append(np.concatenate((Aprop[ii], np.zeros(n2)), axis=None))

    return Aprop2


# prepare train and test dataset


def prepare_data(op='EAT'):
    # load data from csv file
    data_dir = '/scratch/ws/1/medranos-DFTB/raghav/data/'
    # data_dir = '../'
    desc = pd.read_csv(data_dir + '/desc.csv', header=None)
    desc = pd.DataFrame(desc)
    desc = np.array(desc.values)

    dftb = pd.read_csv(data_dir + '/dftb.csv', header=None)
    dftb = pd.DataFrame(dftb)
    dftb = np.array(dftb.values)
    # DFTB data is standardized property wise

    Y = pd.read_csv(data_dir + '/Y.csv', header=None)
    Y = pd.DataFrame(Y)
    Y = np.array(Y.values)

    return [desc, dftb], Y


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    X_train, X_val, X_test = (
        np.array(Repre[:n_train]),
        np.array(Repre[-n_test - n_val : -n_test]),
        np.array(Repre[-n_test:]),
    )
    Y_train, Y_val, Y_test = (
        np.array(Target[:n_train]),
        np.array(Target[-n_test - n_val : -n_test]),
        np.array(Target[-n_test:]),
    )

    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, x_scaler, y_scaler


# fit a model and plot learning curve


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):

    desc = iX[0]
    dftb = iX[1]
    trainX1, trainy, valX1, valy, testX1, testy, x_scaler1, y_scaler = split_data(
        n_train, n_val, n_test, desc, iY
    )

    trainX2, trainy, valX2, valy, testX2, testy, x_scaler2, y_scaler = split_data(
        n_train, n_val, n_test, dftb, iY
    )

    n_input = int(len(iX[0][0]))
    # n_output = int(len(iY[0]))
    n_output = int(1)

    n_inout = n_input + n_output
    # define model

    # 1st model

    visible = Input(shape=(n_input,))
    hidden1 = Dense(
        n_inout,
        activation='tanh',
        kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.01),
    )(visible)
    hidden2 = Dense(units=256, activation='tanh')(hidden1)
    hidden3 = Dense(units=64, activation='tanh')(hidden2)
    out1 = Dense(units=64, activation='relu')(hidden3)

    # 2nd model
    n_input = int(len(iX[1][0]))
    n_inout = n_input + n_output

    visible2 = Input(shape=(n_input,))
    hidden21 = Dense(
        n_inout,
        activation='tanh',
        kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.01),
    )(visible2)
    hidden22 = Dense(units=256, activation='tanh')(hidden21)
    hidden23 = Dense(units=64, activation='tanh')(hidden22)
    out2 = Dense(units=64, activation='relu')(hidden23)

    hidden4 = Add()([out1, out2])
    out = Dense(n_output, activation='linear')(hidden4)

    model = Model(inputs=[visible, visible2], outputs=[out])

    #    plot_model(model, to_file='combined_NN.png')

    # compile model
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    # fit model
    rlrp = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=patience, min_delta=1e-5, min_lr=1e-6
    )
    lrm = LearningRateMonitor()
    history = model.fit(
        [trainX1, trainX2],
        trainy,
        validation_data=([valX1, valX2], valy),
        batch_size=32,
        epochs=20000,
        verbose=2,
        callbacks=[rlrp, lrm],
    )

    return (
        model,
        lrm.lrates,
        history.history['loss'],
        history.history['mae'],
        [testX1, testX2],
        testy,
    )


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


def save_plot(n_val):
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

    plt.plot(x, y, 'b.')
    temp = np.arange(mini, maxi, 0.1)
    plt.plot(temp, temp)
    plt.xlabel("True EAT")
    plt.ylabel("Predicted EAT")
    plt.savefig(str(n_val) + '.png')
    plt.close()


# prepare dataset
train_set = ['1000', '2000', '4000', '8000', '10000', '20000', '30000']
n_val = 1000
n_test = 10000
try:
    op = sys.argv[1]
except Exception as e:
    op = 'EAT'

iX, iY = prepare_data(op)

# fit model and plot learning curves for a patience
patience = 100

current_dir = os.getcwd()

for ii in range(len(train_set)):
    print('Trainset= {:}'.format(train_set[ii]))
    chdir(current_dir)
    os.chdir(
        current_dir + '/conv/withdft/new/'
    )  # Overwrite the existing redundant results
    try:
        os.mkdir(str(train_set[ii]))
    except:
        pass
    os.chdir(current_dir + '/conv/withdft/new/' + str(train_set[ii]))

    if sys.argv[2] == 'fit':

        model, lr, loss, acc, testX, testy = fit_model_dense(
            int(train_set[ii]), int(n_val), int(n_test), iX, iY, patience
        )

        lhis = open('learning-history.dat', 'w')
        for ii in range(0, len(lr)):
            lhis.write(
                '{:8d}'.format(ii)
                + '{:16f}'.format(lr[ii])
                + '{:16f}'.format(loss[ii])
                + '{:16f}'.format(acc[ii])
                + '\n'
            )
        lhis.close()

        # Saving NN model
        save_nnmodel(model)
    else:
        cfile = 'ncomp-test.dat'
        # to evaluate new test
        model = load_nnmodel(current_dir + '/NNmodel')

    # Saving results
    try:
        plotting_results(model, testX, testy)
        save_plot(n_val)
    except:
        pass
