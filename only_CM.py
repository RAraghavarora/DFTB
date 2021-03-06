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
from qml.representations import generate_coulomb_matrix

import logging
import schnetpack as spk

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


def prepare_data(op):
    # read dataset
    # data_dir = '../'
    data_dir = '/scratch/ws/1/medranos-DFTB/raghav/data/'
    properties = [
        'RMSD',
        'EAT',
        'EMBD',
        'EGAP',
        'KSE',
        'FermiEne',
        'BandEne',
        'NumElec',
        'h0Ene',
        'sccEne',
        '3rdEne',
        'RepEne',
        'mbdEne',
        'TBdip',
        'TBeig',
        'TBchg',
    ]

    # data preparation
    logging.info("get dataset")
    dataset = spk.data.AtomsData(data_dir + 'qm7x-eq-n1.db', load_only=properties)

    n = len(dataset)
    print(n)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    # computing predicted property
    logging.info("get predicted property")
    AE, xyz, Z = [], [], []
    EGAP, KSE, TPROP = [], [], []
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = ([] for i in range(11))
    for i in idx2[:n]:
        atoms, props = dataset.get_properties(i)
        AE.append(float(props['EAT']))
        EGAP.append(float(props['EGAP']))
        KSE.append(props['KSE'])
        TPROP.append(float(props[op]))
        xyz.append(atoms.get_positions())
        Z.append(atoms.get_atomic_numbers())
        p1.append(float(props['FermiEne']))
        p2.append(float(props['BandEne']))
        p3.append(float(props['NumElec']))
        p4.append(float(props['h0Ene']))
        p5.append(float(props['sccEne']))
        p6.append(float(props['3rdEne']))
        p7.append(float(props['RepEne']))
        p8.append(float(props['mbdEne']))
        p9.append(props['TBdip'])
        p10.append(props['TBeig'])
        p11.append(props['TBchg'])

    AE = np.array(AE)
    EGAP = np.array(EGAP)
    TPROP = np.array(TPROP)

    # Generate representations
    xyz_reps = np.array(
        [generate_coulomb_matrix(
            Z[mol], xyz[mol], sorting='unsorted') for mol in idx2]
    )

    TPROP2 = []
    p1b, p2b, p11b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for nn in idx2:
        p1b.append(p1[nn])
        p2b.append(p2[nn])
        p3b.append(p3[nn])
        p4b.append(p4[nn])
        p5b.append(p5[nn])
        p6b.append(p6[nn])
        p7b.append(p7[nn])
        p8b.append(p8[nn])
        p9b.append(p9[nn].numpy())
        p10b.append(p10[nn].numpy())
        p11b.append(p11[nn].numpy())
        TPROP2.append(TPROP[nn])

    p11b = complete_array(p11b)

    reps2 = []
    for ii in range(len(idx2)):
        reps2.append(xyz_reps[ii])
     
    reps2 = np.array(reps2)

    return reps2, TPROP2


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    Target = np.array(Target)
    # Shuffle the data
    indices = np.arange(Repre.shape[0])
    np.random.shuffle(indices)
    Repre = Repre[indices]
    Target = Target[indices]

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

    # Data standardization
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, x_scaler, y_scaler


# fit a model and plot learning curve


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):

    trainX, trainy, valX, valy, testX, testy, x_scaler, y_scaler = split_data(
        n_train, n_val, n_test, iX, iY
    )

    n_input = int(len(iX[0]))
    # n_output = int(len(iY[0]))
    n_output = int(1)

    # define model
    model = Sequential()
    initializer = HeNormal()
    model.add(
        Dense(
            4,
            input_dim=n_input,
            activation='elu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(
        Dense(
            units=32,
            activation='elu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(
        Dense(
            units=32,
            activation='elu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(Dense(n_output, activation='linear', kernel_initializer=initializer))
    # compile model
    opt = Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    # fit model
    rlrp = ReduceLROnPlateau(
        monitor='val_loss', factor=0.59, patience=patience, min_delta=1e-5, min_lr=1e-6
    )
    lrm = LearningRateMonitor()
    history = model.fit(
        trainX,
        trainy,
        validation_data=(valX, valy),
        batch_size=16,
        epochs=20000,
        verbose=0,
        callbacks=[rlrp, lrm],
    )

    return (
        model,
        lrm.lrates,
        history.history['loss'],
        history.history['mae'],
        testX,
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

    plt.plot(x, y, 'ro')
    temp = np.arange(mini, maxi, 0.1)
    plt.plot(temp, temp)
    plt.xlabel("True EAT")
    plt.ylabel("Predicted EAT")
    plt.title('Results for training size of %s' % n_val)
    plt.savefig('Results.png')
    plt.close()


# prepare dataset
train_set = ['1000', '2000', '4000', '8000', '10000', '20000','30000']
n_val = 1000
n_test = 10000
op = sys.argv[1]

iX, iY = prepare_data(op)

# fit model and plot learning curves for a patience
patience = 500  # If no improvement is seen for these epochs, Learning rate is reduced

current_dir = os.getcwd()

for ii in range(len(train_set)):
    print('Trainset= {:}'.format(train_set[ii]))
    chdir(current_dir + '/only_CM/eq/egap/')
    os.chdir(current_dir + '/only_CM/eq/egap/')
    try:
        os.mkdir(str(train_set[ii]))
    except FileExistsError:
        pass
    os.chdir(current_dir + '/only_CM/eq/egap/' + str(train_set[ii]))

    if sys.argv[2] == 'fit':

        temp = train_set[ii]
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
        model = load_nnmodel(current_dir + '/%s' % train_set[ii])

    # Saving results
    plotting_results(model, testX, testy)
    save_plot(temp)
