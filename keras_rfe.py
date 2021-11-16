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

import logging
import schnetpack as spk
import pandas as pd

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
        'EAT',
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

    input_data = {}

    for prop in properties:
        input_data[prop] = []

    # data preparation
    logging.info("get dataset")
    dataset = spk.data.AtomsData(
        data_dir + 'qm7x-eq-n1.db', load_only=properties)

    n = len(dataset)
    print(n)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    # computing predicted property
    logging.info("get predicted property")
    AE, xyz, Z = [], [], []
    EGAP, KSE, TPROP = [], [], []
    for i in idx2[:n]:
        try:
            atoms, props = dataset.get_properties(int(i))
        except:
            print(i)
            pdb.set_trace()
        for prop in properties:
            if prop not in ['TBchg', 'TBeig', 'TBdip']:
                input_data[prop].append(float(props[prop]))
            elif prop == 'TBdip':
                input_data[prop].append(np.linalg.norm(props[prop]))
            else:
                input_data[prop].append(np.array(props[prop]))
        TPROP.append(float(props[op]))
        xyz.append(atoms.get_positions())
        Z.append(atoms.get_atomic_numbers())

    TPROP = np.array(TPROP)

    # Generate representations
    # # Coulomb matrix
    # xyz_reps = np.array(
    #     [generate_coulomb_matrix(Z[mol], xyz[mol], sorting='unsorted') for mol in idx2]
    # )

    input_data['TBchg'] = complete_array(input_data['TBchg'])

    df = pd.DataFrame.from_dict(input_data)
    # Standardize the data property wise

    standard_df = (df-df.mean())/df.std()
    standard_df['TBeig'] = df['TBeig']
    standard_df['TBchg'] = df['TBchg']

    return df, TPROP


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

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, x_scaler, y_scaler


# fit a model and plot learning curve


def fit_model_dense(trainX, trainY, valX, valY):

    n_input = int(len(trainX[0]))
    n_output = 1

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
    model.add(Dense(n_output, activation='linear',
                    kernel_initializer=initializer))
    # compile model
    opt = Adam(learning_rate=1e-5)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    lrm = LearningRateMonitor()
    history = model.fit(
        trainX,
        trainY,
        validation_data=(valX, valY),
        batch_size=16,
        epochs=4000,
        verbose=0,
        callbacks=[lrm],
    )

    return (
        model,
        lrm.lrates,
        history.history['loss'],
        history.history['mae'],
        testX,
        testY,
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
            s.format(*testy[ii]) + s.format(*y_test[ii]) +
            s.format(*dtest[ii]) + '\n'
        )
    ctest.close()
    return MAE_PROP


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


op = sys.argv[1]
df, iY = prepare_data(op)

properties = [
    'TBdip',
    'FermiEne',
    'BandEne',
    'NumElec',
    'h0Ene',
    'sccEne',
    '3rdEne',
    'RepEne',
    'mbdEne',
    'TBeig',
    'TBchg',
]

iX = []
n = len(iY)
idx = np.arange(n)
for i in idx:
    iX.append(np.concatenate([df[prop][i] for prop in properties], axis=None))

n_train = 4000
n_val = 1000
n_test = 10000

trainX, trainY, valX, valY, testX, testY, x_scaler, y_scaler = split_data(
    n_train, n_val, n_test, iX, iY
)
pdb.set_trace()

model, lr, loss, acc, testX, testy = fit_model_dense(
    trainX, trainY, valX, valY
)


mae = plotting_results(model, testX, testy)

mae_min = mae
delta = 0.04

for prop in properties:
    # current_dir = "/scratch/ws/1/medranos-DFTB/raghav/code"
    # chdir(current_dir + '/rfe/')
    # try:
    #     os.mkdir(str(prop))
    # except FileExistsError:
    #     pass
    # chdir(current_dir + '/rfe/' + str(prop))

    logging.info("Removing property: " + prop)

    df_modified = df.drop([prop], axis=1)
    iX = []
    for i in idx:
        iX.append(np.concatenate([df[prop][i]
                                  for prop in properties], axis=None))
    iX = np.array(iX)

    trainX, trainy, valX, valy, testX, testy, x_scaler, y_scaler = split_data(
        n_train, n_val, n_test, iX, iY
    )

    model, lr, loss, acc, testX, testy = fit_model_dense(
        trainX, trainY, valX, valY
    )

    mae = plotting_results(model, testX, testy)
    logging.info("MAE obtained: " + mae)

    if mae_min - mae > delta:
        # Removing the property improved the model. Keep the property removed
        logging.log(
            10, "Keeping the property removed. Modifying the min mae: " + mae)
        mae_min = mae
        df = df_modified
        logging.info("Properties remaining:")
        print(df.columns)
        continue
    else:
        # Removing the property did not improve the model, don't remove the prop
        logging.info("No improve in mae, don't remove the property.")
        continue
