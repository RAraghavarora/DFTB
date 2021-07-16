# NN model
import sys
import pdb
import os
from os import path, mkdir, chdir
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import torch
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error

from kerastuner.tuners import BayesianOptimization

from tensorflow.keras.layers import Dense, Input, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from qml.representations import generate_coulomb_matrix
from qml.representations import generate_bob

import logging
import schnetpack as spk
from ase.io import read
from ase.db import connect
from ase.atoms import Atoms
from ase.calculators.dftb import Dftb
from ase.units import Hartree, Bohr

# monitor the learning rate


class LearningRateMonitor(Callback):
    # start of training
    def on_train_begin(self, logs={}):
        self.lrates = list()

    # end of each training epoch
    def on_epoch_end(self, epoch, logs={}):
        # get and store the learning rate
        optimizer = self.model.optimizer
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
    #  # read dataset
    data_dir = '/scratch/ws/1/medranos-DFTB/props/dftb/data/n1-2/'

    properties = ['RMSD', 'EAT', 'EMBD', 'EGAP', 'KSE', 'FermiEne', 'BandEne', 'NumElec', 'h0Ene', 'sccEne', '3rdEne', 'RepEne', 'mbdEne', 'TBdip', 'TBeig', 'TBchg']

    # data preparation
    logging.info("get dataset")
    dataset = spk.data.AtomsData(data_dir + 'totgdb7x_pbe0.db', load_only=properties)

    n = len(dataset)
    print(n)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    # computing predicted property
    logging.info("get predicted property")
    AE, xyz, Z = [], [], []
    EGAP, KSE, TPROP = [], [], []
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = [], [], [], [], [], [], [], [], [], [], []
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
    # Coulomb matrix
    xyz_reps = np.array([generate_coulomb_matrix(Z[mol], xyz[mol], sorting='unsorted') for mol in idx2])

    TPROP2 = []
    p1b, p2b, p11b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b = [], [], [], [], [], [], [], [], [], [], []
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
    desc = []
    dftb = []
    for ii in range(len(idx2)):
        desc.append(xyz_reps[ii])
        dftb.append(np.concatenate((p1b[ii], p2b[ii], p3b[ii], p4b[ii], p5b[ii], p6b[ii], p7b[ii], p8b[ii], np.linalg.norm(p9b[ii]), p10b[ii], p11b[ii]), axis=None))

    desc = np.array(desc)
    dftb = np.array(dftb)

    return [desc, dftb], TPROP2


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    X_train, X_val, X_test = np.array(Repre[:n_train]), np.array(Repre[-n_test - n_val:-n_test]), np.array(Repre[-n_test:])
    Y_train, Y_val, Y_test = np.array(Target[:n_train]), np.array(Target[-n_test - n_val:-n_test]), np.array(Target[-n_test:])

    # Data standardization
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, x_scaler, y_scaler

# fit a model and plot learning curve


def build_model(hp):
    '''
    Building model for Hyper parameter tuning
    '''

    # Define the NN model

    n_input = 276
    n_output = 1
    n_inout = n_input + n_output

    # 1st Model (Keras Functional API)
    visible = Input(shape=(n_input,))

    # Add one hidden layer with nodes between 32 and 256
    hp_units = hp.Int('unit1', min_value=32, max_value=256, step=32)
    hp_activation = hp.Choice('activation',
                              values=['relu', 'tanh', 'sigmoid'],
                              default='relu'
                              )
    hidden1 = Dense(hp_units, activation=hp_activation, 
                    kernel_initializer='he_uniform',
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01))(visible)

    # 2nd Hidden Layer
    hp_units = hp.Int('unit2', min_value=32, max_value=256, step=32)
    hp_activation = hp.Choice('activation2',
                              values=['relu', 'tanh', 'sigmoid'],
                              default='relu'
                              )
    hidden2 = Dense(hp_units, activation=hp_activation)(hidden1)

    # 3rd Hidden Layer
    hp_units = hp.Int('unit3', min_value=32, max_value=256, step=32)
    hidden3 = Dense(units=hp_units, activation='tanh')(hidden2)

    # Output layer of 1st model
    hp_activation = hp.Choice('activation3',
                              values=['relu', 'tanh', 'sigmoid'],
                              default='relu'
                              )
    out_units = hp.Int('unit_out', min_value=32, max_value=256, step=32)
    out1 = Dense(units=out_units, activation=hp_activation)(hidden3)

    # 2nd Model
    n_input = 40
    n_inout = n_input + n_output

    visible2 = Input(shape=(n_input,))

    hp_units = hp.Int('unit5', min_value=32, max_value=256, step=32)
    hp_activation = hp.Choice('activation4',
                              values=['relu', 'tanh', 'sigmoid'],
                              default='relu'
                              )
    hidden21 = Dense(hp_units, activation=hp_activation, 
                     kernel_initializer='he_uniform',
                     kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.01))(visible2)

    hp_units = hp.Int('unit6', min_value=32, max_value=256, step=32)
    hp_activation = hp.Choice('activation5',
                              values=['relu', 'tanh', 'sigmoid'],
                              default='relu'
                              )
    hidden22 = Dense(hp_units, activation=hp_activation)(hidden21)

    hp_units = hp.Int('unit7', min_value=32, max_value=256, step=32)
    hidden23 = Dense(units=hp_units, activation='tanh')(hidden22)

    hp_activation = hp.Choice('activation6',
                              values=['relu', 'tanh', 'sigmoid'],
                              default='relu'
                              )
    out2 = Dense(units=out_units, activation=hp_activation)(hidden23)

    # Add the layers
    hidden4 = Add()([out1, out2])
    # Final output layer
    out = Dense(n_output, activation='linear')(hidden4)

    # Define the model
    model = Model(inputs=[visible, visible2], outputs=[out])

    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):

    desc = iX[0]
    dftb = iX[1]
    trainX1, trainy, valX1, valy, testX1, testy, x_scaler1, y_scaler = split_data(n_train, n_val, n_test, desc, iY)
    trainX2, trainy, valX2, valy, testX2, testy, x_scaler2, y_scaler = split_data(n_train, n_val, n_test, dftb, iY)

    n_input = int(len(iX[0][0]))
    #n_output = int(len(iY[0]))
    n_output = int(1)

    n_inout = n_input + n_output

    print("Before Tuner")

    class MyTuner(BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            '''
            Overriding the run_trial method to change the batchsize and the number of epochs
            '''
            kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 64, step=32)
            kwargs['epochs'] = trial.hyperparameters.Int('epochs', 20, 50)
            super(MyTuner, self).run_trial(trial, *args, **kwargs)

    tuner = MyTuner(
        build_model,
        objective='val_mae',
        max_trials=2,
        project_name='Functional_tuner',
        directory='./Logs2/')

    # If the mae does not improve over a span of 30 epochs, stop the training.
    stop_early = EarlyStopping(monitor='val_mae', patience=30)

    # Search for optimal hyperparameters
    print("Tuning Started")
    tuner.search([trainX1, trainX2], trainy, validation_data=([valX1, valX2], valy), shuffle=True, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The hyperparameter search is complete. 
    {best_hps.get('unit1')} --- {best_hps.get('activation')} 
    {best_hps.get('unit2')} --- {best_hps.get('activation2')} 
    {best_hps.get('unit3')} --- 
    {best_hps.get('unit_out')} --- {best_hps.get('activation3')} 

    {best_hps.get('unit5')} --- {best_hps.get('activation4')} 
    {best_hps.get('unit6')} --- {best_hps.get('activation5')} 
    {best_hps.get('unit7')} --- 
    {best_hps.get('unit_out')} --- {best_hps.get('activation6')} 
    """)

    # Build the model
    model = tuner.hypermodel.build(best_hps)

    # compile model
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    # fit model
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience, min_delta=1E-5, min_lr=1E-6)
    lrm = LearningRateMonitor()
    history = model.fit([trainX1, trainX2], trainy, validation_data=([valX1, valX2], valy), 
                        batch_size=32, epochs=20000, verbose=2, callbacks=[rlrp, lrm])

    return model, lrm.lrates, history.history['loss'], history.history['mae'], [testX1, testX2], testy


def plotting_results(model, testX, testy):
    # applying nn model
    y_test = model.predict(testX)
    #y_test = y_scaler.inverse_transform(y_test)
    MAE_PROP = float(mean_absolute_error(testy, y_test))
    MSE_PROP = float(mean_squared_error(testy, y_test))
    STD_PROP = float(testy.std())

    out2 = open('errors_test.dat', 'w')
    out2.write('{:>24}'.format(STD_PROP) + '{:>24}'.format(MAE_PROP) + '{:>24}'.format(MSE_PROP) + "\n")
    out2.close()

    # writing ouput for comparing values
    dtest = np.array(testy - y_test)
    format_list1 = ['{:16f}' for item1 in testy[0]]
    s = ' '.join(format_list1)
    ctest = open('comp-test.dat', 'w')
    for ii in range(0, len(testy)):
        ctest.write(s.format(*testy[ii]) + s.format(*y_test[ii]) + s.format(*dtest[ii]) + '\n')
    ctest.close

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
    import matplotlib.pyplot as plt
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
    plt.savefig(str(n_val) + '.png')
    plt.close()


# prepare dataset
train_set = ['2000', '4000', '8000', '10000', '20000', '30000']
n_val = 1000
n_test = 10000
op = sys.argv[1]

iX, iY = prepare_data(op)

# fit model and plot learning curves for a patience
patience = 100 

current_dir = os.getcwd()

for ii in range(len(train_set)):
    print('Trainset= {:}'.format(train_set[ii]))
    chdir(current_dir)
    os.chdir(current_dir)
    try:
        os.mkdir(str(train_set[ii]))
    except:
        pass
    os.chdir(current_dir + str(train_set[ii]))

    if sys.argv[2] == 'fit':

        model, lr, loss, acc, testX, testy = fit_model_dense(int(train_set[ii]), int(n_val), int(n_test), iX, iY, patience)

        lhis = open('learning-history.dat', 'w')
        for ii in range(0, len(lr)):
            lhis.write('{:8d}'.format(ii) + '{:16f}'.format(lr[ii]) + '{:16f}'.format(loss[ii]) + '{:16f}'.format(acc[ii]) + '\n')
        lhis.close

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
