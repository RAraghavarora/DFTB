# NN model
import numpy as np
import logging
import schnetpack as spk
import pdb

from os import mkdir, chdir, getcwd

from qml.kernels import gaussian_kernel
from qml.representations import generate_coulomb_matrix

from tensorflow.keras import regularizers, backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


def prepare_data(op):
    #  # read dataset

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
    try:
        data_dir = '/scratch/ws/1/medranos-DFTB/props/dftb/data/n1-2/'
        dataset = spk.data.AtomsData(
            data_dir + 'totgdb7x_pbe0.db', load_only=properties
        )
    except:
        data_dir = '../'
        dataset = spk.data.AtomsData(
            data_dir + 'totgdb7x_pbe0.db', load_only=properties
        )

    n = len(dataset)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    # computing predicted property
    logging.info("get predicted property")
    AE, xyz, Z = [], [], []
    EGAP, KSE, TPROP = [], [], []
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = ([] for i in range(11))
    atoms_data = []
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
        atoms_data.append(atoms)

    AE = np.array(AE)
    EGAP = np.array(EGAP)
    TPROP = np.array(TPROP)
    atoms_data = np.array(atoms_data)

    # Generate representations
    # Coulomb matrix
    xyz_reps = np.array(
        [generate_coulomb_matrix(Z[mol], xyz[mol], sorting='unsorted') for mol in idx2]
    )

    TPROP2 = []
    p1b, p2b, p11b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b = ([] for i in range(11))
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

    temp = []
    for var in [p1b, p2b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b, p11b]:
        var2 = np.array(var)
        var2 = var2.reshape(-1, 1)
        scaler = StandardScaler()
        var3 = scaler.fit_transform(var2)
        temp.append(var3)

    p1b, p2b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b, p11b = temp

    reps2 = []
    for ii in range(len(idx2)):
        reps2.append(
            np.concatenate(
                (
                    xyz_reps[ii],
                    p1b[ii],
                    p2b[ii],
                    p3b[ii],
                    p4b[ii],
                    p5b[ii],
                    p6b[ii],
                    p7b[ii],
                    p8b[ii],
                    np.linalg.norm(p9b[ii]),
                    p10b[ii],
                    p11b[ii],
                ),
                axis=None,
            )
        )

    return np.array(reps2), TPROP2, atoms_data


def fit_model_dense(K_train, K_val, K_test, Y_val, Y_train, patience=1000):

    n_input = K_train.shape[0]
    n_output = int(1)
    # define model
    model = Sequential()
    initializer = HeNormal()

    input_len = [2000,1]

    model.add(
        Conv1D(
            filters=32,
            kernel_size=15,
            input_shape=input_len,
            activation='elu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )

    model.add(MaxPooling1D(pool_size=4))

    model.add(
        Conv1D(
            filters=16,
            kernel_size=15,
            strides=2,
            input_shape=input_len,
            activation='elu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )

    model.add(MaxPooling1D(pool_size=2))

    model.add(
        Conv1D(
            filters=8,
            kernel_size=3,
            input_shape=input_len,
            activation='elu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )

    model.add(MaxPooling1D(pool_size=4))

    model.add(
        Conv1D(
            filters=1,
            kernel_size=3,
            strides=2,
            input_shape=input_len,
            activation='elu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )

    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='linear'))
    # compile model
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    # fit model
    rlrp = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=patience, min_delta=1e-5, min_lr=1e-6
    )
    lrm = LearningRateMonitor()
    K_train.shape=(2000,2000,1)
    history = model.fit(
        K_train,
        Y_train,
        validation_data=(K_val, Y_val),
        batch_size=32,
        epochs=20000,
        verbose=1,
        callbacks=[rlrp, lrm],
    )

    return (model, lrm.lrates, history.history['loss'], history.history['mae'])


def plotting_results(model, K_test, testy):
    # applying nn model
    y_test = model.predict(K_test)
    testy.shape = (testy.shape[0], 1)
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

    format_list1 = ['{:16f}']

    s = ' '.join(format_list1)
    ctest = open('comp-test.dat', 'w')
    for ii in range(0, len(testy)):
        ctest.write(
            s.format(*testy[ii]) + s.format(*y_test[ii]) + s.format(*dtest[ii]) + '\n'
        )
    ctest.close()


Repre, Target, atoms_data = prepare_data('EAT')
Target = np.array(Target)

sigma = 158.495
gamma = 1.92823619e-05

n_test = 10000
n_val = 2000

train_set = [2000]

current_dir = getcwd()

for n_train in train_set:
    chdir(current_dir + '/kernel/')

    n_val = n_test = n_train
    print('Trainset= {:}'.format(n_train))
    try:
        mkdir(str(n_train))
    except FileExistsError:
        pass
    chdir(current_dir + '/kernel/' + str(n_train))

    indices = np.arange(Repre.shape[0])
    np.random.shuffle(indices)
    Repre = Repre[indices]
    Target = Target[indices]

    X_train = np.array(Repre[:n_train])
    X_val = np.array(Repre[-n_test - n_val : -n_test])
    X_test = np.array(Repre[:n_test])
    Y_train, Y_val, Y_test = (
        np.array(Target[:n_train]),
        np.array(Target[-n_test - n_val : -n_test]),
        np.array(Target[-n_test:]),
    )

    # Generate kernels
    K_train = gaussian_kernel(X_train, X_train, sigma)
    K_train[np.diag_indices_from(K_train)] += gamma  # Regularizer

    K_val = gaussian_kernel(X_val, X_val, sigma)
    K_val[np.diag_indices_from(K_val)] += gamma  # Regularizer

    K_test = gaussian_kernel(X_test, X_test, sigma)
    K_test[np.diag_indices_from(K_test)] += gamma  # Regularizer

    model, lr, history_loss, history_mae = fit_model_dense(
        K_train, K_val, K_test, Y_val, Y_train
    )

    lhis = open('learning-history.dat', 'w')
    for ii in range(0, len(lr)):
        lhis.write(
            '{:8d}'.format(ii)
            + '{:16f}'.format(lr[ii])
            + '{:16f}'.format(history_loss[ii])
            + '{:16f}'.format(history_mae[ii])
            + '\n'
        )
    lhis.close()

    model.save("model.h5")
    plotting_results(model, K_test, Y_test)
