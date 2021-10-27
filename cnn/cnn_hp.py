from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    Dropout,
    Flatten
)
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam

from keras_tuner import RandomSearch
import numpy as np


def get_model(hp):
    model = Sequential()
    initializer = HeNormal()
    model.add(
        Conv1D(
            hp.Choice("layer1", [4, 8, 16, 32]),
            kernel_size=3,
            input_shape=(316, 1),
            activation='relu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )

    model.add(
        Conv1D(
            hp.Choice("layer2", [4, 8, 16, 32]),
            kernel_size=3,
            activation='relu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(0.001),
        )
    )

    model.add(Dropout(0.5))

    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(hp.Choice("layer3", [4, 8, 16, 32]), activation="relu"))

    model.add(Dense(1, activation='linear', kernel_initializer=initializer))
    # compile model
    opt = Adam(learning_rate=1e-5)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    # fit model
    # rlrp = ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.59, patience=patience, min_delta=1e-5, min_lr=1e-6
    # )
    return model


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
    dataset = spk.data.AtomsData(
        data_dir + 'qm7x-eq-n1.db', load_only=properties)

    n = len(dataset)
    print(n)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    # computing predicted property
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
    # Coulomb matrix
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
        # reps2.append(xyz_reps[ii])
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
    reps2 = np.array(reps2)

    return reps2, TPROP2


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
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

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print(local_device_protos)


tuner = RandomSearch(
    get_model,
    objective="val_mae",
    max_trials=30,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="cnn-hp",
    seed=42
)
print(tuner.search_space_summary())

iX, iY = prepare_data('EAT')
# np.save('/scratch/ws/1/medranos-DFTB/raghav/data/iX.npy', iX)
np.save('/scratch/ws/1/medranos-DFTB/raghav/data/iY.npy', iY)

# n_train = 5000
# n_val = 2000
# n_test = 2000

# trainX, trainY, valX, valY, testX, testY = split_data(
#     n_train, n_val, n_test, iX, iY
# )

# trainX.shape = (trainX.shape[0], trainX.shape[1], 1)
# trainY.shape = (trainY.shape[0], 1)
# valX.shape = (valX.shape[0], valX.shape[1], 1)
# valY.shape = (valY.shape[0], 1)


# tuner.search(trainX, trainY, epochs=2000, validation_data=(valX, valY))
# models = tuner.get_best_models(num_models=2)
# tuner.results_summary()
# print(models)
