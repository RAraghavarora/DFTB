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
import schnetpack as spk
from qml.representations import generate_coulomb_matrix


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

iX = np.load('/scratch/ws/1/medranos-DFTB/raghav/data/iX.npy')
iY = np.load('/scratch/ws/1/medranos-DFTB/raghav/data/iY.npy')

n_train = 5000
n_val = 2000
n_test = 2000

trainX, trainY, valX, valY, testX, testY = split_data(
    n_train, n_val, n_test, iX, iY
)

trainX.shape = (trainX.shape[0], trainX.shape[1], 1)
trainY.shape = (trainY.shape[0], 1)
valX.shape = (valX.shape[0], valX.shape[1], 1)
valY.shape = (valY.shape[0], 1)


tuner.search(trainX, trainY, epochs=2000, validation_data=(valX, valY))
models = tuner.get_best_models(num_models=2)
tuner.results_summary()
print(models)
