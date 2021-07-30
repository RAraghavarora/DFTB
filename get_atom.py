# NN model
import sys
import os
import pdb
from os import path, mkdir, chdir
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import torch
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
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
    data_dir = '../'
    properties = ['RMSD', 'EAT', 'EMBD', 'EGAP', 'KSE', 'FermiEne', 'BandEne', 'NumElec', 'h0Ene', 'sccEne', '3rdEne', 'RepEne', 'mbdEne', 'TBdip', 'TBeig', 'TBchg']

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
    for ii in range(len(idx2)):
        # reps2.append(xyz_reps[ii])
        reps2.append(np.concatenate((xyz_reps[ii], p1b[ii], p2b[ii], p3b[ii], p4b[ii], p5b[ii], p6b[ii], p7b[ii], p8b[ii], np.linalg.norm(p9b[ii]), p10b[ii], p11b[ii]), axis=None))
    reps2 = np.array(reps2)

    return reps2, TPROP2, atoms_data


n_train = 30000
n_test = 10000
n_val = 1000

Repre, Target, atoms_data = prepare_data('EAT')
print(Repre.shape)
pdb.set_trace()
X_train, X_val, X_test = np.array(Repre[:n_train]), np.array(Repre[-n_test - n_val:-n_test]), np.array(Repre[-n_test:])
Y_train, Y_val, Y_test = np.array(Target[:n_train]), np.array(Target[-n_test - n_val:-n_test]), np.array(Target[-n_test:])

Y_train = Y_train.reshape(-1, 1)
Y_val = Y_val.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(Y_train)

atoms_data = atoms_data[-n_test:]
model = load_model('withdft/30000' + '/model.h5')
y_test = model.predict(X_test)  # in eV

dtest = np.absolute(np.array(Y_test - y_test))

# Sort data according to the mae
indexes = list(range(len(dtest)))
indexes.sort(key=dtest.__getitem__)

with open("atoms_data.txt", 'w') as f:
    for i in indexes:
        f.write(str(atoms_data[i]) + '\t' + str(dtest[i]) + '\n')
