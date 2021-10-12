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

from tensorflow.keras.layers import Dense, BatchNormalization

from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from qml.representations import (
    generate_coulomb_matrix,
    get_slatm_mbtypes,
    generate_slatm,
    generate_bob,
)

from tensorflow.keras.initializers import HeNormal


import logging
import schnetpack as spk
from ase.io import read
from ase.db import connect
from ase.atoms import Atoms
from ase.calculators.dftb import Dftb
from ase.units import Hartree, Bohr

# monitor the learning rate


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
p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = (
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
for i in idx2[:n]:
    atoms, props = dataset.get_properties(i)
    # AE.append(float(props['EAT']))
    # EGAP.append(float(props['EGAP']))
    # KSE.append(props['KSE'])
    # TPROP.append(float(props[op]))
    xyz.append(atoms.get_positions())
    Z.append(atoms.get_atomic_numbers())
    # # p1.append(float(props['FermiEne']))
    # p2.append(float(props['BandEne']))
    # p3.append(float(props['NumElec']))
    # p4.append(float(props['h0Ene']))
    # p5.append(float(props['sccEne']))
    # p6.append(float(props['3rdEne']))
    # p7.append(float(props['RepEne']))
    # p8.append(float(props['mbdEne']))
    # p9.append(props['TBdip'])
    # p10.append(props['TBeig'])
    # p11.append(props['TBchg'])

AE = np.array(AE)
EGAP = np.array(EGAP)
TPROP = np.array(TPROP)

# Generate representations
# SLATM
mbtypes = get_slatm_mbtypes([Z[mol] for mol in idx2])
print(len(mbtypes[0]))
print(mbtypes[0].shape)
slatm = [
    generate_slatm(mbtypes=mbtypes, nuclear_charges=Z[mol], coordinates=xyz[mol])
    for mol in idx2
]

print("SLATM lengths:")
print(len(slatm))
print(len(slatm[0]))
