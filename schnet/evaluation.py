import torch
import logging
import numpy as np
import schnetpack as spk
#import ase
from ase.io import read
from schnetpack.interfaces.ase_interface import SpkCalculator, AseInterface
from ase.db import connect
from statistics import mean

idir = '/scratch/ws/1/medranos-DFTB/raghav/data/'
model_dir = '/scratch/ws/1/medranos-DFTB/raghav/code/schnet/'

properties = ['EAT']

# data preparation
logging.info("get dataset")
dataset = spk.AtomsData(idir+"totgdb7x_pbe0.db", load_only=properties)

# load model
logging.info("get model")
path_to_model = model_dir+'uljj_model/best_model'
model = torch.load(path_to_model, map_location='cuda')

# build calculator
calculator = SpkCalculator(model, device='cuda',energy="EDFT2", forces="forMBD")

split = np.load('split.npz')

test_data = split['test_idx']
w = len(test_data)
print(w)
n=int(w)
red_test = test_data[:n]

#computing predicted property
logging.info("get predicted property")
AE, PAE, FOR, FORCE, PFORCE = [], [], [], [], []
for i in red_test:
    atoms, props = dataset.get_properties(i)
    AE.append(float(props['EDFT2']))
    FORCE.append(props['forMBD'])
    FOR.append(str(atoms.symbols))
    atoms.set_calculator(calculator)
    PAE.append(float(atoms.get_total_energy()))
    PFORCE.append(atoms.get_forces())

AE = np.array(AE)
PAE = np.array(PAE)
FOR = np.array(FOR)
FORCE2 = []
for ii in range(n):
  FORCE2.append(np.asarray(FORCE[ii]))

FORCE2 = np.asarray(FORCE2)
PFORCE = np.asarray(PFORCE)
DAE = AE-PAE
DFORCE = FORCE2-PFORCE

ofile = open('ener-comp-'+str(n)+'.dat', 'w')
for i in range(n):
  ofile.write("{:>80}".format(FOR[i]) +"{:>24}".format(AE[i]) + "{:>24}".format(PAE[i]) + "{:>24}".format(DAE[i]) +  "\n")
ofile.close()

nbin=1000
hist, bin_edges = np.histogram(DAE, bins=nbin, range=(-10.0,10.0))
hist2, bin_edges2 = np.histogram(DAE, bins=nbin, range=(-10.0,10.0), density=True)

ofile2 = open('ener-hist-'+str(n)+'.dat', 'w')
for i in range(nbin):
  ofile2.write("{:>24}".format(bin_edges[i]) + "{:>24}".format(hist[i]) + "{:>24}".format(hist2[i]) +  "\n")
ofile2.close()

MFOR = []
for ii in range(n):
  MFOR.append(np.mean(np.abs(DFORCE[ii])))

MFOR = np.array(MFOR)

ofile3 = open('for-comp-'+str(n)+'.dat', 'w')
for i in range(n):
  ofile3.write("{:>80}".format(FOR[i]) + "{:>24}".format(MFOR[i]) +  "\n")
ofile3.close()

nbin=800
hist, bin_edges = np.histogram(MFOR, bins=nbin, range=(0.0,4.0))
hist2, bin_edges2 = np.histogram(MFOR, bins=nbin, range=(0.0,4.0), density=True)

ofile4 = open('for-hist-'+str(n)+'.dat', 'w')
for i in range(nbin):
  ofile4.write("{:>24}".format(bin_edges[i]) + "{:>24}".format(hist[i]) + "{:>24}".format(hist2[i]) +  "\n")
ofile4.close()




