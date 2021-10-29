import logging
from torch.optim import Adam
import os
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError
from schnetpack.train import build_mse_loss
import torch
import torch.nn as nn

from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print(local_device_protos)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# basic settings
model_dir = '/scratch/ws/1/medranos-DFTB/raghav/code/schnet/model/'
# model_dir = './model/'
# os.makedirs(model_dir)
properties = ['EAT']

idir = '/scratch/ws/1/medranos-DFTB/raghav/data/'
# idir = '../'

# data preparation
logging.info("get dataset")
dataset = spk.AtomsData(idir + "totgdb7x_pbe0.db", load_only=properties)
train, val, test = spk.train_test_split(
    dataset, 30000, 2000, os.path.join(model_dir, "split.npz")
)
train_loader = spk.AtomsLoader(train, batch_size=32)
val_loader = spk.AtomsLoader(val, batch_size=32)

atomrefs = None  #dataset.get_atomrefs(properties)
#per_atom = dict(energy=True)
means, stddevs = train_loader.get_statistics(
    properties,  single_atom_ref=atomrefs
)

# model build
logging.info("build model")
representation = spk.SchNet(n_interactions=3,  cutoff=8.0)
output_modules = [
    spk.atomistic.Atomwise(
        n_in=128,
        property='EAT',
        mean=means['EAT'],
        stddev=stddevs['EAT'],
        atomref= None, #atomrefs['EDFTate'],
    )
]
model = spk.AtomisticModel(representation, output_modules)
model = model.to("cuda:0")
model = nn.DataParallel(model)

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# hooks
logging.info("build trainer")
metrics = [MeanAbsoluteError(p, p) for p in properties]
hooks = [CSVHook(log_path=model_dir, metrics=metrics), ReduceLROnPlateauHook(optimizer, patience=50, factor=0.5, min_lr=1e-6, stop_after_min=True)]

# trainer
loss = build_mse_loss(properties)
trainer = Trainer(
    model_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# run training
logging.info("training")
trainer.train(device="cuda:0")
