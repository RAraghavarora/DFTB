# NN model
import sys
import os
import pdb
from os import path, mkdir, chdir
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from qml.fchl import generate_representation

import logging
import schnetpack as spk
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    #  # read dataset
    data_dir = '/scratch/ws/1/medranos-DFTBprojects/raghav/data/'
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

    fchl = np.array([generate_representation(xyz[mol], Z[mol]) for mol in idx2])

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

    # Standardize the data property wise

    temp = []
    for var in [p1b, p2b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b, p11b]:
        var2 = np.array(var)
        try:
            _ = var2.shape[1]
        except IndexError:
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
                    fchl[ii].flatten(),
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
    print("Perfoming training")
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
    print(X_val.shape)

    # Data standardization
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    print(Y_val.shape)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, x_scaler, y_scaler

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(316,4),
            nn.ELU(),
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    device = "cuda"
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        mae = float(mean_absolute_error(pred,y))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, mae = 0, 0
    device = "cuda"
    with torch.no_grad():
        for batch, X, y in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            mae += float(mean_absolute_error(pred,y))

    test_loss /= num_batches
    mae /= num_batches
    return test_loss, mae


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):
    batch_size = 16
    X_train, Y_train, X_val, Y_val, X_test, Y_test, x_scaler, y_scaler = split_data(
        n_train, n_val, n_test, iX, iY
    )

    train = torch.utils.data.TensorDataset(X_train,Y_train)
    test = torch.utils.data.TensorDataset(X_test,Y_test)
    valid = torch.utils.data.TensorDataset(X_val,Y_val)
    # data loader
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
    valid_loader = DataLoader(valid, batch_size = batch_size, shuffle = False)

    device = "cuda"
    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.57, patience = 500, min_lr=1e-6)

    epochs = 20000
    val_losses, val_errors, lrates = [], [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        valid_loss, valid_mae = test(valid_loader, model, loss_fn)
        print(f"Validation MAE: {valid_mae}\n")
        scheduler.step(valid_mae)
        val_losses.append(valid_loss)
        val_errors.append(valid_mae)
        lrates.append(optimizer.param_groups[0]['lr'])

    test_mae = test(test_loader, model, loss_fn)
    print(f"Finished training on train_size={n_train}\n Testing MAE = {test_mae}")

    return (
        model,
        lrates,
        val_losses,
        val_errors,
        test_loader
    )


def plotting_results(model, test_loader):
    # applying nn model
    with torch.no_grad():
        pred = model(test_loader.dataset.tensors[0])
        y = test_loader.dataset.tensors[1]
        test_loss = loss_fn(pred, y).item()
        mae = float(mean_absolute_error(pred,y))

    STD_PROP = float(pred.std())

    out2 = open('errors_test.dat', 'w')
    out2.write(
        '{:>24}'.format(STD_PROP)
        + '{:>24}'.format(mae)
        + '{:>24}'.format(test_loss)
        + "\n"
    )
    out2.close()

    # writing ouput for comparing values
    dtest = np.array(pred - y)
    Y_test = y.reshape(-1, 1)
    format_list1 = ['{:16f}' for item1 in Y_test[0]]
    s = ' '.join(format_list1)
    ctest = open('comp-test.dat', 'w')
    for ii in range(0, len(pred)):
        ctest.write(
            s.format(*pred[ii]) + s.format(*Y_test[ii]) + s.format(*dtest[ii]) + '\n'
        )
    ctest.close()

    #Save as a plot
    plt.plot(pred,y,'.')
    mini = min(y).item()
    maxi = max(y).item()
    temp = np.arange(mini, maxi, 0.1)
    plt.plot(temp, temp)
    plt.xlabel("True EAT")
    plt.ylabel("Predicted EAT")
    plt.savefig('Result.png')


# prepare dataset
train_set = ['1000', '2000', '4000', '8000', '10000', '20000', '30000']
op = 'EAT'
n_val = 5000

iX, iY = prepare_data(op)

# fit model and plot learning curves for a patience
patience = 500

current_dir = os.getcwd()

for ii in range(len(train_set)):
    n_test = len(iY) - train_set[ii] - n_val
    print('Trainset= {:}'.format(train_set[ii]))
    chdir(current_dir)
    os.chdir(current_dir + '/withdft/fchl/')
    try:
        os.mkdir(str(train_set[ii]))
    except:
        pass
    os.chdir(current_dir + '/withdft/fchl/' + str(train_set[ii]))


    model, lr, loss, mae, test_loader = fit_model_dense(
        int(train_set[ii]), int(n_val), int(n_test), iX, iY, patience
    )

    lhis = open('learning-history.dat', 'w')
    for ii in range(0, len(lr)):
        lhis.write(
            '{:8d}'.format(ii)
            + '{:16f}'.format(lr[ii])
            + '{:16f}'.format(loss[ii])
            + '{:16f}'.format(mae[ii])
            + '\n'
        )
    lhis.close()

    # Saving NN model
    torch.save(model, 'model.pt')
    
    # Saving results
    plotting_results(model, test_loader)
    save_plot(n_val)
