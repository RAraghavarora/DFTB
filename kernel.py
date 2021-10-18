# NN model
import numpy as np
from qml.representations import generate_coulomb_matrix
import logging
import pdb
import schnetpack as spk
from qml.math import cho_solve
from qml.kernels import gaussian_kernel
from scipy.optimize import dual_annealing
import random


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

    dataset = spk.data.AtomsData(data_dir + 'totgdb7x_pbe0.db', load_only=properties)

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

    reps2 = np.array(reps2)
    return reps2, TPROP2, atoms_data


Repre, Target, atoms_data = prepare_data('EAT')
Target = np.array(Target)


def objective(params):
    global Repre
    global Target
    sigma, gamma = params
    print("sigma=", sigma)
    print("gamma=", gamma)

    n_test = 10000
    n_val = 1000

    train_set = [1000,2000,4000,8000,10000,20000,30000]

    # try:
    #     indices = np.arange(desc.shape[0])
    #     np.random.shuffle(indices)
    #     desc = desc[indices]
    #     dftb = dftb[indices]
    #     Target = Target[indices]
    # except Exception as e:
    #     print(e)
    #     pdb.set_trace()

    for n_train in train_set:
        X_train = np.array(Repre[:n_train])
        X_test = np.array(Repre[-n_test:])

        Y_train, Y_val, Y_test = (
            np.array(Target[:n_train]),
            np.array(Target[-n_test - n_val : -n_test]),
            np.array(Target[-n_test:]),
        )

        K = gaussian_kernel(X_train, X_train, sigma)
        K[np.diag_indices_from(K)] += gamma  # Regularizer
        alpha = cho_solve(K, Y_train)  # α=(K+λI)−1y
        Ks = gaussian_kernel(X_test, X_train, sigma)
        Y_predicted = np.dot(Ks, alpha)

        # Writing the true and predicted EAT values
        dtest = np.array(Y_test - Y_predicted)
        format_list1 = ['{:16f}' for item1 in Y_test[0]]
        s = ' '.join(format_list1)
        ctest = open('comp-test_%s.dat'%n_train, 'w')
        for ii in range(0, len(Y_test)):
            ctest.write(
                s.format(*Y_test[ii]) + s.format(*Y_predicted[ii]) + s.format(*dtest[ii]) + '\n'
            )
        ctest.close()

        MAE_PROP = float(mean_absolute_error(Y_test, Y_predicted))
        MSE_PROP = float(mean_squared_error(Y_test, Y_predicted))
        STD_PROP = float(Y_test.std())

        out2 = open('errors_test%s.dat'%n_train, 'w')
        out2.write(
            '{:>24}'.format(STD_PROP)
            + '{:>24}'.format(MAE_PROP)
            + '{:>24}'.format(MSE_PROP)
            + "\n"
        )
        out2.close()

    res = np.mean(np.abs(Y_predicted - Y_test))
    print(res)
    return res

gamma = 9.990000000000001e-06
sigma= 135.55025590223175
objective([sigma,gamma])