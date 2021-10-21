#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import schnetpack as spk
import shap
from tensorflow.keras.models import load_model
from qml.representations import generate_coulomb_matrix

#%%
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


#%%
data_dir = '../'
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
print(n)
idx = np.arange(n)
np.random.seed(2314)
idx2 = np.random.permutation(idx)

#%%
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
    TPROP.append(float(props['EAT']))
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

xyz_reps = np.array(
    [generate_coulomb_matrix(Z[mol], xyz[mol], sorting='unsorted') for mol in idx2]
)
AE = np.array(AE)
EGAP = np.array(EGAP)
TPROP = np.array(TPROP)

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

#%%
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

#%%
properties = [
    'FermiEne',
    'BandEne',
    'NumElec',
    'h0Ene',
    'sccEne',
    '3rdEne',
    'RepEne',
    'mbdEne',
    'TBdip',
    'TBeig1',
    'TBeig2',
    'TBeig3',
    'TBeig4',
    'TBeig5',
    'TBeig6',
    'TBeig7',
    'TBeig8',
    'TBchg1',
    'TBchg2',
    'TBchg3',
    'TBchg4',
    'TBchg5',
    'TBchg6',
    'TBchg7',
    'TBchg8',
    'TBchg9',
    'TBchg10',
    'TBchg11',
    'TBchg12',
    'TBchg13',
    'TBchg14',
    'TBchg15',
    'TBchg16',
    'TBchg17',
    'TBchg18',
    'TBchg19',
    'TBchg20',
    'TBchg21',
    'TBchg22',
    'TBchg23',
]
print(len(properties))

data = pd.DataFrame(data=reps2, columns=properties)
data["target"] = TPROP2

#%%
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.set_style(style='white')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
plot = sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 1},
    ax=ax,
)

#%%
fig = plot.get_figure()
fig.savefig("heatmap.png")


#%%
import numpy as np

np.save('reps.npy', reps2)

#%%
# Use the feature importance on a trained model
model = load_model('withdft/30000' + '/model.h5')
explainer = shap.DeepExplainer(model)
shap_values = explainer(reps2)
        