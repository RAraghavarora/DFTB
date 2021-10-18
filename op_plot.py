#%%
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def calc_fwhm(z, log_scale=None):
    try:
        ax = sns.kdeplot(z, log_scale=log_scale)
        kde_curve = ax.lines[0]
    except:
        pdb.set_trace()
    x = kde_curve.get_xdata()
    y = kde_curve.get_ydata()
    halfmax = y.max() / 2
    maxpos = y.argmax()
    leftpos = (np.abs(y[:maxpos] - halfmax)).argmin()
    rightpos = (np.abs(y[maxpos:] - halfmax)).argmin() + maxpos
    fullwidthathalfmax = x[rightpos] - x[leftpos]
    plt.close()
    return fullwidthathalfmax


def get_errors(file):
    f = open(file, 'r')
    lines = f.readlines()
    z = []
    print(len(lines))

    for line in lines:
        x1, y1, z1 = line.split()
        if z1 == 0.000000:
            z1 = 0.000001
        z.append(float(z1))

    z = np.array(z)
    print("STD=", float(z.std()))
    return z


z1 = get_errors("withdft/30000/comp.dat")
z2 = get_errors("only_CM/30000/comp.dat")
z3 = get_errors("standard/30000/comp-test.dat")
z4 = get_errors("slatm/distorted/30000/comp-test.dat")
z5 = get_errors("slatm/nodft/30000/comp-test.dat")
z6 = get_errors("normalize/30000/comp-test.dat")
z7 = get_errors("normalize/only_bob/30000/comp-test.dat")
z8 = get_errors("normalize/eq/30000/comp-test.dat")
z9 = get_errors("slatm/30000/comp-test.dat")
z10 = get_errors("withdft/eq/30000/comp-test.dat")



# fwhm = calc_fwhm(z1)
# print(fwhm)
# fwhm = calc_fwhm(z2)
# print(fwhm)
# fwhm = calc_fwhm(z3, 10)
# print(fwhm)


# z = pd.DataFrame([z1, z2], columns=['withdft,without_dft'])
z = pd.DataFrame()
z['CM + DFTB'] = z10[:10000]
z['SLATM + DFTB'] = z9[:10000]
z['BOB + DFTB'] = z8[:10000]
# z["CM + DFTB (using standardization)"] = z6
# z['SLATM (with dftb)'] = z3[:10000]
# z['SLATM (without dftb)'] = z5[:10000]
# z['BOB (with dftb)'] = z6
# z['BOB'] = z7


sns.set_theme()

plot = sns.displot(
    data=z, kde=True, bins=700, legend=True, aspect=1.8,# palette=['green','darkorange', 'red']
).set(title='Error Distribution on Equilibrium molecules for train size=30000')
plt.xlim([-1.5,1.5])
plot.set_axis_labels("Predicted EAT - True EAT", "Count")
ax = plot.axes[0][0]
# ax.text(10 ** (-4), 150, "FWHM=%s" % fwhm, fontsize=14)  # add text


plt.show()
fig = plot.get_figure()
fig.savefig("Beta.png")
