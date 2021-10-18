import numpy as np
import matplotlib.pyplot as plt
import pdb
f1 = open("slatm/30000/comp-test.dat", 'r')
f2 = open("slatm/nodft/eq/30000/comp-test.dat", 'r')
# f3 = open("standard/30000/comp-test.dat", 'r')

lines = f1.readlines()
X1 = []
Y1 = []
mini = float(lines[0].split()[0])
maxi = float(lines[0].split()[0])
for line in lines:
    x1, y1, z1 = line.split()
    X1.append(float(x1))
    Y1.append(float(y1))
    if float(x1) < mini:
        mini = float(x1)
    if float(x1) > maxi:
        maxi = float(x1)

lines = f2.readlines()
X2 = []
Y2 = []
for line in lines:
    x1, y1, z1 = line.split()
    X2.append(float(x1))
    Y2.append(float(y1))
    if float(x1) < mini:
        mini = float(x1)
    if float(x1) > maxi:
        maxi = float(x1)

X2 = X2[:10000]


# lines = f3.readlines()
# X3 = []
# Y3 = []
# for line in lines:
#     x1, y1, z1 = line.split()
#     X3.append(float(x1))
#     Y3.append(float(y1))


temp = np.arange(mini, maxi, 0.1)
plt.plot(temp, temp)
print(mini, '\t', maxi)
plt.plot(X2, Y2, '.', label="Using only SLATM", alpha=0.3)
plt.plot(X1, Y1, '.', label="Using SLATM with DFTB properties", alpha=0.3)
# plt.plot(X3, Y3, '.', label="Architecture 2 with data standardization")

plt.xlabel("True EAT")
plt.ylabel("Predicted EAT")
plt.title("Result for train size of 30000 on dataset of non-equilibrium molecules")
plt.legend(loc="upper left")
plt.show()
# plt.savefig('Combined_comparison2.png')
plt.close()