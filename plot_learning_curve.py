import matplotlib.pyplot as plt
import numpy as np

train_set = [
    int(i) for i in ['1000', '2000', '4000', '8000', '10000', '20000', '30000']
]


y1 = []  # Only molecular descriptors
y2 = []  # Mol Desc + DFTB
y3 = []  # Kernel
y4 = []  # Desc + DFTB (with standard)


for i in train_set:
    try:
        f1 = open("only_CM/%s/errors.dat" % i, 'r')
        f2 = open("withdft/%s/errors.dat" % i, 'r')
        f3 = open("errors_test%s.dat" % i, 'r')
        f4 = open("standard/%s/errors_test.dat" % i, 'r')
    except Exception as e:
        print("*******\n")
        print(e)
        print(i)
        continue
    lines1 = f1.readlines()
    a, b, c = lines1[0].split()
    y1.append(round(float(b), 3))  # Rounded upto 3 decimal places

    lines2 = f2.readlines()
    a, b, c = lines2[0].split()
    y2.append(round(float(b), 3))  # Rounded upto 3 decimal places

    lines3 = f3.readlines()
    a, b, c = lines3[0].split()
    y3.append(round(float(b), 3))  # Rounded upto 3 decimal places

    lines4 = f4.readlines()
    a, b, c = lines4[0].split()
    y4.append(round(float(b), 3))  # Rounded upto 3 decimal places

print(y1)
print(y2)
print(y3)
print(y4)

# plt.yscale("log") 
plt.grid(True, which="both")
plt.loglog(train_set, y1, 's-', label='Architecture 1')
plt.loglog(train_set, y2, 's-', label='Architecture 2 Without standardization')
plt.loglog(train_set, y3, 'o:', label='Kernel')
plt.loglog(train_set, y4, 'o:', label='Architecture 2 With Data Standardization')

# plt.annotate('(%s, %s)' % (30000, y1[-1]), xy=(30000, y1[-1]), textcoords='data')
# plt.annotate('(%s, %s)' % (20000, y1[-2]), xy=(20000, y1[-2]), textcoords='data')

# plt.annotate('(%s, %s)' % (30000, y2[-1]), xy=(30000, y2[-1]), textcoords='data')
# plt.annotate('(%s, %s)' % (20000, y2[-2]), xy=(20000, y2[-2]), textcoords='data')

# plt.annotate('(%s, %s)' % (30000, y4[-1]), xy=(30000, y4[-1]), textcoords='data')



plt.xlabel('Training size')
plt.ylabel('MAE (eV)')
plt.title('Learning Curve of simple Sequential network')
plt.legend()
plt.show()
