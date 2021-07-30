import matplotlib.pyplot as plt
train_set = [int(i) for i in ['1000', '2000', '4000',
                              '8000', '10000', '20000', '30000']]


y1 = []  # Only molecular descriptors
y2 = []  # Mol Desc + DFTB
y3 = []  # Separate models (without tuner)
y4 = []  # Separate models (with tuner)
y5 = []  # Separate models (with tuner) new model

for i in train_set:
    try:
        f1 = open("%s/errors.dat" % i, 'r')
        f2 = open("withdft/%s/errors.dat" % i, 'r')
        f3 = open("conv/withdft/%s/errors.dat" % i, 'r')
        f4 = open("conv2/%s/errors.dat" % i, 'r')
        f5 = open("conv2/new/%s/errors.dat" % i, 'r')
    except:
        print("*******\n")
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

    lines5 = f5.readlines()
    a, b, c = lines5[0].split()
    y5.append(round(float(b), 3))  # Rounded upto 3 decimal places

y3 = []
for i in train_set:
    try:
        f = open("conv/withdft/%s/errors_test.dat" % i, 'r')
    except:
        print(i)
        continue
    lines = f.readlines()
    a, b, c = lines[0].split()
    y3.append(round(float(b), 3))

print(y1)
print(y2)
print(y3)
print(y4)
print(y5)

plt.plot(train_set, y1, 's-', label='Without dftb')
plt.plot(train_set, y2, 's-', label='With dftb')
plt.plot(train_set, y3, 'o:', label='Separate models without tuner')
plt.plot(train_set, y4, 'o:', label='Separate models with tuner')
plt.plot(train_set, y5, 'o:', label='Separate models with tuner (new model)')

plt.annotate('(%s, %s)' % (30000, y1[-1]),
             xy=(30000, y1[-1]), textcoords='data')
plt.annotate('(%s, %s)' % (20000, y1[-2]),
             xy=(20000, y1[-2]), textcoords='data')

plt.annotate('(%s, %s)' % (30000, y2[-1]),
             xy=(30000, y2[-1]), textcoords='data')
plt.annotate('(%s, %s)' % (20000, y2[-2]),
             xy=(20000, y2[-2]), textcoords='data')

plt.xlabel('Training size')
plt.ylabel('MAE (eV)')
plt.title('Learning Curve of simple Sequential network')
plt.legend()
plt.show()


