import matplotlib.pyplot as plt
train_set = [int(i) for i in ['1000', '2000', '4000', '8000', '10000', '20000', '30000']]

y1 = []
for i in train_set:
    try:
        f = open("%s/errors_test.dat" % i, 'r')
    except:
        print(i)
        continue
    lines = f.readlines()
    a, b, c = lines[0].split()
    y1.append(round(float(b), 3))


y2 = []
for i in train_set:
    try:
        f = open("withdft/%s/errors_test.dat" % i, 'r')
    except:
        print(i)
        continue
    lines = f.readlines()
    a, b, c = lines[0].split()
    y2.append(round(float(b), 3))

print(y1)
print(y2)

plt.plot(train_set, y1, 'go-', label = 'Without dftb')
plt.plot(train_set, y2, 'bo-', label = 'With dftb')

plt.annotate('(%s, %s)' % (30000, y1[-1]), xy=(30000, y1[-1]), textcoords = 'data') 
plt.annotate('(%s, %s)' % (20000, y1[-2]), xy=(20000, y1[-2]), textcoords = 'data') 

plt.annotate('(%s, %s)' % (30000, y2[-1]), xy=(30000, y2[-1]), textcoords = 'data') 
plt.annotate('(%s, %s)' % (20000, y2[-2]), xy=(20000, y2[-2]), textcoords = 'data') 


plt.xlabel('Training size')
plt.ylabel('MAE (eV)')
plt.title('Learning Curve of simple Sequential network')
plt.legend()
plt.show()
