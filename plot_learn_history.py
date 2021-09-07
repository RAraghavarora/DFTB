import matplotlib.pyplot as plt

# Plot the curve of learning rate

lhis = open('conv2/new2/20000/learning-history.dat', 'r')

lines = lhis.readlines()
x = []
y = []


for line in lines:
    epoch, lr, loss, val_mae, mae = line.split()
    x.append(int(epoch))
    y.append(float(val_mae))

plt.plot(x[10000:], y[10000:], '.')
plt.xlabel("Training MAE")
plt.ylabel("Epoch")
plt.title('Learning history for conv with size of 20000')
plt.show()
plt.close()
