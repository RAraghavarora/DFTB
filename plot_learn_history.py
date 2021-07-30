import matplotlib.pyplot as plt

# Plot the curve of learning rate

lhis = open('cnn/2000/learning-history.dat', 'r')

lines = lhis.readlines()
x = []
y = []


for line in lines:
    epoch, lr, loss, mae = line.split()
    x.append(int(epoch))
    y.append(float(mae))

plt.plot(x[3000:], y[3000:], '.')
plt.xlabel("Training MAE")
plt.ylabel("Epoch")
plt.title('Learning history for conv with size of 20000')
plt.show()
plt.close()
