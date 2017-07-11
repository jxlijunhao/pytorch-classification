import matplotlib.pyplot as plt

model_name = 'resnext26_32x4d'

logfile = './checkpoints/imagenet/{}/log.txt'.format(model_name)

f = open(logfile, 'r')

train_acc = []
val_acc = []

for i in f:
    if i.strip().split('\t')[0] == 'Learning Rate':
        continue
    train_acc.append(float(i.strip().split('\t')[3]))
    val_acc.append(float(i.strip().split('\t')[4]))

# print train_acc, val_acc

plt.plot(train_acc)
plt.plot(val_acc)

plt.legend(['{}-train-acc'.format(model_name), '{}-val-acc'.format(model_name)])
plt.grid(True)

plt.savefig('{}-log.png'.format(model_name), dpi=150)

