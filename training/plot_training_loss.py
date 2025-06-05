import numpy as np
import matplotlib.pyplot as plt

loss = np.loadtxt('training_loss.csv', delimiter=',')
num_ranks = 0
while loss[num_ranks][1] == 0:
    num_ranks += 1

loss = loss[:, 2]
averaged_loss = [np.mean(loss[i:i+num_ranks]) for i in range(0, len(loss), num_ranks)]
fontsize = 20
plt.plot(list(range(0, len(loss)//num_ranks*100, 100)), averaged_loss, label='Training Loss')
plt.xlabel('Training Steps', fontsize=fontsize)
plt.ylabel('Loss', fontsize=fontsize)
plt.yscale('log')
plt.savefig('training_loss_plot.png', bbox_inches='tight')