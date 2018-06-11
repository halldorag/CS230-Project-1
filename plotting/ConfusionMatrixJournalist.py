
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.ion()
A = np.array([[0.6, 0.13, 0.25], [0, 0.38, 0.625], [0.3, 0, 0.75]])
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        c = A[j,i]
        ax.text(i, j, str(c), va='center', ha='center',color='grey', fontsize=11)
ax.matshow(A,cmap=plt.cm.Blues)

x = ['Left','Right','Center']
y = ['Left','Right','Center']
plt.xticks(range(len(x)), x, fontsize=12)
plt.yticks(range(len(y)), y, fontsize=12)

plt.savefig('Matrix.png')