# %%
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from matplotlib import pyplot as plt

L=100

def kron(x,y):
    if x == y:
        return 1
    else:
        return 0

t = np.ones(L)

#Hamiltonian
H = np.zeros((L,L))

for i in range(L):
    for j in range(L):
        H[i][j] = - kron(i,(j-1)%L)*t[(j-1)%L] - kron(i,(j+1)%L)*t[j]


val,vec = linalg.eig(H)
ordered_val = np.sort_complex(val)

x = 2*np.pi*np.array(range(1,L+1))/(L)
cos = np.cos(x)
theory = []
for i in cos:
    theory.append(2*i)

sorted = np.sort(theory)

plt.xlabel('n')
plt.ylabel('Energy (in multiples of hopping parameter t)')
plt.plot(sorted,'k')
plt.scatter(range(L),ordered_val,s=10)
plt.show()

#as expected it is a sine curve. Computed the values of cos(k)
#which are the exact eigenvalues, with k being quantized. Match exactly.
#Also make the shift to sparse matrices

