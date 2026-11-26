# %% Defining and solving the Hamiltonian for a closed n-gon lattice in a uniform magnetic field perpendicular to it
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

n = 4 # For n = 4, we have equivalence with L = 2 for the 2D square lattice with unifrom magnetic field.

def kron(x,y): # Kronecker Delta function
    if x == y:
        return 1
    else:
        return 0

#NOTE Problem: this is a case which is supposed to be identical to the L=2 case for the general lattice 
# (This is a closed chain of 4) think of why the general one isn't working by comparing it to this. 
# It might be helpful to do expand all the terms in your general case to see how difference arises. 
#NOTE Solution: Did that and also looked at the matrix rep of the H in the linearized basis. 
# This helped me see an unnecessary interaction, which was messing up everything at low lattice size, 
# but as size increased it became less relevant, so something like the butterfly appeared.
# But I removed the unnecessary interaction and things work exactly as they should. Also, this interaction was
#covered up in the no mag field case due to Periodic boundary conditions.

length = range(n)

H = np.zeros((n,n),dtype=np.complex128) #intializing the hamiltonian 

def energy(b):
    for i in length:
        for x in length:
            H[i][x] = - kron(i,(x-1)%n)*np.exp(-1j*b/n) - kron(i,(x+1)%n)*np.exp(1j*b/n) 
           
    #solving for spectrum and eigenfunctions
    eigenvalues, eigenvectors = linalg.eig(H)
    return eigenvalues

g = 400 # gradations
B = 2*np.pi*np.array(range(g+1))/g # magnetic field values

# %% Plotting the data (slightly faster than graphing the bands)
points = []
for i in range(len(B)):
    e = energy(B[i])
    for j in range(len(e)):
        points.append((B[i],e[j]))

plt.scatter(*zip(*points),s=1,marker=".")
plt.show()

# %%
