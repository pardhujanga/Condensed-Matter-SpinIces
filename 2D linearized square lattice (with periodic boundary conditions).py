# %% Defining and solving the Hamiltonian
from numba import jit
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from matplotlib import pyplot as plt

L=40 #periodicity of the lattice

@jit(nopython=True)
def kron(a,b): #kronecker delta function
    if a == b:
        return 1
    else:
        return 0


index = np.array(range(L**2))
length = np.array(range(L))

@jit(nopython=True)
def Hamiltonian():
    H = np.ones((L**2,L**2),dtype=np.complex128) #intializing the Hamiltonian
    for i in index: #NOTE the modulo is there to enforce periodic boundary conditions
        for x in length:
            for y in length:
                H[i][L*y + x] *= - kron(i,L*y+(x-1)%L) - kron(i,L*y+(x+1)%L) - kron(i,L*((y-1)%L)+x) - kron(i,L*((y+1)%L)+x)
    return H

h = Hamiltonian()
#solving for spectrum and eigenfunctions
val, vec = linalg.eig(h)
ordered_val = np.sort(val)

#generating basis to be able to visualize the eigenfunctions
#basis = np.zeros((L**2,L**2))
#for i in index:
    #basis[i][i]=1

# %%  Defining a function to find the expansion of eigenfunctions in the lattice basis
n = 550
eigf = np.abs(vec[:,n])
#phase = np.angle(vec[:,n])

# %%
# make data
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = eigf[L*Y + X]
#Zp = phase[L*Y + X]

# plot
fig, ax = plt.subplots()

im = ax.imshow(Z, cmap='plasma') #replace Z with Zp to see phase of eigenfunction across the lattice.
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im)
plt.show()


#NOTE The "phase" of the eigenfunction at each lattice point is just as important as the amplitude.
#Like, in the 2D no magnetic field system, the highest and lowest energy eigenfucntions
#both have very similar amplitude distributions. But looking at their phases, we see that
#the lowest energy state has uniform phase, whereas the highest energy one has fully alternating phases.

# %% code to create 2cos(k_x) + 2cos(k_y) to compare the energy spectrum with theory
x = 2*np.pi*np.array(range(1,L+1))/(L)
cos = np.cos(x)
theory = []
for i in cos:
    for j in cos:
        theory.append(2*i+2*j)

sorted = np.sort(theory)

plt.plot(ordered_val)
plt.plot(sorted)
plt.show()
# %%
plt.xlabel('n')
plt.ylabel('Energy (in multiples of hopping parameter t)')
plt.plot(sorted,'k')
plt.scatter(range(L**2),ordered_val,s=1)
plt.show()

# %%
