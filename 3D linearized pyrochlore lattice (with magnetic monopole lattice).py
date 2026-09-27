# %% Defining and solving the Hamiltonian for a 3D pyrocholore lattice with a monopole lattice (alternating) (actually visons)
from numba import jit
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from matplotlib import pyplot as plt

L=8
index = np.array(range(8*L**3))
xlength = np.array(range(L))
ylength = np.array(range(2*L))
zlength = np.array(range(4*L))

@jit(nopython = True)
def kron(x,y):
    if x == y:
        return 1
    else:
        return 0

#NOTE: 
# Problem --> Numba says H is read only, but it isn't. How do we fix this?
#Answer --> Numba can't modify global variables in "nopython" mode. So, change
#code to accomodate that.

df = pd.read_csv('phases.csv')

# This is much faster at making the Hamiltonian
@jit(nopython = True)
def Hamiltonian():
    M = np.ones((8*L**3,8*L**3),dtype=np.complex128) #intializing the hamiltonian 
    for i in index:
        for x in xlength:
            for y in ylength:
                for z in zlength:
                    if (z%2==0):
                        if ((z%4)/2 + y%2 == 1):
                            M[i][z*2*L**2 + y*L + x] *=  kron(i,((z+1)%(4*L))*2*L**2 + ((y-1)%(2*L))*L + (x)%L) + kron(i,((z-1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x)%L) + kron(i,((z+1)%(4*L))*2*L**2 + ((y-1)%(2*L))*L + (x)%L) + kron(i,((z+1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x)%L)
                        else:
                            M[i][z*2*L**2 + y*L + x] *=  kron(i,((z-1)%(4*L))*2*L**2 + ((y-1)%(2*L))*L + (x-1)%L) + kron(i,((z-1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x)%L) + kron(i,((z+1)%(4*L))*2*L**2 + ((y-1)%(2*L))*L + (x)%L) + kron(i,((z+1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x-1)%L)
                    else:
                        if (((z-1)%4)/2 + y%2 == 1):
                            M[i][z*2*L**2 + y*L + x] *=  kron(i,((z-1)%(4*L))*2*L**2 + ((y+1)%(2*L))*L + (x)%L) + kron(i,((z-1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x)%L) + kron(i,((z+1)%(4*L))*2*L**2 + ((y+1)%(2*L))*L + (x)%L) + kron(i,((z+1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x)%L)
                        else:
                            M[i][z*2*L**2 + y*L + x] *=  kron(i,((z+1)%(4*L))*2*L**2 + ((y+1)%(2*L))*L + (x+1)%L) + kron(i,((z+1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x)%L) + kron(i,((z-1)%(4*L))*2*L**2 + ((y+1)%(2*L))*L + (x)%L) + kron(i,((z-1)%(4*L))*2*L**2 + ((y)%(2*L))*L + (x+1)%L)

                    #This is with periodic boundary conditions (suggested for both full monopole lattice and the disorder study)
    return -M


H = Hamiltonian()

#NOTE: this is the bottle neck now. Making the Hamiltonian is very quick
#solving for spectrum and eigenfunctions
val, vec = linalg.eig(H)

# %% Plotting the energy spectrum
x = np.pi*np.cos(0.5*np.pi*np.array(range(L))/(L-1)) #max is arccos(-1/3)
y = np.pi*np.cos(0.5*np.pi*np.array(range(2*L))/(2*L-1))
z = np.pi*np.cos(0.5*np.pi*np.array(range(4*L))/(4*L-1))
#NOTE the expression for the argument of the cosines in the spectrum are of the form sqrt(3)*k*cos(0 - 109.47 degrees).
# That's why the arguments are made to be sampled as a cosine. 
# the range of the cosine should be slighlty larger and also we have additional restrictions on
#  which "triplets" of angles are allowed (haven't implemeted that). (to get the true distribution of "k_bar.theta_bar", we need to 
# sample k over a sphere and take the dot product with the three vectors, i.e. just do to the full fat computation)
# As long as the range of energies is the same, the only thing affecting our coded spectrum is how we sample the argument of the cosines in energy.

theory = []
l=0
a = np.sqrt(3) #this is to account for the magnitude of the hopping vector which is sqrt(3)
for i in x:
    for j in y:
        for k in z:
            energy = 4 + 2*(np.cos(a*i)+np.cos(a*j)+np.cos(a*k)+np.cos(a*i-a*j)+np.cos(a*j-a*k)+np.cos(a*k-a*i))
            l += 1
            if l%2 == 1:
                theory.append(np.sqrt(energy))
                theory.append(-np.sqrt(energy))

sorted = np.sort(theory)

ordered_val = np.sort(val)
plt.plot(ordered_val,'k')
plt.plot(sorted)
plt.show()

# %%
print(np.argmax(val))

# %%
n=0
# %%  Defining a function to find the expansion of eigenfunctions in the lattice basis
eigf = np.abs(vec[:,n]) 

#NOTE might need to visualize each plane seperately as this is a 3D lattice
# defining the lattice
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = eigf[L*Y + X]

# plot
fig, ax = plt.subplots()

im = ax.imshow(Z, cmap='plasma')
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im)
plt.show()

ipr = np.sum(np.square(np.square(eigf))) #inverse participation ratio - measures localization - keep for 3d
print("Inverse participation ratio is " + str(ipr))

# %% plotting the phase of the eigenfunctions
phase = np.angle(vec[:,n])

X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Zp = phase[L*Y + X]

# plot
fig, ax = plt.subplots()

im = ax.imshow(Zp, cmap='plasma')
ax.set_title("phase of eigenfunction on the lattice")
fig.colorbar(im)
plt.show()
# %%
plt.plot(val)
plt.show()
# %%