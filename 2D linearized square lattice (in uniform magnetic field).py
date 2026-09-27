# %% Defining and solving the Hamiltonian for a 2D lattice in a uniform magnetic field perpendicular to it
from numba import jit
import numpy as np
import scipy.sparse as sp
from numpy import linalg #NOTE numba doesn't recognize scipy. can be made to if needed
from matplotlib import pyplot as plt

L=20

@jit
def kron(x,y):
    if x == y:
        return 1
    else:
        return 0


index = np.array(range(L**2))
length = np.array(range(L))

@jit(nopython = True)
def energy(b):
    H = np.zeros((L**2,L**2),dtype=np.complex128) #intializing the hamiltonian 
    for i in index:
        for x in length:
            for y in length:
                if x!=0 and x!=L-1:
                    H[i][L*y + x] = - kron(i,L*y+(x-1))*np.exp(1j*b*y/2) - kron(i,L*y+(x+1))*np.exp(-1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
                elif x==0:
                    H[i][L*y + x] = - kron(i,L*y+(x+1))*np.exp(-1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
                elif x==L-1:
                    H[i][L*y + x] = - kron(i,L*y+(x-1))*np.exp(1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
    #NOTE Needed to split the hamiltoninan generation as the way the lattice to avoid interactions between
    # endpoints of consecutive "lines of atoms". Fixed it and removed Periodic boundary conditions as 
    # Hofstatder is meant to be on a normal finite lattice. The periodicity arises naturally when we have
    #rational flux ratio (all B in our case as we can only store rational numbers)

    #solving for spectrum and eigenfunctions
    return linalg.eig(H)
     

n = 400
B = 2*np.pi*np.array(range(n+1))/n

# %%
val,vec = energy(B[200])
# %% Plotting the energy spectrum
x = 2*np.pi*np.array(range(1,L+1))/(L)
cos = np.cos(x)
theory = []
for i in cos:
    for j in cos:
        theory.append(2*i+2*j)

sorted = np.sort(theory)

ordered_val = np.sort(val)
plt.plot(ordered_val,'k')
plt.plot(sorted)
plt.show()

#NOTE:The energy spectrum varies just a bit from the no magnetic field case.
# Could be due to the size of the lattice wahing out the effect of the monopole.

# %%
plt.xlabel('n')
plt.ylabel('Energy (in multiples of hopping parameter t)')
plt.plot(sorted,'k')
plt.scatter(range(L**2),ordered_val,s=1)
plt.show() 

# %% Plotting data in bands
bands = np.zeros((L**2,n+1))
for p in np.array(range(n+1)):
    ener,level = energy(B[p]) 
    e = np.sort(ener)
    for q in np.array(range(len(e))):
        bands[q][p]=e[q]
for p in np.array(range(L**2)):
    plt.plot(bands[p],'k',linewidth = 0.1)
plt.show()


# %%  Choosing a mag field strength to find the expansion of eigenfunctions in the lattice basis
a =100 #picking the strength of mag field
E,landau_levels = energy(B[a])

# %% 
n = 1 #picking the eigenfucntion to display (between 0 and L**2 -1 inclusive)

# %% plotting the eigenfunctions
eigf = np.abs(landau_levels[:,n])

# make data
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = eigf[L*Y + X]

# plot
fig, ax = plt.subplots()

im = ax.imshow(Z, cmap='plasma')
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im)
plt.show()

# %% plotting the phase of the eigenfunctions
phase = np.angle(landau_levels[:,n])

X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Zp = phase[L*Y + X]

# plot
fig, ax = plt.subplots()

im = ax.imshow(Zp, cmap='plasma')
ax.set_title("phase of eigenfunction on the lattice")
fig.colorbar(im)
plt.show()
