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


# %% code to create 2cos(k_x) + 2cos(k_y) to compare the energy spectrum with theory
x = np.pi*np.array(range(L))/(L-1)
cos = np.cos(x)
theory = []
for i in cos:
    for j in cos:
        theory.append(2*i+2*j)

sorted = np.sort(theory)

plt.scatter(index,ordered_val,s=0.1)
#plt.plot(sorted)
plt.show()
# %%
# gamma to X
g = np.meshgrid(np.linspace(0,L-1,L,dtype=np.float64), np.linspace(0,L-1,L,dtype=np.float64))
r = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1)

# %% BANDS!!!
ordered_val = np.sort(val)
energy_dict = []
for i in ordered_val:
    energy_dict.append(list(val).index(i))
energy_dict = np.array(energy_dict)

l = L
xlength = np.array(range(L))
point_X = np.array([1,0],np.float64)
eta = 2*np.pi*np.array(range(l))/(l)

@jit(nopython=True)
def ft(m):
    eigfunc = vec[:,m]
    phi = []
    for e in xlength:
        phi_e = 0
        for p in index:
            phi_e += np.exp(-1j*eta[e]*np.dot(point_X,r[p]))*eigfunc[p]
        phi.append(np.square(np.abs(phi_e)))

    return phi

@jit(nopython = True)
def bands():
    band = []
    for n in index:
        band.append(ft(energy_dict[n]))
    return band

bds = bands()
# %% plotting
X, Y = np.meshgrid(np.linspace(0,l-1,l,dtype=int), np.linspace(0,L**2-1,L**2,dtype=int))
#Z = bds[Y][X]

# plot
fig, ax = plt.subplots()
import matplotlib.colors as colour

im = ax.pcolormesh(bds, cmap='plasma')
ax.set_title("Band structure")
fig.colorbar(im)
plt.show()
# %% ORDERED BANDS!!!
ordered_val = np.sort_complex(val)
energy_dict = []
for i in ordered_val:
    energy_dict.append(list(val).index(i))
energy_dict = np.array(energy_dict)

l = L
xlength = np.array(range(L))
point_X = np.array([1,0],np.float64)
point_Y = np.array([0,1],np.float64)
eta = np.pi*np.array(range(-l-1,l,2))/(l)

@jit(nopython=True)
def ft(m):
    eigfunc = vec[:,m]
    phi = []
    for e in xlength:
        phi_e = 0
        for p in index:
            phi_e += np.exp(-1j*eta[e]*np.dot(point_X,r[p]) )*eigfunc[p]
        phi.append(np.square(np.abs(phi_e)))

    return phi

@jit(nopython = True)
def bands():
    band = []
    n=0
    while n < L**2: 
        p=0          
        k = np.array(ft(energy_dict[n]))
        for l in index:
            if l!=n and np.abs(ordered_val[n]-ordered_val[l])<10**-3:
                k = k+np.array(ft(energy_dict[l]))
                p+=1
        n += 1+p
        band.append(list(k))
    return band

bds = bands()
# %%
