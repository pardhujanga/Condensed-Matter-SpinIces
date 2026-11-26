# %% This defines a Jupyter-like code cell, while still being in a .py file

from numba import jit
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

L = 40 # Size/periodicity of the lattice

@jit(nopython=True) # Numba decorator to allow this function to be compiled "just in time" (jit)
def kron(a,b): # Kronecker delta function
    if a == b:
        return 1
    else:
        return 0

# Rewritten in 2026 for clarity
index = np.arange(L**2)
length = np.arange(L)

# %% Creating the Hamiltonian for the model
@jit(nopython=True)
def Hamiltonian():
    H = np.ones((L**2,L**2),dtype=np.complex128) # Intializing the Hamiltonian
    for i in index: #NOTE The modulo is present in the kronecker function to enforce periodic boundary conditions
        for x in length:
            for y in length:
                H[i][L*y + x] *= - kron(i,L*y+(x-1)%L) - kron(i,L*y+(x+1)%L) - kron(i,L*((y-1)%L)+x) - kron(i,L*((y+1)%L)+x)
    return H

# %% Solving for spectrum and eigenfunctions
h = Hamiltonian()
eigenvalues, eigenvectors = linalg.eig(h)
sorted_eigenvalues = np.sort(eigenvalues)


# %% Creating the theoretical energy spectrum
x = 2*np.pi*np.arange(1,L+1)/(L)
cos = np.cos(x)
# Modified for clarity in 2026
theory_energy_values = [2*i + 2*j for i in cos  for j in cos]
sorted_theory_values = np.sort(theory_energy_values)

# Changing the render resolution to be able to see things more clearly
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Plotting the two solutions against each other
plt.xlabel('Energy level, n')
plt.ylabel('Energy (in multiples of hopping parameter t)')
plt.plot(sorted_theory_values,'k')
plt.scatter(range(L**2), sorted_eigenvalues,s=1, c="Red")
plt.show()

# %% Finding the lowest and highest energy eigenfunctions
print(f"The index of the lowest energy eigenfucntion is {np.argmin(eigenvalues)}")
print(f"The index of the highest energy eigenfucntion is {np.argmax(eigenvalues)}")


# %%  Plotting the eigenfunctions in the lattice basis. This is one way to study the system.

n = 0 # Choosing the eigenfunction we want to plot
Amplitude = np.abs(eigenvectors[:,n]) # We are selecting all the numbers in the nth column. This is our nth eigenfunction in a flattened form.

# Remapping the eigenfunction to the lattice basis (reversing the linearisation)
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = Amplitude[L*Y + X]

# Plotting the Amplitude
fig, ax = plt.subplots()

im_amp = ax.imshow(Z, cmap='plasma', vmin = 0, origin="lower")
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im_amp)
plt.show()

# %% Plotting the Phase
#NOTE The "phase" of the eigenfunction at each lattice point is just as important as the amplitude.
# In the 2D no magnetic field system, the highest and lowest energy eigenfucntions
# both have very similar amplitude distributions. But looking at their phases, we see that
# the lowest energy state has uniform phase, whereas the highest energy one has fully alternating phases.
# So, looking at phase map is just as valuable as looking at the amplitude.

Phase = np.angle(eigenvectors[:,n])
Zp = Phase[L*Y + X]

fig, ax = plt.subplots()

im_phase = ax.imshow(Zp, cmap='twilight_shifted', vmin = -3.14, vmax = 3.14, origin="lower") # This cmap has a cyclic colour scheme
ax.set_title("phase of eigenfunction on the lattice")
fig.colorbar(im_phase)
plt.show()


# %% BANDSTRUCTURE!!!!
sorted_eigenvalues = np.sort(eigenvalues)
energy_dict = []
for i in sorted_eigenvalues:
    energy_dict.append(list(eigenvalues).index(i))
energy_dict = np.array(energy_dict)

point_X = np.array([1,0],np.float64) # Point on the BZ boundary directly along the k_x direction in the reciprocal lattice 
point_Gamma = np.array([0,0],np.float64) # Origin in the reciprocal lattice
point_M = np.array([1,1],np.float64) # Corner of the square BZ

g = np.meshgrid(np.linspace(0,L-1,L,dtype=np.float64), np.linspace(0,L-1,L,dtype=np.float64))
r = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1) #this is the lattice

g_k = np.meshgrid(np.linspace(-1,1,L,endpoint=False,dtype = np.float64), np.linspace(-1,1,L,endpoint=False,dtype = np.float64))
k_space = np.pi*np.append(g_k[0].reshape(-1,1),g_k[1].reshape(-1,1),axis=1) #this is the reciprocal lattice, kinda

# %%
import time

@jit(nopython=True)
def factors():
    factors = np.empty(shape=(L**4,), dtype=np.complex128)
    for k in index:
        for p in index:
            factors[k*(L**2) + p] = np.exp(-1j * np.dot(k_space[k], r[p]))

    return factors

Factors = factors()

@jit(nopython=True)
def ft(m):
    eigfunc = eigenvectors[:,m]
    i = 0
    phi = np.empty(shape=(L**2,), dtype=np.int64)
    for k in index:
        phi_k = 0
        for p in index:
            phi_k += Factors[k*(L**2) + p] * eigfunc[p] # would having the exponential part be preccmputed be faster? that is the same for every single ft
        phi[i] = np.square(np.abs(phi_k))
        i += 1

    return np.reshape(phi, (L,L))

time_start = time.time()
FT_amp = ft(energy_dict[0])
time_end = time.time()
print(f"Took {time_end - time_start} seconds")

time_start = time.time()
FT_amp = ft(energy_dict[1])
time_end = time.time()
print(f"Took {time_end - time_start} seconds for the second time")

time_start = time.time()
FT_amp = ft(energy_dict[2])
time_end = time.time()
print(f"Took {time_end - time_start} seconds for the third time")


# %%
Origin = int(L/2)

@jit(nopython = True)
def bands():
    band = np.empty(shape = (3*L**2, int(L/2)), dtype=np.int64)
    for n in index:
        Fourier = ft(energy_dict[n])
        band[3*n] = Fourier[Origin: ,  Origin] #This is the section we want for Gamma to X 
        band[3*n +1] = Fourier[L-1,  Origin:] #This is the section we want for X to M
        band[3*n + 2] = np.array([Fourier[x,x] for x in np.arange(L-1,Origin-1,-1)]) #This is the section we want for M to Gamma
    return band

bds = bands()

# %% plotting
a = int(3*L/2)
X, Y = np.meshgrid(np.linspace(0, a-1, a,dtype=int), sorted_eigenvalues)
Z = np.reshape(bds,(L**2, a)) # needs changing to fit the size of bds properly
Bands = Z

# plot
fig, ax = plt.subplots()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.ylabel("Energy")
plt.xlabel("k")
plt.vlines([Origin-1, L-1],-4,4,"k")
plt.xticks([0,Origin-1,L-1,a-1], ['\N{GREEK CAPITAL LETTER GAMMA}','X','M','\N{GREEK CAPITAL LETTER GAMMA}'])
im = ax.pcolormesh(X, Y, Bands, vmax = 5, cmap='Greys')
ax.set_title("Band structure")
fig.colorbar(im)
plt.show()

# %%
