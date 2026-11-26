# %% Defining and solving the Hamiltonian for a 2D lattice in a uniform magnetic field perpendicular to it
from numba import jit
import numpy as np
import scipy.sparse as sp # Tried sparse matrix techniques, but my Hamiltonian was not sparse
from numpy import linalg #NOTE numba doesn't recognize scipy. can be made to if needed
from matplotlib import pyplot as plt
import time

L = 60 # Size of the lattice

@jit(nopython = True) # Numba decorator to allow this function to be compiled "just in time" (jit)
def kron(x,y): # Kronecker delta function
    if x == y:
        return 1
    else:
        return 0


# Modified for clarity in 2026
index = np.arange(L**2)
length = np.arange(L)


@jit(nopython = True)
def energy(b):
    H = np.zeros((L**2,L**2),dtype=np.complex128) # intializing the Hamiltonian 
    for i in index:
        for x in length:
            for y in length:
                if x!=0 and x!=L-1:
                    H[i][L*y + x] = - kron(i,L*y+(x-1))*np.exp(1j*b*y/2) - kron(i,L*y+(x+1))*np.exp(-1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
                elif x==0:
                    H[i][L*y + x] = - kron(i,L*y+(x+1))*np.exp(-1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
                elif x==L-1:
                    H[i][L*y + x] = - kron(i,L*y+(x-1))*np.exp(1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
    
    #NOTE Needed to split the hamiltoninan generation to avoid interactions between endpoints of consecutive "lines of atoms"
    # Removed Periodic boundary conditions as Hofstatder is meant to be on a normal finite lattice.
    # The periodicity arises naturally when we have rational flux ratio (all B in our case as we can only store rational numbers).

    # solving for spectrum and eigenfunctions
    # included in the Hamiltonian function as we call this several times in this script unlike in the other ones.
    return linalg.eig(H)
     

# Modified in 2026 for clarity
g = 400 # Gradations in the magnetic field
B = 2*np.pi*np.arange(g+1)/g # An array of magnetic field values (this is technically the flux ratio, but is physically the same idea)

# %%
eigenvalues,eigenvectors = energy(B[200])
sorted_eigenvalues = np.sort(eigenvalues)

# %%
# Creating the theoretical energy spectrum for the no-magnetic field case.
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
plt.scatter(range(L**2), sorted_eigenvalues, s=1)
plt.show() 


# %% Producing Hofstadter's butterfly
# Rewritten for clarity in 2026
Hofstatder_bands = np.zeros((L**2,g+1))
for p in np.arange(g+1):
    ener,level = energy(B[p]) 
    e = np.sort(ener)
    for q in np.arange(len(e)):
        Hofstatder_bands[q][p]=e[q] 
        # each energy level's energy values at a given magnetic field are stored
        # in the corresponding 1D array within the overall 2D array.

# %% Plotting the Hofstadter's butterfly (Energy levels vs Magnetic field)
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500

for p in np.arange(L**2):
    plt.plot(np.linspace(0,2*np.pi, len(Hofstatder_bands[p])), Hofstatder_bands[p], 'k',linewidth = 0.1)

plt.xlabel('Flux Ratio')
plt.ylabel('Energy')
plt.show()


# %%  Choosing a mag field strength to find the expansion of eigenfunctions in the lattice basis
a = 100 # picking the strength of mag field
E,landau_levels = energy(B[a])

# %% Finding the lowest and highest energy eigenfunctions
print(f"The index of the lowest energy eigenfucntion is {np.argmin(E)}")
print(f"The index of the highest energy eigenfucntion is {np.argmax(E)}")

# %% 
n = 50 # picking the eigenfucntion to display (between 0 and L**2 -1 inclusive)

# %% plotting the amplitude of the eigenfunctions
Amplitude = np.abs(landau_levels[:,n]) # We are selecting all the numbers in the nth column. This is our nth eigenfunction in a flattened form.

# Remapping the eigenfunction to the lattice basis (reversing the linearisation)
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = Amplitude[L*Y + X] 

fig, ax = plt.subplots()

im_amp = ax.imshow(Z, cmap='plasma', vmin = 0, origin="lower")
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im_amp)
plt.show()

# %% plotting the phase of the eigenfunctions
Phase = np.angle(landau_levels[:,n])

X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Zp = Phase[L*Y + X]

fig, ax = plt.subplots()

im_phase = ax.imshow(Zp, cmap='twilight_shifted', vmin = -3.14, vmax = 3.14, origin="lower") # This cmap has a cyclic colour scheme
ax.set_title("phase of eigenfunction on the lattice")
fig.colorbar(im_phase)
plt.show()


# %% BANDSTRUCTURE!!! (added in 2026)
eigenvalues,eigenvectors = energy(B[100]) # calling it again so that we can choose the flux ratio
sorted_eigenvalues = np.sort(eigenvalues)

# making a correspondence between the sorted energy eigenvalues and the eigenfunctions in the order they are produced in
energy_dict = []
for i in sorted_eigenvalues:
    energy_dict.append(list(eigenvalues).index(i))
energy_dict = np.array(energy_dict)

# These are just for reference, they are not used in the code
point_X = np.array([1,0],np.float64) # Point on the BZ boundary directly along the k_x direction in the reciprocal lattice 
point_Gamma = np.array([0,0],np.float64) # Origin in the reciprocal lattice
point_M = np.array([1,1],np.float64) # Corner of the square BZ

g = np.meshgrid(np.linspace(0,L-1,L,dtype=np.float64), np.linspace(0,L-1,L,dtype=np.float64))
r = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1) #this is the lattice

g_k = np.meshgrid(np.linspace(-1,1,L,endpoint=False,dtype = np.float64), np.linspace(-1,1,L,endpoint=False,dtype = np.float64))
k_space = np.pi*np.append(g_k[0].reshape(-1,1),g_k[1].reshape(-1,1),axis=1) #this is the reciprocal lattice, kinda

# Precalculating the Fourier exponential factors as they are the same for all wavefunctions
# This gives a 10x performance increase
@jit(nopython=True)
def factors():
    factors = np.empty(shape=(L**4,), dtype=np.complex128)
    for k in index:
        for p in index:
            factors[k*(L**2) + p] = np.exp(-1j * np.dot(k_space[k], r[p]))

    return factors

Factors = factors()

@jit(nopython=True)
def Fourier_Transform(m):
    eigfunc = eigenvectors[:,m] # Loading the chosen wavefunction
    i = 0
    phi = np.empty(shape=(L**2,), dtype=np.int64) # initialising the fourier transformed wavefunction
    for k in index:
        phi_k = 0
        for p in index:
            phi_k += Factors[k*(L**2) + p] * eigfunc[p] # I am saving the absolute value squared of each term of the fourier transform. This is because we need how big each "term" is.
        phi[i] = np.square(np.abs(phi_k))
        i += 1

    return np.reshape(phi, (L,L))

# Just a primitive test to see how quickly each iteration of the fourier transform takes
time_start = time.time()
FT_amp = Fourier_Transform(energy_dict[0])
time_end = time.time()
print(f"Took {time_end - time_start} seconds")

time_start = time.time()
FT_amp = Fourier_Transform(energy_dict[1])
time_end = time.time()
print(f"Took {time_end - time_start} seconds for the second time")

time_start = time.time()
FT_amp = Fourier_Transform(energy_dict[2])
time_end = time.time()
print(f"Took {time_end - time_start} seconds for the third time")

Origin = int(L/2) # The location of the origin of the reciprocal lattice is at this index for both x and y

@jit(nopython = True)
def bandstructure():
    band = np.empty(shape = (3*L**2, int(L/2)), dtype=np.int64) # initialising the bandstructure
    for n in index:
        Fourier = Fourier_Transform(energy_dict[n]) # Written to limit the number of calls to Fourier_Transform
        band[3*n] = Fourier[Origin: ,  Origin] # Gamma to X 
        band[3*n +1] = Fourier[L-1,  Origin:] # X to M
        band[3*n + 2] = np.array([Fourier[x,x] for x in np.arange(L-1,Origin-1,-1)]) # M to Gamma
    return band

bds = bandstructure()

# %% plotting
a = int(3*L/2) # just a convenient constant
X, Y = np.meshgrid(np.linspace(0, a-1, a,dtype=int), sorted_eigenvalues)
# Y has to be the sorted energy values otherwise, it plots everything "linearly". 
# The energy values for each wavefunction are not uniform and that is important.
Bands = np.reshape(bds,(L**2, a)) # needs changing to "create" the bands properly


# plot
fig, ax = plt.subplots()
plt.rcParams['figure.dpi'] = 600 # change these based on the lattice resolution
plt.rcParams['savefig.dpi'] = 600

plt.ylabel("Energy")
plt.xlabel("k")
plt.vlines([Origin-1, L-1], sorted_eigenvalues[0], sorted_eigenvalues[L**2-1],"k")
plt.xticks([0,Origin-1,L-1,a-1], ['\N{GREEK CAPITAL LETTER GAMMA}','X','M','\N{GREEK CAPITAL LETTER GAMMA}'])
im = ax.pcolormesh(X, Y, Bands, vmax = 10, cmap='Greys')
# Limiting the vmax allows us to see the details better. Can log Bands instead to be a bit more rigorous
ax.set_title("Band structure")
fig.colorbar(im)
plt.show()

# %%
