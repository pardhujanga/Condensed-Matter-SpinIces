# %% Defining and solving the Hamiltonian for a 3D pyrocholore lattice 
#(hopping is actually for spinons on a diamond sub-lattice formed in the pyrochlore)
from numba import jit
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy import linalg
from matplotlib import pyplot as plt

L=12
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
sH = sp.bsr_matrix(H)

# %%
#NOTE: this is the bottle neck now. Making the Hamiltonian is very quick
#solving for spectrum and eigenfunctions
val, vec = linalg.eig(H) #these are the nomralized eigenfucntions

# %%
sval, svec = scipy.sparse.linalg.eigs(sH)

# %%
#linearized coordinates to cartesian coordinates
@jit(nopython = True)
def generate_lattice():
    lattice = np.ones((8*L**3,3))
    for N in index:
        z = int(N/(2*L**2))
        #finding y
        if z%2 == 0:
            y = 2*int((N - z*2*L**2)/L)
        else:
            y = 2*int((N - z*2*L**2)/L) + 1
        #finding x
        if z%4==0:
            if (y/2)%2==0:
                x = 4*((N%(2*L**2))%L)
            else:
                x = 2 + 4*((N%(2*L**2))%L)
        if z%4==1:
            if (y/2)%2==0:
                x = 3 + 4*((N%(2*L**2))%L)
            else:
                x = 1 + 4*((N%(2*L**2))%L)
        if z%4==2:
            if (y/2)%2==0:
                x = 2 + 4*((N%(2*L**2))%L)
            else:
                x = 4*((N%(2*L**2))%L)
        if z%4==3:
            if (y/2)%2==0:
                x = 1 + 4*((N%(2*L**2))%L)
            else:
                x = 3+ 4*((N%(2*L**2))%L)
        

        lattice[N] = np.array([x,y,z])
    
    return lattice

r = generate_lattice()

# %% Plotting the energy spectrum
#NOTE this is the actual distribution of k. So, this is the true spectrum we got analytically.
t0 = np.array([1,1,1]) #these are the tetrahedral corners with the centre at origin (these are also the hopping directions)
t1 = np.array([1,-1,-1])
t2 = np.array([-1,1,-1])
t3 = np.array([-1,-1,1])

tc = np.pi*np.cos(np.pi*np.array(range(8*L))/(8*L-1)) #cos(polar angle - theta)
pc = np.pi*np.cos(2*np.pi*np.array(range(L**2))/(L**2-1)) #cos(azimuthal angle - phi)

k= []
for l in range(8*L):
    for m in range(L**2):
        ts = np.sqrt(np.square(np.pi)-np.square(tc[l]))
        ps = np.sqrt(np.square(np.pi)-np.square(pc[m]))
        k.append([ts*pc[m],ts*ps,tc[l]]) #k is sampled uniformly over all angles. (does create an over representation near the poles)

kx = np.inner(t1,k) # the dot products of k with each hopping vector
ky = np.inner(t2,k)
kz = np.inner(t3,k)

#NOTE the expression for the argument of the cosines in the spectrum are of the form sqrt(3)*k*cos(0 - 109.47 degrees).
# That's why the arguments are made to be sampled over a sphere while taking dot products with the tetrahedral vectors.
# As long as the range of energies is the same, the only thing affecting our coded spectrum is how we sample the argument of the cosines in energy.

theory = []
l=0
for i in range(8*L**3):
        energy = 4 + 2*(np.cos(kx[i])+np.cos(ky[i])+np.cos(kz[i])+np.cos(kx[i]-ky[i])+np.cos(ky[i]-kz[i])+np.cos(kz[i]-kx[i]))
        l += 1
        if l%2 == 1:
            theory.append(np.sqrt(energy))
            theory.append(-np.sqrt(energy))

sorted = np.sort(theory)
ordered_val = np.sort(val)

plt.scatter(index, ordered_val,s=0.01)
#plt.plot(sorted,'k')
plt.show()

# %% Overlap integral of the numerical and analytical spectra
overlap = 0
norm = 0
for i in range(8*L**3):
    overlap += sorted[i]*ordered_val[i]
    norm += ordered_val[i]**2

normalized_overlap = overlap/norm
print(normalized_overlap)
    
# %%
print(np.argmax(val))

# %% Dynamical structure factor (energy bands along symmetry directions)
# gamma to X
# ORDERED BANDS!!!
ordered_val = np.sort_complex(val)
energy_dict = []
for i in ordered_val:
    energy_dict.append(list(val).index(i))
energy_dict = np.array(energy_dict)

l = L
#\Gamma X W K \Gamma L U X \Gamma
point_X = np.array([1,0,0],np.float64)
point_Y = np.array([0,1,0],np.float64)
point_Z = np.array([0,0,1],np.float64)
point_W = np.array([1.,0.5,0.],np.float64)
point_K = np.array([0.75,0.75,0.],np.float64)
point_L = np.array([0.5,0.5,0.5],np.float64)
point_U = np.array([1.0, 0.25,0.25],np.float64)

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
    n=0
    while n < 8*L**3: 
        p=0          
        k = np.array(ft(energy_dict[n]))
        for l in index:
            if l!=n and np.abs(ordered_val[n+p]-ordered_val[l])<6*10**-4:
                k = k+np.array(ft(energy_dict[l]))
                p+=1

        #norm = 0
        #for i in k:
            #norm += i**2

        #k = k*(1/np.sqrt(norm))
        #k = np.log(k)
        k = list(k)

        n += 1+p
        band.append(k)
    return band

bds = bands()

# %% Plotting the band structure
X = np.linspace(0,L-1,L,dtype=int)
Y = np.linspace(0,8*L**3-1,8*L**3,dtype=int)
Ye= np.real(ordered_val)
Yt= np.real(sorted)

# plot
fig, ax = plt.subplots()

im = ax.pcolormesh(bds,cmap='plasma')
ax.set_title("Band structure")
fig.colorbar(im)
plt.show()

#plt.xlabel("reduced wave vector")
#plt.ylabel(" ")
#plt.plot(eta,Phi)
#plt.scatter(eta,Phi,s=1)
#plt.show()

# %% inverse participation ratio - measures localization - keep for 3d
n = 10
eigf = vec[:,n]
ipr = np.sum(np.square(np.square(eigf))) 
print("Inverse participation ratio is " + str(ipr))

# %% the sparse result handler
# ORDERED BANDS!!!
ordered_sval = np.sort_complex(sval)
energy_dict = []
for i in ordered_sval:
    energy_dict.append(list(sval).index(i))
energy_dict = np.array(energy_dict)

l = L
sindex = np.array(range(sval.size))
xlength = np.array(range(L))
point_X = np.array([1,0,0],np.float64)
point_Y = np.array([0,1,0],np.float64)
point_Z = np.array([0,0,1],np.float64)
eta = 2*np.pi*np.array(range(l))/(l)

@jit(nopython=True)
def ft(m):
    eigfunc = svec[:,m]
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
    n=0
    while n < sval.size: 
        p=0          
        k = np.array(ft(energy_dict[n]))
        for l in sindex:
            if l!=n and np.abs(ordered_sval[n]-ordered_sval[l])<10**-3:
                k = k+np.array(ft(energy_dict[l]))
                p+=1
        n += 1+p
        band.append(list(k))
    return band

bds = bands()

# %%
print(ordered_val[0:20])
# %% save array to a CSV file using numpy.savetxt()
np.savetxt('side 12 eigenvectors.csv', vec, delimiter=',')

# load arr from the CSV file using numpy.loadtxt()
vec_loaded = np.loadtxt('side 12 eigenvectors.csv', delimiter=',')

# verify that arr and arr_loaded are the same
print("done")
# %%
