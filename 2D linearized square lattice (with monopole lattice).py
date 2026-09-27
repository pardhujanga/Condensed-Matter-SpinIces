# %% Defining and solving the Hamiltonian for a 2D lattice with magnetic monopole lattice in plaquettes
from numba import jit
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from matplotlib import pyplot as plt

L=20
index = np.array(range(L**2))
length = np.array(range(L))
points = np.array(range((L-1)**2))
positions = []

for i in np.array(range(L-1)):
    for j in np.array(range(L-1)):
        positions.append((i+0.5,j+0.5))

monopoles = np.array(positions)


@jit(nopython  = True)
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
    M = np.ones((L**2,L**2),dtype=np.complex128) #intializing the hamiltonian 
    for p in points:
        a,b = monopoles[p]
        if (a + b)%2 == 0:
            s = 1
        else:
            s = -1 #modifier to choose the type of monopole - north or south
        #can be made more efficient by choosing the points where there are monopoles earlier and only looping over them
        for i in index:
            for x in length:
                for y in length:
                    if x!=0 and x!=L-1:
                        M[i][L*y + x] *=  kron(i,L*y+(x-1))*np.exp(s*0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x-1-a)/(y-b)))) + kron(i,L*y+(x+1))*np.exp(s*0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x+1-a)/(y-b)))) + kron(i,L*((y-1))+x)*np.exp(s*0.5j*(np.arctan((y-1-b)/(x-a))-np.arctan((y-b)/(x-a)))) + kron(i,L*((y+1))+x)*np.exp(s*0.5j*(np.arctan((y+1-b)/(x-a))-np.arctan((y-b)/(x-a))))
                    elif x==0:
                        M[i][L*y + x] *=  kron(i,L*y+(x+1))*np.exp(s*0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x+1-a)/(y-b)))) + kron(i,L*((y-1))+x)*np.exp(s*0.5j*(np.arctan((y-1-b)/(x-a))-np.arctan((y-b)/(x-a)))) + kron(i,L*((y+1))+x)*np.exp(s*0.5j*(np.arctan((y+1-b)/(x-a))-np.arctan((y-b)/(x-a))))
                    elif x==L-1:
                        M[i][L*y + x] *=  kron(i,L*y+(x-1))*np.exp(s*0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x-1-a)/(y-b)))) + kron(i,L*((y-1))+x)*np.exp(s*0.5j*(np.arctan((y-1-b)/(x-a))-np.arctan((y-b)/(x-a)))) + kron(i,L*((y+1))+x)*np.exp(s*0.5j*(np.arctan((y+1-b)/(x-a))-np.arctan((y-b)/(x-a))))
    return -M

#NOTE Needed to split the hamiltoninan generation as the way the lattice to avoid interactions between
# endpoints of consecutive "lines of atoms".
H = Hamiltonian()

#NOTE: this is the bottle neck now. Making the Hamiltonian is very quick
#solving for spectrum and eigenfunctions
val, vec = linalg.eig(H)

# %% Plotting the energy spectrum
x = np.pi*np.array(range(L))/(L-1)
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

# there is an energy cost to adding monopoles to the lattice
# %%
print(np.argmax(val))

# %%
n=0
# %%  Defining a function to find the expansion of eigenfunctions in the lattice basis
eigf = np.abs(vec[:,n])

# defining the lattice
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = eigf[L*Y + X]

# plot
fig, ax = plt.subplots()

im = ax.imshow(Z, cmap='plasma')
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im)
plt.show()

ipr = np.sum(np.square(np.square(eigf))) #inverse participation ratio - measures localization
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

# %% lattice
# gamma to X
g = np.meshgrid(np.linspace(0,L-1,L,dtype=np.float64), np.linspace(0,L-1,L,dtype=np.float64))
r = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1)


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
eta = 2*np.pi*np.array(range(0,l,1))/(l)

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
    while n < L**2: 
        p=0          
        k = np.array(ft(energy_dict[n]))
        for l in index:
            if l!=n and np.abs(ordered_val[n+p]-ordered_val[l])<5*2*10**-3:
                k = k+np.array(ft(energy_dict[l]))
                p+=1
        n += 1+p
        band.append(list(k))
    return band

bds = bands()

# %% plotting
X, Y = np.meshgrid(np.linspace(0,l-1,l,dtype=int), np.linspace(0,L**2-1,L**2,dtype=int))
#Z = bds[Y][X]

# plot
fig, ax = plt.subplots()
#import matplotlib.colors as colour

im = ax.pcolormesh(np.log(bds+np.ones(np.shape(bds))), cmap='plasma')
ax.set_title("Band structure")
fig.colorbar(im)
plt.show()

# %%
plt.plot(val)
plt.show()
# %%