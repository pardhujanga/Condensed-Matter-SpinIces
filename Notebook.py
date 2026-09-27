# %% Defining and solving the Hamiltonian for a 2D lattice with magnetic monopole lattice in plaquettes
from numba import jit
import numpy as np
import math
import scipy.sparse as sp
from scipy import linalg
from matplotlib import pyplot as plt

L=30
index = np.array(range(L**2))
length = np.array(range(L))
points = np.array(range((L-1)**2))
positions = []

for i in np.array(range(L-1)):
    for j in np.array(range(L-1)):
        positions.append((i+0.5,j+0.5))

monopoles = np.array(positions)

#lattice
g = np.meshgrid(np.linspace(0,L-1,L,dtype=np.float64), np.linspace(0,L-1,L,dtype=np.float64))
r = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1)

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
        s = np.random.choice(np.array([-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))#modifier to choose the type of monopole - north or south
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

# ORDERED BANDS!!!
l = L
xlength = np.array(range(L))
point_X = np.array([1,0],np.float64)
point_Y = np.array([0,1],np.float64)
eta = 2*np.pi*np.array(range(0,l,1))/(l)

@jit(nopython=True)
def ft(vec,m):
    eigfunc = vec[:,m]
    phi = []
    for e in xlength:
        phi_e = 0
        for p in index:
            phi_e += np.exp(-1j*eta[e]*np.dot(point_X,r[p]))*eigfunc[p]
        phi.append(np.square(np.abs(phi_e)))

    return phi

@jit(nopython = True)
def bands(vec):
    band = []
    n=0
    p = 0
    while n < L**2:          
        k = np.array(ft(vec,energy_dict[n]))
        #p = int(np.real((4+ordered_val[n])*100))
        x = 0
        while (m+p*10**-2) <= np.real(ordered_val[n+x]) and np.real(ordered_val[n+x]) <= (m+(p+1)*10**-2):
            k = k+np.array(ft(vec,energy_dict[n+x]))
            x += 1
        p+=1
        n += x 
        band.append(list(k))
    return band

def my_sum(*nested_lists):
    return [[sum(items) for items in zip(*zipped_list)] for zipped_list in zip(*nested_lists)]

# %% 
bds = []
h = Hamiltonian()
values = linalg.eigvals(h)
total_band = np.zeros((int(np.abs(math.ceil(100*np.max(values)))+np.abs(math.floor(100*np.min(values)))),L))

# %%
for i in np.array(range(2)): #because of linalg module, this is not jit-able
    H = Hamiltonian()
    old_bds = bds
    val, vec = linalg.eig(H)
    ordered_val = np.sort_complex(val)
    energy_dict = []
    for i in ordered_val:
        energy_dict.append(list(val).index(i))
    energy_dict = np.array(energy_dict)
    m = np.min(np.real(ordered_val))
    bds = bands(vec)
    if i == 1:
        total_band = my_sum(old_bds,bds) #this addition dosen't do what we want :( Not satisfed with this
    elif i > 1:
        total_band = my_sum(total_band,bds)
    
# %%
print(np.size(bds))
print(np.size(total_band))
# %% plotting
X, Y = np.meshgrid(np.linspace(0,l-1,l,dtype=int), np.linspace(0,L**2-1,L**2,dtype=int))
#Z = bds[Y][X]

# plot
fig, ax = plt.subplots()
#import matplotlib.colors as colour

im = ax.pcolormesh(total_band, cmap='plasma')
#im = ax.pcolormesh(np.log(bds+0.01*np.ones(np.shape(bds))), cmap='plasma')
ax.set_title("Band structure")
fig.colorbar(im)
plt.show()

# %% Plotting the energy spectrum
x = np.pi*np.array(range(L))/(L-1)
cos = np.cos(x)
theory = []
for i in cos:
    for j in cos:
        theory.append(2*i+2*j)

#sorted = np.sort(theory)

ordered_val = np.sort(val)
plt.plot(ordered_val,'k')
#plt.plot(sorted)
plt.show()

# there is an energy cost to adding monopoles to the lattice
# %%
print(np.argmax(val))

# %%
n=400
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
