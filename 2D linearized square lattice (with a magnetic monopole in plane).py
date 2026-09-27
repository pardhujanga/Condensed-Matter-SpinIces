# %% Defining and solving the Hamiltonian for a 2D lattice with a magnetic monopole in it
from numba import jit
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from matplotlib import pyplot as plt

L=40
index = np.array(range(L**2))
length = np.array(range(L))

a = -1/2 
b = -1/2 #(a,b) is the monopole location

@jit(nopython = True)
def kron(x,y):
    if x == y:
        return 1
    else:
        return 0


# This is much faster at making the Hamiltonian
@jit(nopython = True)
def Hamiltonian():
    M = np.ones((L**2,L**2),dtype=np.complex128) #intializing the hamiltonian 

    for i in index:
        for x in length:
            for y in length:
                if x!=0 and x!=L-1:
                    M[i][L*y + x] *= - kron(i,L*y+(x-1))*np.exp(0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x-1-a)/(y-b)))) - kron(i,L*y+(x+1))*np.exp(0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x+1-a)/(y-b)))) - kron(i,L*((y-1))+x)*np.exp(0.5j*(np.arctan((y-1-b)/(x-a))-np.arctan((y-b)/(x-a)))) - kron(i,L*((y+1))+x)*np.exp(0.5j*(np.arctan((y+1-b)/(x-a))-np.arctan((y-b)/(x-a))))
                elif x==0:
                    M[i][L*y + x] *= - kron(i,L*y+(x+1))*np.exp(0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x+1-a)/(y-b)))) - kron(i,L*((y-1))+x)*np.exp(0.5j*(np.arctan((y-1-b)/(x-a))-np.arctan((y-b)/(x-a)))) - kron(i,L*((y+1))+x)*np.exp(0.5j*(np.arctan((y+1-b)/(x-a))-np.arctan((y-b)/(x-a))))
                elif x==L-1:
                    M[i][L*y + x] *= - kron(i,L*y+(x-1))*np.exp(0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x-1-a)/(y-b)))) - kron(i,L*((y-1))+x)*np.exp(0.5j*(np.arctan((y-1-b)/(x-a))-np.arctan((y-b)/(x-a)))) - kron(i,L*((y+1))+x)*np.exp(0.5j*(np.arctan((y+1-b)/(x-a))-np.arctan((y-b)/(x-a))))
    return M

#NOTE Needed to split the hamiltoninan generation as the way the lattice to avoid interactions between
# endpoints of consecutive "lines of atoms".
H = Hamiltonian()

#NOTE: this is the bottle neck now. Making the Hamiltonian is very quick
#solving for spectrum and eigenfunctions
val, vec = linalg.eig(H)

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
 
# %%
print(np.argmin(val))

# %%
n=10
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
