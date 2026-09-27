# %% Defining and solving the Hamiltonian for a 2D lattice with magnetic monopole lattice in plaquettes
from numba import jit
import numpy as np
import math
from scipy import linalg
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2

L=20
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

filling = np.zeros((L-1)**2, dtype=np.int64)
k = int(np.random.randint(0,(L-1)**2)/2)

# %%
filling[k] += 1
filling[10-k] += 1

@jit(nopython = True) 
def filling_function(j,divisions): #filling is j/divisions
    filling = np.zeros((L-1)**2, dtype=np.int64)
    k = 0
    while k<(j*(L-1)**2/(2*divisions)):
        filling[k] += 1
        filling[(L-1)**2-k-1] += -1
        k+=1
    return filling

#filling = filling_function(10,10)
#np.random.shuffle(filling)

# %%
# Can be made faster by making points only the filled ones
@jit(nopython = True)
def Hamiltonian(alpha):
    M = np.ones((L**2,L**2),dtype=np.complex128) #intializing the hamiltonian 
    for p in points:
        a,b = monopoles[p]
        s = alpha*filling[p] #modifier to choose the type of monopole - north or south
        #can be made more efficient by choosing the points where there are monopoles earlier and only looping over them
        for i in index:
            for x in length:
                for y in length:
                    M[i][L*y + x] *=  kron(i,L*y+(x-1)%L)*np.exp(s*0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x-1-a)/(y-b)))) + kron(i,L*y+(x+1)%L)*np.exp(s*0.5j*(np.arctan((x-a)/(y-b))-np.arctan((x+1-a)/(y-b)))) + kron(i,L*((y-1)%L)+x)*np.exp(s*0.5j*(np.arctan((y-1-b)/(x-a))-np.arctan((y-b)/(x-a)))) + kron(i,L*((y+1)%L)+x)*np.exp(s*0.5j*(np.arctan((y+1-b)/(x-a))-np.arctan((y-b)/(x-a))))

    return -M

H = Hamiltonian(1)
val, vec = linalg.eig(H)

# %% Bott index
@jit(nopython= True)
def phi_x(eigfunc):
    t = eigfunc
    phi = np.zeros((L,L),dtype=np.complex128)
    for i in length:
        for j in length:
            phi += np.exp(2*np.pi*1j*j/L)*t[i][j]
    return phi

@jit(nopython= True)
def phi_y(eigfunc):
    t = eigfunc
    phi = np.zeros((L,L),dtype=np.complex128)
    for i in length:
        for j in length:
            phi += np.exp(2*np.pi*1j*i/L)*t[i][j]
    return phi

# Original formulation
fermi = 1
order = int(fermi*L**2)

#@jit(nopython=True)
def Phi_x():
    sum = 0
    for i in range(order):
        for j in range(order):
            eigf_i = vec[:,i].reshape(L,L)
            eigf_j = vec[:,j].reshape(L,L)
            sum += np.matmul(np.matmul(eigf_i,np.conjugate(np.transpose(eigf_i))),np.matmul(phi_x(eigf_j),np.conjugate(np.transpose(eigf_j))))
    return sum

#@jit(nopython=True)
def Phi_y():
    sum = 0
    for i in range(order):
        for j in range(order):
            eigf_i = vec[:,i].reshape(L,L)
            eigf_j = vec[:,j].reshape(L,L)
            sum += np.matmul(np.matmul(eigf_i,np.conjugate(np.transpose(eigf_i))),np.matmul(phi_y(eigf_j),np.conjugate(np.transpose(eigf_j))))
    return sum

pX = Phi_x()
pY = Phi_y()
Result = np.trace(np.log(np.matmul(np.matmul(pX,pY),np.matmul(np.conjugate(np.transpose(pX)),np.conjugate(np.transpose(pY))))))
Bott_index = np.imag(Result)/(2*np.pi)

print("Bott index is  " + str(Bott_index))

# %% ""Localised Bott""
local = np.imag(np.log(np.matmul(np.matmul(pX,pY),np.matmul(np.conjugate(np.transpose(pX)),np.conjugate(np.transpose(pY))))))/(2*np.pi)

fig, ax = plt.subplots()

im = ax.imshow(local, cmap='plasma')
ax.set_title("localized bott index")
fig.colorbar(im)
plt.show()

# %% Slightly different Bott index formulation
#NOTE: Keeps crashing the kernel
m = int(0.5*L**2) #this is the fermi level index
W = vec[:,0:m] #needs to be modified such that the indices are ordered by energy
Wt = np.transpose(np.conjugate(W))

@jit(nopython=True)
def U(eigfunc):
    u = eigfunc
    phi = np.zeros((L**2,m),dtype=np.complex128)
    for i in index:
        for j in range(m):
            phi += np.exp(2*np.pi*1j*j/L)*u[i][j]
    return phi

@jit(nopython=True)
def V(eigfunc):
    v = eigfunc
    phi = np.zeros((L**2,m),dtype=np.complex128)
    for i in index:
        for j in range(m):
            phi += np.exp(2*np.pi*1j*i/L)*v[i][j]
    return phi

U1 = np.matmul(Wt,U(Wt))
V1 = np.matmul(Wt,V(Wt))

Result = np.trace(np.log(np.matmul(np.matmul(U1,V1),np.matmul(np.conjugate(np.transpose(U1)),np.conjugate(np.transpose(V1))))))
Bott_index = np.imag(Result)/(2*np.pi)

print("Bott index is  " + str(Bott_index))

# %% FFT
i = 12
eigf = vec[:,i].reshape(L,L)
k_eigf = fft2(eigf)

X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
#Z = eigf[L*Y + X]

# plot
fig, ax = plt.subplots()

im = ax.imshow(np.abs(k_eigf), cmap='plasma')
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im)
plt.show()