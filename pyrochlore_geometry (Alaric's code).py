import numpy as np

Lx = 8
Ly = 8
Lz = 8

pyro = np.array([
    [ 1, 1, 1],
    [ 1,-1,-1],
    [-1, 1,-1],
    [-1,-1, 1]
])

# r = [pyro[0]*0.125, pyro[1]*0.125,
#                               pyro[2]*0.125, pyro[3]*0.125]

diamond = np.array([[0,0,0], [2,2,2]],dtype=np.int32)

fcc_Dy = np.array([
    [0,0,0],
    [0,4,4],
    [4,0,4],
    [4,4,0]
],dtype=np.int32)

fcc_Ti = np.array([
    [4,4,4],
    [4,0,0],
    [0,4,0],
    [0,0,4]
],dtype=np.int32)


plaqt = np.array([
    [
        [ 0,-2, 2],
        [ 2,-2, 0],
        [ 2, 0,-2],
        [ 0, 2,-2],
        [-2, 2, 0],
        [-2, 0, 2]],
    [
        [ 0, 2,-2],
        [ 2, 2, 0],
        [ 2, 0, 2],
        [ 0,-2, 2],
        [-2,-2, 0],
        [-2, 0,-2]
    ],
    [
        [ 0,-2,-2],
        [-2,-2, 0],
        [-2, 0, 2],
        [ 0, 2, 2],
        [ 2, 2, 0],
        [ 2, 0,-2]],
    [
        [ 0, 2, 2],
        [-2, 2, 0],
        [-2, 0,-2],
        [ 0,-2,-2],
        [ 2,-2, 0],
        [ 2, 0, 2]
    ]
],dtype=np.int32)


# takes pos[nplaqs, 3]
def get_sublattice(pos:np.ndarray):
    nplaqs = pos.shape[0]
    sl = -1*np.ones(nplaqs,dtype=np.int32)
    for i in range(4):
        pd = np.outer(np.ones(nplaqs), pyro[i])
        sl[ np.all((pos - pd)%4==0,axis=1) ] = i
    assert np.all(sl >= 0)
    return sl

S2 = np.sqrt(2)
S3 = np.sqrt(3)
S6 = np.sqrt(6)

# Local axes [sl][1,2,3]
axis = np.array([
    [np.array([ 1, 1,-2])/S6, np.array([-1, 1, 0])/S2, np.array([ 1, 1, 1])/S3],
    [np.array([ 1,-1, 2])/S6, np.array([-1,-1, 0])/S2, np.array([ 1,-1,-1])/S3],
    [np.array([-1, 1, 2])/S6, np.array([ 1, 1, 0])/S2, np.array([-1, 1,-1])/S3],
    [np.array([-1,-1,-2])/S6, np.array([ 1,-1, 0])/S2, np.array([-1,-1, 1])/S3]
]);





# Path of dynamic_correlator::path4 as an array (units: 2pi/a)
p4_times = np.array([    0,    4,    6,    7,   10,   12,   13,   14])
p4_kpoints = np.array([
    [  0.0,  1.0,  1.0, 0.75,  0.0,  0.5, 1.00,  1.0],
    [  0.0,  0.0,  0.5, 0.75,  0.0,  0.5, 0.25,  0.0],
    [  0.0,  0.0,  0.0, 0.00,  0.0,  0.5, 0.25,  0.0]]).T


high_symmetry_points = {
    '\Gamma': [0.,0.,0.],
    'X': [1.,0.,0.],
    'W': [1.,0.5,0.],
    'K': [0.75,0.75,0.],
    'L': [0.5,0.5,0.5],
    'U': [1.0, 0.25,0.25]
}

BZ_paths = {
    'path4': '\Gamma X W K \Gamma L U X \Gamma'.split(' ')
    }


def parameterised_path(pathspec):
    coords = [high_symmetry_points[s] for s in pathspec]
    
    x0y0 = coords[0]
    
    # stores the times at which special points appear
    t_list = [0]
    uvec_list = []
    
    for i in range(1,len(pathspec)):
        xy = coords[i]
        arclen = np.linalg.norm(xy-x0y0)
        t_list.append(t_list[i-1]+arclen)
        uvec_list.append((xy-x0y0)/arclen)
        x0y0 = xy
    
    t_list = np.array(t_list)
    uvec_list = np.array(uvec_list)
    
    def p(t):
        retval = []
        for tau in t:
            idx = t_list.searchsorted(tau)
            if idx==0:
                retval.append(coords[0])
            elif idx < len(pathspec):
                retval.append(coords[idx-1] + (tau-t_list[idx-1])*uvec_list[idx-1])
        return np.array(retval)
            
    return p, t_list