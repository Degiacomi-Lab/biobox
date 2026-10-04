# Copyright (c) 2014-2026 Matteo Degiacomi
#
# BiobOx is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# BiobOx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with BiobOx ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author : Matteo Degiacomi, matteo.degiacomi@gmail.com


# CYTHON CALLS FOR HEAVY-DUTY METHODS IN PATH CLASS

import numpy as np
cimport numpy as np
from cpython cimport bool
import scipy.spatial.distance as S


### HEURISTIC FOR SHORTEST PATH ###
cdef cython_min_manhattan(np.ndarray a, np.ndarray b):
    cdef float l
    cdef float themin = 1000000

    for a1 in a: 
        l=abs(a1[0] - b[0]) + abs(a1[1] - b[1]) + abs(a1[2] - b[2])
        if l<themin:
            themin=l
            
    return themin

### BRESENHAM ALGORITHM ###

cdef cython_line_of_sight(np.ndarray access_grid, np.ndarray a, np.ndarray b):
    
    cdef np.ndarray point
    cdef np.ndarray delta
    cdef int x_inc
    cdef int y_inc
    cdef int z_inc
    cdef int l
    cdef int m
    cdef int n
    cdef int dx2
    cdef int dy2
    cdef int dz2
    cdef int err1
    cdef int err2
    
    point=a.copy()
    delta=b-a 
    
    if delta[0]<0:
        x_inc=-1
    else:
        x_inc=1  
    
    if delta[1]<0:
        y_inc=-1
    else:
        y_inc=1
    
    if delta[2]<0:
        z_inc=-1
    else:
        z_inc=1
    
    l = abs(delta[0])
    m = abs(delta[1])
    n = abs(delta[2])
    dx2 = (l << 1)
    dy2 = (m << 1)
    dz2 = (n << 1)
    
    if l >= m and l >= n:
        err_1 = dy2 - l
        err_2 = dz2 - l
        for i in range(0, l, 1):
    
            if err_1 > 0:
                point[1] += y_inc
                err_1 -= dx2
    
            if err_2 > 0:
                point[2] += z_inc
                err_2 -= dx2
     
            err_1 += dy2
            err_2 += dz2
            point[0] += x_inc
    
            if not access_grid[point[0],point[1],point[2]]:
                return False
    
    
    elif m >= l and m >= n:
        err_1 = dx2 - m
        err_2 = dz2 - m
        for i in range(0, m, 1):
    
            if err_1 > 0:
                point[0] += x_inc
                err_1 -= dy2
    
            if err_2 > 0:
                point[2] += z_inc
                err_2 -= dy2
    
            err_1 += dx2
            err_2 += dz2
            point[1] += y_inc
    
            if not access_grid[point[0],point[1],point[2]]:
                return False       
    
    else:
        err_1 = dy2 - n
        err_2 = dx2 - n
        for i in range(0, n, 1):
    
            if err_1 > 0:
                point[1] += y_inc
                err_1 -= dz2
    
            if err_2 > 0:
                point[0] += x_inc
                err_2 -= dz2
    
            err_1 += dy2
            err_2 += dx2
            point[2] += z_inc
    
            if not access_grid[point[0],point[1],point[2]]:
                return False
    
    return True


### SASA ###
cpdef cython_get_surface(np.ndarray points, np.ndarray radii, float probe, int n_sphere_point, float threshold):
    '''
    compute the accessible surface area of a set of atoms with the Shrake-Rupley algorithm.

    Every atom is surrounded by n_sphere_point mesh points at distance radius+probe from its centre, placed
    along a golden spiral. A mesh point is exposed when it lies farther than radius+probe from every other atom.
    Unlike :func:`sasa <biobox.measures.calculators.sasa>`, only atoms whose number of exposed points exceeds
    n_sphere_point*threshold contribute to the area.

    :param points: nx3 numpy array of atomic coordinates, in A
    :param radii: numpy array of n atomic radii, in A
    :param probe: radius of the "rolling ball", in A
    :param n_sphere_point: number of mesh points per atom
    :param threshold: fraction of mesh points that must be exposed for an atom to count as a surface atom
    :returns: accessible surface area in A^2, summed over surface atoms only
    :returns: mx3 numpy array of the exposed mesh points of all atoms
    :returns: numpy array of int, indices of the surface atoms
    '''

    cdef np.ndarray mesh
    cdef np.ndarray dist
    cdef np.ndarray test
    cdef int cnt
    cdef float asa=0.0
    cdef float const=4.0*np.pi/n_sphere_point
    cdef np.ndarray sphere_points
    cdef bool fail
    cdef float thethreshold=n_sphere_point*threshold
    
    cdef float inc
    cdef float offset
    cdef float y
    cdef float r
    cdef float phi
    cdef int k

    #create unit sphere points cloud (using golden spiral)
    pts = []
    inc = np.pi*(3-np.sqrt(5))
    offset =2/float(n_sphere_point)
    for k in range(int(n_sphere_point)):
            y=k*offset-1+(offset/2)
            r=np.sqrt(1 - y*y)
            phi=k*inc
            pts.append([np.cos(phi)*r, y, np.sin(phi)*r])

    sphere_points=np.array(pts)

    contact_map=S.cdist(points,points)

    asa=0.0
    surface_atoms=[]
    mesh_pts=[]
    #compute accessible surface for every atom
    for i in range(0,len(points),1):

            #place mesh points around atom of choice
            mesh=sphere_points*(radii[i]+probe)+points[i]

            #compute distance matrix between mesh points and neighboring atoms. Atom j can cover
            #a mesh point of atom i only within radii[i]+radii[j]+2*probe, and atom i itself is
            #excluded, since rounding would otherwise bury about half of its own mesh
            test=np.where(contact_map[i,:]<radii[i]+radii+probe*2)[0]
            test=test[test!=i]
            neigh=points[test]
            dist=S.cdist(neigh,mesh)-radii[test][:,np.newaxis]

            #lines=atoms, columns=mesh points. Count columns containing values greater than probe*2
            #i.e. allowing sufficient space for a probe to fit completely
            cnt=0
            for m in range(dist.shape[1]):
                    if not np.any(dist[:,m]<probe):
                            cnt+=1
                            mesh_pts.append(mesh[m])
            
            #calculate asa for current atom, if a sufficient amount of mesh points is exposed (NOTE: to verify)
            if cnt>thethreshold:
                    surface_atoms.append(i)
                    asa+=const*cnt*(radii[i]+probe)**2

    return asa, np.array(mesh_pts), np.array(surface_atoms)

 

### RAVEL INDEX ####
cdef cython_ravel(np.ndarray thepos, np.ndarray theshape):
    return theshape[1]*theshape[2]*thepos[0]+theshape[2]*thepos[1]+thepos[2]

#########################################################################################

### EXPORTS ###
#a: all possible targets, b: starting points
def c_heuristic(np.ndarray a, np.ndarray b):
    '''
    minimal Manhattan distance between a point and a set of points, used as heuristic for shortest path search.

    :param a: nx3 numpy array of candidate points (e.g. all possible targets)
    :param b: numpy array with 3 elements, the starting point
    :returns: smallest Manhattan distance between b and any row of a (float, single precision). 1000000 if a is empty
    '''
    return cython_min_manhattan(a,b)

def c_line_of_sight(np.ndarray access_grid, np.ndarray a, np.ndarray b):
    '''
    check whether a straight line between two grid cells crosses only accessible cells (3D Bresenham algorithm).

    :param access_grid: 3D boolean numpy array, True where a cell is accessible
    :param a: numpy array of 3 integer indices, starting cell
    :param b: numpy array of 3 integer indices, end cell
    :returns: True if every cell traversed from a to b (a excluded, b included) is accessible, False otherwise
    '''
    return cython_line_of_sight(access_grid,a,b)

def c_get_surface(np.ndarray points, np.ndarray radii, float probe, int n_sphere_point, float threshold):
    '''
    compute the accessible surface area of a set of atoms with the Shrake-Rupley algorithm. Wrapper of cython_get_surface.

    :param points: nx3 numpy array of atomic coordinates, in A
    :param radii: numpy array of n atomic radii, in A
    :param probe: radius of the "rolling ball", in A
    :param n_sphere_point: number of mesh points per atom
    :param threshold: fraction of mesh points that must be exposed for an atom to count as a surface atom
    :returns: accessible surface area in A^2, summed over surface atoms only
    :returns: mx3 numpy array of the exposed mesh points of all atoms
    :returns: numpy array of int, indices of the surface atoms
    '''
    return cython_get_surface(points, radii, probe, n_sphere_point, threshold)

def c_ravel(np.ndarray thepos, np.ndarray theshape):
    '''
    convert a 3D index into the flat index of a C-ordered array.

    :param thepos: numpy array of 3 integer indices
    :param theshape: numpy array with the 3 dimensions of the array
    :returns: flat index of thepos (integer)
    '''
    return cython_ravel(thepos, theshape)
