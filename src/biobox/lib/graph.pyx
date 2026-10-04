# Copyright (c) 2014-2026 Matteo Degiacomi
#
# biobox is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# biobox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with biobox ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author : Matteo Degiacomi, matteo.degiacomi@gmail.com


import numpy as np
cimport numpy as np
from cpython cimport bool
from libc.math cimport sqrt
import scipy.spatial.distance as S
import scipy.signal
from scipy.spatial import Delaunay


cdef class Graph(object):
        '''
        Accessibility grid built around a cloud of obstacle points, and the graph operations used for path search on it.
        Grid points are addressed by 3D indices or by flat indices (row-major order over the grid shape).
        '''

        cdef np.ndarray xax
        cdef np.ndarray yax
        cdef np.ndarray zax
        cdef public np.ndarray center #grid center
        cdef public float step
        cdef public np.ndarray prot_points
        cdef public np.ndarray params
        cdef np.ndarray g
        cdef np.ndarray w
        cdef np.ndarray points
        cdef public np.ndarray access_grid
        cdef public np.ndarray access_grid_shape

        ## Load protein and make an accessibility grid out of it.
        #@param prot_points atoms to consider for clash detection
        def __init__(self, prot_points):
            '''
            :param prot_points: obstacle points, as an (n, 3) numpy array
            '''
            self.prot_points=prot_points


        #@param step grid step size
        #@param maxdist maximal grid size. If equal to -1, a grid around the whole points ensemble is built
        #@param boundaries build a grid within the desired box boundaries (if defined, maxdist parameter is ignored)
        cpdef make_grid(self, float step=1.0, float maxdist=-1, np.ndarray boundaries=np.array([]), int degree=5, int sigma=2, np.ndarray params=np.array([])):
            '''
            define the grid axes and center, and the Gaussian kernel used for density maps. The accessibility map is not computed here.

            :param step: grid step size, in A
            :param maxdist: edge length of a cubic grid centred on the origin, in A. If equal to -1, a grid around the whole obstacle points ensemble (plus one step on every side) is built
            :param boundaries: build a grid within the desired box boundaries, given as a 3x2 array [[xmin, xmax], [ymin, ymax], [zmin, zmax]] (if defined, maxdist parameter is ignored)
            :param degree: half size of the Gaussian kernel, in grid points
            :param sigma: standard deviation of the Gaussian kernel, in grid points
            :param params: per obstacle point parameters ([type, sigma, amplitude] rows), stored for the computation of the accessibility map. If empty, the single Gaussian kernel is used for all points
            '''

            self.g=self._make_3d_gaussian(degree, sigma)

            self.step=step
            self.params=params

            #define box according to desired boundaries
            if len(boundaries)==3:
                if np.any(boundaries[:,0]>=boundaries[:,1]):
                    raise Exception("upper grid boundary is greater than a lower boundary!")

                self.xax=np.arange(boundaries[0][0],boundaries[0][1]+step,step)
                self.yax=np.arange(boundaries[1][0],boundaries[1][1]+step,step)
                self.zax=np.arange(boundaries[2][0],boundaries[2][1]+step,step)
                #grid=np.array(np.meshgrid(s1,s2,s3))
                #self.center=np.mean(self.prot_points, axis=0)
                self.center=np.array([(boundaries[0][0]+boundaries[0][1])/2.0,\
                                      (boundaries[1][0]+boundaries[1][1])/2.0,\
                                      (boundaries[2][0]+boundaries[2][1])/2.0])

            #define box vertices to make a generic cube
            elif maxdist != -1:
                self.xax=np.arange(-maxdist/2.0,maxdist/2.0+step,step)
                self.yax=np.arange(-maxdist/2.0,maxdist/2.0+step,step)
                self.zax=np.arange(-maxdist/2.0,maxdist/2.0+step,step)
                #grid=np.array(np.meshgrid(s,s,s))         
                self.center=np.array([0.0,0.0,0.0])
                #self.center=np.mean(self.prot_points, axis=0)


            #define box vertices and create grid around points ensemble
            else:
                s=(np.max(self.prot_points, axis=0)-np.min(self.prot_points, axis=0))/2.0+step
                self.xax=np.arange(-s[0],s[0]+step,step)
                self.yax=np.arange(-s[1],s[1]+step,step)
                self.zax=np.arange(-s[2],s[2]+step,step)
                #grid=np.array(np.meshgrid(s1,s2,s3))
                self.center=(np.max(self.prot_points, axis=0)+np.min(self.prot_points, axis=0))/2.0
         
                 
        #@param step grid step size
        #@param use_hull if True, points not laying with the points convex hull will be excluded
        #@param boundaries build a grid within the desired box boundaries (if defined, maxdist parameter is ignored)
        #@param cloud build a grid using a points cloud as extrema for the construction of the box. If defined, maxdist and boundaries parameters are ignored.
        cpdef make_global_grid(self, float step=1.0, bool use_hull=False, np.ndarray boundaries=np.array([]), np.ndarray cloud=np.array([]), params=np.array([])):
            '''
            build a grid and compute its accessibility map. Obstacle points are binned in the grid and convolved with a Gaussian kernel. Without params, grid points where the density is below its maximum minus 3 standard deviations are accessible. With params, each point type has its own kernel (sigma scaled by 1.5) and amplitude, and grid points where the summed density is below 1 are accessible.

            :param step: grid step size, in A
            :param use_hull: if True, grid points not laying within the convex hull of the obstacle points (or of cloud, if provided), scaled about their centroid by a swelling factor, are made inaccessible
            :param boundaries: build a grid within the desired box boundaries, given as a 3x2 array [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If neither boundaries nor cloud is defined, the grid wraps all obstacle points
            :param cloud: build a grid using a points cloud as extrema for the construction of the box (extended by one step on every side). If defined, the boundaries parameter is ignored.
            :param params: per obstacle point parameters ([type, sigma, amplitude] rows). If empty, a single Gaussian kernel is used for all points
            '''

            #if cloud is provided, use that as reference for grid building
            if len(cloud)>0:
                smax=np.max(cloud, axis=0)+step
                smin=np.min(cloud, axis=0)-step
                s=np.array([smin,smax]).T
                self.make_grid(step=step, boundaries=s, params=params)

            else:
                self.make_grid(step=step, boundaries=boundaries, params=params)

            if len(self.params)==0:                
    
                #prepare mesh grid, and place, Kronecker deltas
                grid=np.zeros((len(self.xax),len(self.yax),len(self.zax)))
                for p in self.prot_points:
                    xpos=np.argmin(np.abs(self.xax+self.center[0]-p[0]))
                    ypos=np.argmin(np.abs(self.yax+self.center[1]-p[1]))
                    zpos=np.argmin(np.abs(self.zax+self.center[2]-p[2]))
                    grid[xpos,ypos,zpos]=1
    
                b=scipy.signal.fftconvolve(grid, self.g, mode='same')
    
                #accept points where density is under threshold (i.e., region is accessible)
                #without obstacles in the grid, every point is accessible
                if np.max(b)>0:
                    self.access_grid=b<np.max(b)-np.std(b)*3
                else:
                    self.access_grid=np.ones(b.shape, dtype=bool)

            else:
                b=[]
                for a in np.unique(self.params[:,0]):      
                    
                    grid=np.zeros((len(self.xax),len(self.yax),len(self.zax)))

                    test=np.where(self.params[:,0]==a)[0]
                    points=self.prot_points[test]

                    sigma=self.params[test,1][0]
                    ampl=self.params[test,2][0]
                    self.g=self._make_3d_gaussian(5, sigma*1.5)


                    #prepare mesh grid, and place, Kronecker deltas
                    grid=np.zeros((len(self.xax),len(self.yax),len(self.zax)))
                    for p in points:#self.prot_points:
                        xpos=np.argmin(np.abs(self.xax+self.center[0]-p[0]))
                        ypos=np.argmin(np.abs(self.yax+self.center[1]-p[1]))
                        zpos=np.argmin(np.abs(self.zax+self.center[2]-p[2]))
                        grid[xpos,ypos,zpos]=1
            
                    if len(b)==0:                        
                        b=scipy.signal.fftconvolve(grid, self.g, mode='same')
                        b/=np.max(b)
                        b/=ampl
                    else:
                        b_tmp=scipy.signal.fftconvolve(grid, self.g, mode='same')
                        b_tmp/=np.max(b_tmp)
                        b_tmp/=ampl
                        b+=b_tmp
    
                #accept points where density is under threshold (i.e., region is accessible)
                self.access_grid=b<1.0
                
                    
            self.access_grid_shape=np.array(b.shape)

            # if true, accept only access grid points inside of the point cloud convex hull
            if use_hull:
                
                #scaling factor artificially "swelling the protein" about its centroid, to allow the convex hull to wrap around the points cloud
                minbox=np.min(np.array(self.access_grid_shape)*self.step).astype(float)
                scaling=(minbox+self.step*10)/minbox

                #compute Delaunay triangulation
                if len(cloud)==0:
                    hull_points=self.prot_points
                else:
                    hull_points=cloud
                centroid=np.mean(hull_points, axis=0)
                hull=Delaunay((hull_points-centroid)*scaling+centroid)

                #extract accessible gridpoints coordinates
                w=np.array(np.where(self.access_grid)).T.astype(float)
                gridpoints=self.get_points_from_idx(w)

                #disable gridpoints laying outside the convex hull
                hullscore=hull.find_simplex(gridpoints)
                bad_idx=w[hullscore==-1].astype(int)
                self.access_grid[tuple(bad_idx.T)]=False
     
            #self.w=np.array(np.where(self.access_grid)).T     
            #self.points=self.get_points_from_idx(self.w) #accessible points!
                
        #@param start coordinates of the first point to link
        #@param end coordinates of the second point to link
        #@param stds number of standard deviations for electron density boundaries definition
        cpdef place_local_grid(self, np.ndarray start, np.ndarray end, float stds=3.0):
                '''
                center the grid on the midpoint between two points, and compute its accessibility map from the obstacle points inside the grid. Nothing is done if the grid is already centred there. With params, each point type has its own kernel (sigma scaled by 2) and amplitude, and grid points where the summed density is below 1 are accessible.

                :param start: coordinates of the first point to link
                :param end: coordinates of the second point to link
                :param stds: without params, grid points where the density is below its maximum minus stds standard deviations are accessible
                '''

                cdef np.ndarray grid
                cdef int xpos
                cdef int ypos
                cdef int zpos
                cdef np.ndarray avgpos=(start+end)/2.0

                #if the same position as current position is requested, skip process
                if not np.any(avgpos!=self.center):
                    return

                #self.points+=(avgpos-self.center)
                self.center=avgpos

                if len(self.params)==0:                
                    #prepare mesh grid, and place, Kronecker deltas
                    grid=np.zeros((len(self.xax),len(self.yax),len(self.zax)))
    
                    #if protein point in region of interest, place a Kroneker delta
                    for p in self.prot_points:
                        if p[0]>self.xax[0]+self.center[0] and p[0]<-self.xax[0]+self.center[0]:
                            if p[1]>self.yax[0]+self.center[1] and p[1]<-self.yax[0]+self.center[1]:
                                if p[2]>self.zax[0]+self.center[2] and p[2]<-self.zax[0]+self.center[2]:
                                    xpos=np.argmin(np.abs(self.xax+self.center[0]-p[0]))
                                    ypos=np.argmin(np.abs(self.yax+self.center[1]-p[1]))
                                    zpos=np.argmin(np.abs(self.zax+self.center[2]-p[2]))              
                                    grid[xpos,ypos,zpos]=1
                                                
                    b=scipy.signal.fftconvolve(grid, self.g, mode='same')
    
                    #accept points where density is under threshold (i.e., region is accessible)
                    #without obstacles in the grid, every point is accessible
                    if np.max(b)>0:
                        self.access_grid=b<np.max(b)-np.std(b)*stds
                    else:
                        self.access_grid=np.ones(b.shape, dtype=bool)
                    
                else:
                    b=[]
                    for a in np.unique(self.params[:,0]):      
                        
                        grid=np.zeros((len(self.xax),len(self.yax),len(self.zax)))

                        test=np.where(self.params[:,0]==a)[0]
                        points=self.prot_points[test]

                        sigma=self.params[test,1][0]
                        ampl=self.params[test,2][0]
                        self.g=self._make_3d_gaussian(5, sigma*2.0)
        
                        #if protein point in region of interest, place a Kroneker delta
                        for p in points: #self.prot_points:
                            if p[0]>self.xax[0]+self.center[0] and p[0]<-self.xax[0]+self.center[0]:
                                if p[1]>self.yax[0]+self.center[1] and p[1]<-self.yax[0]+self.center[1]:
                                    if p[2]>self.zax[0]+self.center[2] and p[2]<-self.zax[0]+self.center[2]:
                                        xpos=np.argmin(np.abs(self.xax+self.center[0]-p[0]))
                                        ypos=np.argmin(np.abs(self.yax+self.center[1]-p[1]))
                                        zpos=np.argmin(np.abs(self.zax+self.center[2]-p[2]))              
                                        grid[xpos,ypos,zpos]=1
                        
                        if len(b)==0:                        
                            b=scipy.signal.fftconvolve(grid, self.g, mode='same')
                            b/=np.max(b)
                            b/=ampl
                        else:
                            b_tmp=scipy.signal.fftconvolve(grid, self.g, mode='same')
                            b_tmp/=np.max(b_tmp)
                            b_tmp/=ampl
                            b+=b_tmp
        
                    #accept points where density is under threshold (i.e., region is accessible)
                    self.access_grid=b<1.0
                    
                self.access_grid_shape=np.array(b.shape)
                #self.w=np.array(np.where(self.access_grid)).T                
                #self.points=self.get_points_from_idx(self.w)


        ## convert accessibility map coordinates into a position
        # @param accessibility grid index, can be either flat or 3D
        # @param flat_index if true, the index will be first converted into 3D
        cpdef np.ndarray get_points_from_idx_flat(self, int idx2):
                '''
                convert a flat grid index into a position.

                :param idx2: flat index of a grid point
                :returns: coordinates of the grid point, as a numpy array of 3 floats
                '''
                cdef np.ndarray idx=np.array(self.get_3d_index(idx2))
                return idx*self.step+self.get_origin()


        ## convert accessibility map coordinates into a position
        # @param accessibility grid index, can be either flat or 3D
        # @param flat_index if true, the index will be first converted into 3D
        cpdef np.ndarray get_points_from_idx(self, np.ndarray idx):
                '''
                convert 3D grid indices into positions.

                :param idx: 3D index of a grid point, or an (n, 3) numpy array of 3D indices
                :returns: coordinates of the grid points, with the same shape as idx
                '''
                return idx*self.step+self.get_origin()


        ## coordinates of the grid point with index [0, 0, 0]
        cpdef np.ndarray get_origin(self):
                '''
                :returns: coordinates of the grid point with index [0, 0, 0], as a numpy array of 3 floats
                '''
                return self.center+np.array([self.xax[0], self.yax[0], self.zax[0]])


        ## indices of the grid points closest to a list of positions
        #@param pts positions, as an (n, 3) array
        cpdef np.ndarray get_idx_from_points(self, np.ndarray pts):
                '''
                indices of the grid points closest to a list of positions. Positions outside the grid give indices out of the grid bounds.

                :param pts: positions, as an (n, 3) array
                :returns: 3D indices, as an (n, 3) numpy array of integers
                '''
                return np.rint((pts-self.get_origin())/self.step).astype(int)


        ## test whether the grid points closest to a list of positions are accessible (positions outside the grid are not)
        #@param pts positions, as an (n, 3) array
        cpdef np.ndarray is_accessible(self, np.ndarray pts):
                '''
                test whether the grid points closest to a list of positions are accessible (positions outside the grid are not).

                :param pts: positions, as an (n, 3) array
                :returns: numpy array of n booleans
                '''
                cdef np.ndarray idx=self.get_idx_from_points(pts)
                cdef np.ndarray inside=np.all((idx>=0) & (idx<self.access_grid_shape), axis=1)
                cdef np.ndarray result=np.zeros(len(pts), dtype=bool)
                result[inside]=self.access_grid[tuple(idx[inside].T)]
                return result


        ##given a target point, give the closest node in accessibility graph.
        #@param target point in space next to protein (typically an atom coordinate selected with atomselect)
        cpdef get_closest_nodes(self, np.ndarray[double, ndim=2] target):
                '''
                for each target, find the closest accessible grid point among the 5x5x5 grid points around the grid point nearest to it.

                :param target: positions, as an (n, 3) numpy array of floats (typically atom coordinates selected with atomselect)
                :returns: list of the squared distances (in A2) between each target and its closest accessible grid point (10000 if none is found)
                :returns: 3D indices of the closest accessible grid points, as an (n, 3) numpy array (an empty index for a target without accessible grid point)
                '''
   
                cdef int pos
                cdef list idx=[]
                cdef list dists=[]
                #cdef np.ndarray d=S.cdist(target,self.points)
                # 1) find approximate mapping using self.xax, self.yax and self.zax
                # 2) take coordinate having minimum distance between immediate neighbors being true in self.access_grid
                for t in target:
                    i=np.argmin(np.abs(self.xax+self.center[0]-t[0]))
                    j=np.argmin(np.abs(self.yax+self.center[1]-t[1]))
                    k=np.argmin(np.abs(self.zax+self.center[2]-t[2]))
    
                    bestpos=[]
                    bestdist=10000
                    for x in xrange(i-2,i+3,1):
                        for y in xrange(j-2,j+3,1):
                            for z in xrange(k-2,k+3,1):
    
                                if x>=self.access_grid_shape[0] or y>=self.access_grid_shape[1] or z>=self.access_grid_shape[2]:
                                    continue
    
                                if x<0 or y<0 or z<0:
                                    continue
    
                                if not self.access_grid[x,y,z]:
                                    continue
                                
                                pt=self.get_points_from_idx(np.array([x,y,z]))
                                v=pt-t
                                
                                dist=v[0]*v[0]+v[1]*v[1]+v[2]*v[2]
                                if dist<bestdist:
                                    bestdist=dist
                                    bestpos=[x,y,z]
                                    
                    idx.append(bestpos)
                    dists.append(bestdist)
                
                '''
                print idx, dists
                
                #cdef np.ndarray w=np.array(np.where(self.access_grid)).T
                #cdef np.ndarray points=self.get_points_from_idx(w)
                idx=[]
                dists=[]

                for i in xrange(0,len(target),1):
                        pos=np.argmin(d[i])
                        idx.append(self.w[pos])
                        dists.append(np.min(d[i]))
                
                print idx, dists
                '''
                    
                return dists, np.array(idx)


                
        #return indices of neighbors of point p (3d position).
        #@param position of point p
        cpdef np.ndarray neighbors(self, int idx, bool flattened):
                '''
                accessible neighbors of a grid point, among the 26 grid points around it.

                :param idx: flat index of the grid point
                :param flattened: if True, neighbors are returned as flat indices, otherwise as 3D indices
                :returns: numpy array of flat indices, or (m, 3) numpy array of 3D indices
                '''
        
                cdef int x
                cdef int y
                cdef int z
                cdef list n=[]
                cdef list p=self.get_3d_index(idx)

                for x in xrange(p[0]-1,p[0]+2,1):
                    for y in xrange(p[1]-1,p[1]+2,1):
                        for z in xrange(p[2]-1,p[2]+2,1):

                            if x==p[0] and y==p[1] and z==p[2]:
                                continue

                            if x>=self.access_grid_shape[0] or y>=self.access_grid_shape[1] or z>=self.access_grid_shape[2]:
                                continue

                            if x<0 or y<0 or z<0:
                                continue

                            if self.access_grid[x,y,z]:
                                if flattened:
                                    n.append(self.access_grid_shape[1]*self.access_grid_shape[2]*x+self.access_grid_shape[2]*y+z)
                                    #n.append(self.get_flat_index(np.array([x,y,z])))
                                else:
                                    n.append([x,y,z])

                return np.array(n)


        ##return flattened index from 3d one.
        #@param idx 3D coordinate of a point in the graph
        #cpdef get_flat_index(self,idx):
        #        return np.ravel_multi_index(idx,self.access_grid_shape)
        cpdef get_flat_index(self, np.ndarray thepos):
            '''
            convert 3D grid indices into flat indices.

            :param thepos: 3D index of a grid point, or a (3, n) numpy array of 3D indices (one column per grid point)
            :returns: flat index, or numpy array of n flat indices
            '''
            return self.access_grid_shape[1]*self.access_grid_shape[2]*thepos[0]+self.access_grid_shape[2]*thepos[1]+thepos[2]


        ##return 3d index from flattened one
        #@param idx flattened coordinate of a point in the graph
        cpdef list get_3d_index(self, int idx):
            '''
            convert a flat grid index into a 3D index.

            :param idx: flat index of a grid point
            :returns: 3D index, as a list [x, y, z]
            '''

            #unravelling explicitely implemented (faster than calling numpy unravel)
            cdef int p1 = idx%self.access_grid_shape[2]
            cdef int p2 = (idx/self.access_grid_shape[2])%self.access_grid_shape[1]
            cdef int p3 = idx/(self.access_grid_shape[1]*self.access_grid_shape[2]) 
            return [p3,p2,p1]
            #return list(np.unravel_index(idx, tuple(self.access_grid_shape)))

        
        ##euclidean distance between two points in the graph, in grid steps (admissible heuristic for A*)
        cpdef double heuristic(self, a, b):
            '''
            Euclidean distance between two grid points, in grid steps (admissible heuristic for A*).

            :param a: flat index of the first grid point (a numpy integer)
            :param b: flat index of the second grid point (a numpy integer)
            :returns: distance in grid steps
            '''
            cdef list aa=self.get_3d_index(a.T)
            cdef list bb=self.get_3d_index(b.T)
            cdef int dx=bb[0]-aa[0]
            cdef int dy=bb[1]-aa[1]
            cdef int dz=bb[2]-aa[2]
            return sqrt(dx*dx+dy*dy+dz*dz)


        ##cost from moving between two points in the graph: their euclidean distance, in grid steps.
        #@param a point (in flat coordiantes)
        #@param b point (in flat coordiantes)
        cpdef double cost(self, int a, int b):
                '''
                cost of moving between two grid points: their Euclidean distance, in grid steps.

                :param a: flat index of the first grid point
                :param b: flat index of the second grid point
                :returns: distance in grid steps
                '''
                cdef list aa=self.get_3d_index(a)
                cdef list bb=self.get_3d_index(b)
                cdef int dx=bb[0]-aa[0]
                cdef int dy=bb[1]-aa[1]
                cdef int dz=bb[2]-aa[2]
                return sqrt(dx*dx+dy*dy+dz*dz)


        ##generate 3D gaussian, used for density map generation
        # @param half kernel size
        # @param gaussian standard deviation
        # @retval 3d grid containing a binned gaussian density
        cdef _make_3d_gaussian(self, int degree=5, float sigma=0.5):
                '''
                generate a 3D Gaussian kernel, used for density map generation.

                :param degree: half size of the kernel, in grid points
                :param sigma: standard deviation of the Gaussian, in grid points
                :returns: (2*degree+1)x(2*degree+1)x(2*degree+1) numpy array, normalised to sum 1
                '''

                cdef int window=degree*2+1
                shape=(window,window,window)

                m,n,k = [(ss-1.)/2. for ss in shape]

                x_ = np.arange(-m,m+1,1).astype(int)
                y_ = np.arange(-n,n+1,1).astype(int)
                z_ = np.arange(-k,k+1,1).astype(int)
                x, y, z = np.meshgrid(x_, y_, z_)

                h = np.exp( -(x*x + y*y + z*z) / (2.*sigma*sigma) )
                h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
                sumh = h.sum()
                if sumh != 0:
                    h /= sumh

                return h
