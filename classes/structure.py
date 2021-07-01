# Copyright (c) 2014-2017 Matteo Degiacomi
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
# Author : Matteo Degiacomi, matteothomas.degiacomi@gmail.com

from copy import deepcopy
import numpy as np
import scipy.signal
import pandas as pd

class Structure(object):
    '''
    A Structure consists of an ensemble of points in 3D space, and metadata associated to each of them.
    '''

    def __init__(self, p=np.array([[], []]), r=1.0):
        '''
        Point coordinates and properties data structures are first initialized.
        properties is a dictionary initially containing an entry for 'center' (center of geometry) and 'radius' (average radius of points).

        :param p: coordinates data structure as a mxnx3 numpy array (alternative conformation x atom x 3D coordinate). nx3 numpy array can be supplied, in case a single conformation is present.
        :param r: average radius of every point in dataset (float), or radius of every point (numpy array)
        '''
        if p.ndim == 3:
            self.coordinates = p
            '''numpy array containing an ensemble of alternative coordinates in 3D space'''

        elif p.ndim == 2:
            self.coordinates = np.array([p])
        else:
            raise Exception("ERROR: expected numpy array with 2 or three dimensions, but %s dimensions were found" %p.ndim)

        self.current = 0
        '''index of currently selected conformation'''

        self.points = self.coordinates[self.current]
        '''pointer to currently selected conformation'''

        self.properties = {}
        '''collection of properties. By default, 'center' (geometric center of the Structure) is defined'''

        self.properties['center'] = self.get_center()
   
        idx = np.arange(len(self.points))     
        if isinstance(r, list) or type(r).__module__ == 'numpy':
            if len(r) > 0:
                self.data = pd.DataFrame(r, index=idx, columns=["radius"])
        else:
                rad = r*np.ones(len(self.points))           
                self.data = pd.DataFrame(rad, index=idx, columns=["radius"])
                ''' metadata about each atom (pandas Dataframe)'''

    def __len__(self, dim="atoms"):
        if dim == "atoms":
            return len(self.points)

    def __getitem__(self, key):
        return self.coordinates[key]

    def set_current(self, pos):
        '''
        select current frame (place frame pointer at desired position)

        :param pos: number of alternative conformation (starting from 0)
        '''
        if pos < self.coordinates.shape[0]:
            self.current = pos
            self.points = self.coordinates[self.current]
            self.properties['center'] = self.get_center()
        else:
            raise Exception("ERROR: position %s requested, but only %s conformations available" %(pos, self.coordinates.shape[0]))

    def get_xyz(self, indices=[]):
        '''
        get points coordinates.

        :param indices: indices of points to select. If none is provided, all points coordinates are returned.
        :returns: coordinates of all points indexed by the provided indices list, or all of them if no list is provided.
        '''
        if indices == []:
            return self.points
        else:
            return self.points[indices]

    def set_xyz(self, coords):
        '''
        set point coordinates.

        :param coords: array of 3D points
        '''
        self.coordinates[self.current] = deepcopy(coords)
        self.points = self.coordinates[self.current]

    def add_xyz(self, coords):
        '''
        add a new alternative conformation to the database

        :param coords: array of 3D points, or array of arrays of 3D points (in case multiple alternative coordinates must be added at the same time)
        '''
        # self.coordinates numpy array containing an ensemble of alternative
        # coordinates in 3D space

        if self.coordinates.size == 0 and coords.ndim == 3:
            self.coordinates = deepcopy(coords)
            self.set_current(0)

        elif self.coordinates.size == 0 and coords.ndim == 2:
            self.coordinates = deepcopy(np.array([coords]))
            self.set_current(0)

        elif self.coordinates.size > 0 and coords.ndim == 3:
            self.coordinates = np.concatenate((self.coordinates, coords))
            # set new frame to the first of the newly inserted ones
            self.set_current(self.current + 1)

        elif self.coordinates.size > 0 and coords.ndim == 2:
            self.coordinates = np.concatenate((self.coordinates, np.array([coords])))
            # set new frame to the first of the newly inserted ones
            self.set_current(self.current + 1)

        else:
            raise Exception("ERROR: expected numpy array with 2 or three dimensions, but %s dimensions were found" %np.ndim)

    def delete_xyz(self, index):
        '''
        remove one conformation from the conformations database.

        the new current conformation will be the previous one.

        :param index: alternative coordinates set to remove
        '''
        self.coordinates = np.delete(self.coordinates, index, axis=0)
        if index > 0:
            self.set_current(index - 1)
        else:
            self.set_current(0)

    def clear(self):
        '''
        remove all the coordinates and empty metadata
        '''
        self.coordinates = np.array([[[], []], [[], []]])
        self.points = self.coordinates[0]
        self.data = pd.DataFrame(index=[], columns=[])

    def translate(self, x, y, z):
        '''
        translate the whole structure by a given amount.

        :param x: translation around x axis
        :param y: translation around y axis
        :param z: translation around z axis
        '''

        # if center has not been defined yet (may happen when using
        # subclasses), compute it
        if 'center' not in self.properties:
            self.get_center()

        # translate all points
        self.properties['center'][0] += x
        self.properties['center'][1] += y
        self.properties['center'][2] += z

        # move every frame with first :
        self.coordinates[:, :, 0] += x
        self.coordinates[:, :, 1] += y
        self.coordinates[:, :, 2] += z

    def rotate(self, x, y, z):
        '''
        rotate the structure provided angles of rotation around x, y and z axes (in degrees).

        This is a rotation with respect of the origin.
        Make sure that the center of your structure is at the origin, if you don't want to get a translation as well!
        rotating an object being not centered requires to first translate the ellipsoid at the origin, rotate it, and bringing it back.

        :param x: rotation around x axis
        :param y: rotation around y axis
        :param z: rotation around z axis
        '''
        alpha = np.radians(x)
        beta = np.radians(y)
        gamma = np.radians(z)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), - np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])
        rotation = np.dot(Rx, np.dot(Ry, Rz))
        # multiply rotation matrix with each point of the ellipsoid
        self.apply_transformation(rotation.T)
        self.get_center()

    def apply_transformation(self, M):
        '''
        apply a 3x3 transformation matrix

        :param M: 3x3 transformation matrix (2D numpy array)
        '''
        self.coordinates[self.current, :, :] = np.dot(self.points, M)
        # new memory allocated? Pointer needs to be moved
        self.points = self.coordinates[self.current]

    def get_center(self):
        '''
        compute protein center of geometry (also assigns it to self.properties["center"] key).
        '''
        if len(self.points) > 0:
            self.properties['center'] = np.mean(self.points, axis=0)
        else:
            self.properties['center'] = np.array([0.0, 0.0, 0.0])

        return self.properties['center']

    def center_to_origin(self):
        '''
        move the structure so that its center of geometry is at the origin.
        '''
        c = self.get_center()
        self.translate(-c[0], -c[1], -c[2])

    def get_size(self):
        '''
        compute the dimensions of the object along x, y and z.

        .. note: points radii are not kept into account.
        '''
        x = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        # +self.properties['radius']*2
        y = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        # +self.properties['radius']*2
        z = np.max(self.points[:, 2]) - np.min(self.points[:, 2])
        return np.array([x, y, z])

    def rotation_matrix(self, axis, theta):
        '''
        compute matrix needed to rotate the system around an arbitrary axis (using Euler-Rodrigues formula).

        :param axis: 3d vector (numpy array), representing the axis around which to rotate
        :param theta: desired rotation angle
        :returns: 3x3 rotation matrix
        '''

        # if rotation angle is equal to zero, no rotation is needed
        if theta == 0:
            return np.identity(3)

        # method taken from
        # http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2)
        b, c, d = -axis * np.sin(theta / 2)
        return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

    def get_principal_axes(self):
        '''
        compute Structure's principal axes.

        :returns: 3x3 numpy array, containing the 3 principal axes ranked from smallest to biggest.
        '''
        # method taken from chempy source code, geometry.py, method
        # getMomentOfInertiaTensor()

        # compute moment of inertia tensor
        I0 = np.zeros((3, 3), np.float64)
        for i in range(0, len(self.points), 1):
            mass = 1  # self.mass[atom] / constants.Na
            I0[0, 0] += mass * (self.points[i, 1] * self.points[i, 1] + self.points[i, 2] * self.points[i, 2])
            I0[1, 1] += mass * (self.points[i, 0] * self.points[i, 0] + self.points[i, 2] * self.points[i, 2])
            I0[2, 2] += mass * (self.points[i, 0] * self.points[i, 0] + self.points[i, 1] * self.points[i, 1])
            I0[0, 1] -= mass * self.points[i, 0] * self.points[i, 1]
            I0[0, 2] -= mass * self.points[i, 0] * self.points[i, 2]
            I0[1, 2] -= mass * self.points[i, 1] * self.points[i, 2]

        I0[1, 0] = I0[0, 1]
        I0[2, 0] = I0[0, 2]
        I0[2, 1] = I0[1, 2]

        # Calculate and return the principal moments of inertia and corresponding
        # principal axes for the current geometry.
        e_values, e_vectors = np.linalg.eig(I0)

        indices = np.argsort(e_values)
        e_values = e_values[indices]
        e_vectors = e_vectors.T[indices]

        return e_vectors

    def align_axes(self):
        '''
        Align structure on its principal axes.

        First principal axis aligned along x, second along y and third along z.
        '''

        # this method is inspired from the procedure followed in in VMD's orient package:
        # set I [draw principalaxes $sel]           <--- show/calc the principal axes
        # set A [orient $sel [lindex $I 2] {0 0 1}] <--- rotate axis 2 to match Z
        # $sel move $A
        # set I [draw principalaxes $sel]           <--- recalc principal axes to check
        # set A [orient $sel [lindex $I 1] {0 1 0}] <--- rotate axis 1 to match Y
        # $sel move $A
        # set I [draw principalaxes $sel]           <--- recalc principal axes
        # to check

        # this align axes has been modified to allow us to backmap a structure after alignment
        c = self.get_center()

        # center the Structure
        self.center_to_origin()

        # get principal axes (ranked from smallest to biggest)
        axes = self.get_principal_axes()

        # align smallest principal axis against z axis
        rotvec = np.cross(axes[0], np.array([1, 0, 0]))  # rotation axis
        sine = np.linalg.norm(rotvec)
        cosine = np.dot(axes[0], np.array([1, 0, 0]))
        angle = np.arctan2(sine, cosine)  # angle to rotate around axis

        rotmatrix0 = self.rotation_matrix(rotvec, angle)
        self.apply_transformation(rotmatrix0)

        # compute new principal axes (after previous rotation)
        axes = self.get_principal_axes()

        # align second principal axis against y axis
        rotvec = np.cross(axes[1], np.array([0, 1, 0]))  # rotation axis
        sine = np.linalg.norm(rotvec)
        cosine = np.dot(axes[1], np.array([0, 1, 0]))
        angle = np.arctan2(sine, cosine)  # angle to rotate around axis

        rotmatrix1 = self.rotation_matrix(rotvec, angle)
        self.apply_transformation(rotmatrix1)

        # return the center and matrix for backmapping
        # do the opposite of these transformations
        return c, rotmatrix0, rotmatrix1

    def write_pdb(self, filename, index=[]):
        '''
        write a multi PDB file where every point is a sphere. VdW radius is written into beta factor.

        :param filename: name of file to output
        :param index: list of frame indices to write to file. By default, a multipdb with all frames will be produced.
        '''

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(index) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(index) < len(self.coordinates):
                frames = index
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(index), len(self.coordinates)))

        fout = open(filename, "w")

        for f in frames:
            
            # Build our hexidecimal array if num. of atoms > 99999
            idx_val = np.arange(1, self.coordinates.shape[1] + 1, 1)

            if len(idx_val) > 99999:
                vhex = np.vectorize(hex)
                idx_val = vhex(idx_val)   # convert index values to hexidecimal
                idx_val = [num[2:] for num in idx_val]  # remove 0x at start of hexidecimal number

            for i in range(0, len(self.coordinates[0]), 1):
                #if i > 99999:
                #    nb = hex(i).split('x')[1]
                #else:
                #    nb = str(i)
                
                l = (idx_val[i], "SPH", "SPH", "A", np.mod(i, 9999),
                     self.coordinates[f, i, 0],
                     self.coordinates[f, i, 1],
                     self.coordinates[f, i, 2],
                     self.data['radius'].values[i],
                     1.0, "Z")
                L = 'ATOM  %5s  %-4s%-4s%1s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % l
                fout.write(L)

            fout.write("END\n")

        fout.close()

    def convex_hull(self):
        '''
        compute Structure's convex Hull using QuickHull algorithm.

        .. note:: Qhull available only on scipy >=0.12

        :returns: :func:`Structure <structure.Structure>` object, containing the coordinates of vertices composing the convex hull
        '''
        try:
            from scipy.spatial import ConvexHull
            verts = ConvexHull(self.points)
            return Structure(verts)

        except Exception as e:
            raise Exception("Quick Hull algorithm available in scipy >=0.12!")

    def get_density(self, step=1.0, sigma=1.0, kernel_half_width=5, buff=3):
        '''
        generate density map from points

        :param step: size of cubic voxels, in Angstrom
        :param sigma: gaussian kernel sigma
        :param kernel_half_width: kernel half width, in voxels
        :param buff: padding to add at points cloud boundaries
        :returns: :func:`Density <density.Density>` object, containing a simulated density map
        '''
        pts = self.points

        # rectangular box boundaries
        bnds = np.array([[np.min(pts[:, 0]) - buff, np.max(pts[:, 0]) + buff],
                         [np.min(pts[:, 1]) - buff, np.max(pts[:, 1]) + buff],
                         [np.min(pts[:, 2]) - buff, np.max(pts[:, 2]) + buff]])

        xax = np.arange(bnds[0, 0], bnds[0, 1] + step, step)
        yax = np.arange(bnds[1, 0], bnds[1, 1] + step, step)
        zax = np.arange(bnds[2, 0], bnds[2, 1] + step, step)

        # create empty box
        d = np.zeros((len(xax), len(yax), len(zax)))

        # place Kronecker deltas in mesh grid
        for p in pts:
            xpos = np.argmin(np.abs(xax - p[0]))
            ypos = np.argmin(np.abs(yax - p[1]))
            zpos = np.argmin(np.abs(zax - p[2]))
            d[xpos, ypos, zpos] = 1

        # create 3d gaussian kernel
        window = kernel_half_width * 2 + 1
        shape = (window, window, window)

        m, n, k = [(ss - 1.) / 2. for ss in shape]

        x_ = np.arange(-m, m + 1, 1).astype(int)
        y_ = np.arange(-n, n + 1, 1).astype(int)
        z_ = np.arange(-k, k + 1, 1).astype(int)
        x, y, z = np.meshgrid(x_, y_, z_)

        h = np.exp(-(x * x + y * y + z * z) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh

        # convolve point mesh with 3d gaussian kernel
        b = scipy.signal.fftconvolve(d, h, mode='same')
        b /= np.max(b)

        # prepare density data structure
        from biobox.classes.density import Density
        D = Density()
        D.properties['density'] = b
        D.properties['size'] = np.array(b.shape)
        D.properties['origin'] = np.min(self.points, axis=0) - kernel_half_width / 2.0 + step #np.mean(self.points, axis=0) - step * np.array(b.shape) / 2.0
        D.properties['delta'] = np.identity(3) * step
        D.properties['format'] = 'dx'
        D.properties['filename'] = ''
        D.properties["sigma"] = np.std(b)

        return D

    def rmsf(self, indices=-1, step=1):
        '''
        compute Root Mean Square Fluctuation (RMSF) of selected atoms.

        :param indices: indices of points for which RMSF will be calculated. If no indices list is provided, RMSF of all points will be calculated.
        :param step: timestep between two conformations (useful when using conformations extracted from molecular dynamics)
        :returns: numpy aray with RMSF of all provided indices, in the same order
        '''

        if self.coordinates.shape[0] < 2:
            raise Exception("ERROR: to compute RMSF several conformations must be available!")

        # if no index is provided, compute RMSF of all points
        if indices == -1:
            indices = np.linspace(0, len(self.coordinates[0, :, 0]) - 1, len(self.coordinates[0, :, 0])).astype(int)

        means = np.mean(self.coordinates[:, indices], axis=0)

        # cumulate all squared distances with respect of mean
        d = []
        for i in range(0, self.coordinates.shape[0], 1):
            d.append(np.sum((self.coordinates[i, indices] - means)**2, axis=1))

        # compute square root of sum of mean squared distances
        dist = np.array(d)
        return np.sqrt(np.sum(dist, axis=0) / (float(self.coordinates.shape[0]) * step))

    def pca(self, components, indices=-1):
        '''
        compute Principal Components Analysis (PCA) on specific points within all the alternative coordinates.

        :param components: eigenspace dimensions
        :param indices: points indices to be considered for PCA
        :returns: numpy array of projection of each conformation into the n-dimensional eigenspace
        :returns: sklearn PCA object
        '''
     
        from sklearn.decomposition import PCA

        # define conformational space (flatten coordinates of desired atoms
        if indices != -1:
            X = self.coordinates[:, indices].reshape(
                     (len(self.coordinates), len(indices) * 3))
        else:
            X = self.coordinates.reshape(
                     (self.coordinates.shape[0], self.coordinates.shape[1]*3))

        # calculate system PCA and project conformations into the eigenspace
        pca = PCA(n_components=components)
        pca.fit(X)
        Xproj = pca.transform(X)

        return Xproj, pca

    def rmsd_one_vs_all(self, ref_index, points_index=[], align=False):
        '''
        Calculate the RMSD between all structures with respect of a reference structure.
        uses Kabsch alignement algorithm.

        :param ref_index: index of reference structure in conformations database
        :param points_index: if set, only specific points will be considered for comparison
        :param align: if set to true, all conformations will be aligned to reference (note: cannot be undone!)
        :returns: RMSD of all structures with respect of reference structure (in a numpy array)
        '''

        # see: http://www.pymolwiki.org/index.php/Kabsch#The_Code

        bkpcurrent = self.current

        if ref_index >= len(self.coordinates):
            raise Exception("ERROR: index %s requested, but only %s exist in database" %(len(self.coordinates)))

        # define reference frame, and center it
        if len(points_index) == 0:
            m1 = deepcopy(self.coordinates[ref_index])
        elif isinstance(points_index, list) or type(points_index).__module__ == 'numpy':
            m1 = deepcopy(self.coordinates[ref_index, points_index])
        else:
            raise Exception("ERROR: please, provide me with a list of indices to compute RMSD (or no index at all)")

        L = len(m1)
        COM1 = np.sum(m1, axis=0) / float(L)
        m1 -= COM1
        m1sum = np.sum(np.sum(m1 * m1, axis=0), axis=0)

        RMSD = []
        for i in range(0, len(self.coordinates), 1):

            if i == ref_index:
                RMSD.append(0.0)
            else:

                # define current frame, and center it
                if len(points_index) == 0:
                    m2 = deepcopy(self.coordinates[i])
                elif isinstance(points_index, list) or type(points_index).__module__ == 'numpy':
                    m2 = deepcopy(self.coordinates[i, points_index])

                COM2 = np.sum(m2, axis=0) / float(L)
                m2 -= COM2

                E0 = m1sum + np.sum(np.sum(m2 * m2, axis=0), axis=0)

                # This beautiful step provides the answer. V and Wt are the orthonormal
                # bases that when multiplied by each other give us the rotation matrix, U.
                # S, (Sigma, from SVD) provides us with the error!  Isn't SVD
                # great!
                V, S, Wt = np.linalg.svd(np.dot(np.transpose(m2), m1))

                # if alignement is required, move pointer to current frame, and
                # apply rotation matrix
                if align:
                    self.set_current(i)
                    rotation = np.dot(V, Wt)
                    self.apply_transformation(rotation)

                reflect = float(
                    str(float(np.linalg.det(V) * np.linalg.det(Wt))))

                if reflect == -1.0:
                    S[-1] = -S[-1]
                    V[:, -1] = -V[:, -1]

                rmsdval = E0 - (2.0 * sum(S))
                rmsdval = np.sqrt(abs(rmsdval / L))

                RMSD.append(rmsdval)

        self.set_current(bkpcurrent)
        return np.array(RMSD)

    def rmsd(self, i, j, points_index=[], full=False):
        '''
        Calculate the RMSD between two structures in alternative coordinates ensemble.
        uses Kabsch alignement algorithm.

        :param i: index of the first structure
        :param j: index of the second structure
        :param points_index: if set, only specific points will be considered for comparison
        :param full: if True, RMSD an rotation matrx are returned, RMSD only otherwise
        :returns: RMSD of the two structures. If full is True, the rotation matrix is also returned
        '''

        # see: http://www.pymolwiki.org/index.php/Kabsch#The_Code

        if i >= len(self.coordinates):
            raise Exception("ERROR: index %s requested, but only %s exist in database" %(i, len(self.coordinates)))

        if j >= len(self.coordinates):
            raise Exception("ERROR: index %s requested, but only %s exist in database" %(j, len(self.coordinates)))

        # get first structure and center it
        if len(points_index) == 0:
            m1 = deepcopy(self.coordinates[i])
        elif isinstance(points_index, list) or type(points_index).__module__ == 'numpy':
            m1 = deepcopy(self.coordinates[i, points_index])
        else:
            raise Exception("ERROR: give me a list of indices to compute RMSD, or nothing at all, please!")

        # get second structure
        if len(points_index) == 0:
            m2 = deepcopy(self.coordinates[j])
        elif isinstance(points_index, list) or type(points_index).__module__ == 'numpy':
            m2 = deepcopy(self.coordinates[j, points_index])
        else:
            raise Exception("ERROR: give me a list of indices to compute RMSD, or nothing at all, please!")

        L = len(m1)
        COM1 = np.sum(m1, axis=0) / float(L)
        m1 -= COM1
        m1sum = np.sum(np.sum(m1 * m1, axis=0), axis=0)

        COM2 = np.sum(m2, axis=0) / float(L)
        m2 -= COM2

        E0 = m1sum + np.sum(np.sum(m2 * m2, axis=0), axis=0)

        # This beautiful step provides the answer. V and Wt are the orthonormal
        # bases that when multiplied by each other give us the rotation matrix, U.
        # S, (Sigma, from SVD) provides us with the error!  Isn't SVD great!
        V, S, Wt = np.linalg.svd(np.dot(np.transpose(m2), m1))

        reflect = float(str(float(np.linalg.det(V) * np.linalg.det(Wt))))

        if reflect == -1.0:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        rmsdval = E0 - (2.0 * sum(S))
        if full:
            return np.sqrt(abs(rmsdval / L)), np.matmul(V, Wt)
        else:
            return np.sqrt(abs(rmsdval / L))           

    def rmsd_distance_matrix(self, points_index=[], flat=False):
        '''
        compute distance matrix between structures (using RMSD as metric).

        :param points_index: if set, only specific points will be considered for comparison
        :param flat: if True, returns flattened distance matrix
        :returns: RMSD distance matrix
        '''

        if flat:
            rmsd = []
        else:
            rmsd = np.zeros((len(self.coordinates), len(self.coordinates)))

        for i in range(0, len(self.coordinates) - 1, 1):
            for j in range(i + 1, len(self.coordinates), 1):
                r = self.rmsd(i, j, points_index)

                if flat:
                    rmsd.append(r)
                else:
                    rmsd[i, j] = r
                    rmsd[j, i] = r

        if flat:
            return np.array(rmsd)
        else:
            return rmsd
