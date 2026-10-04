# Copyright (c) 2014-2022 Matteo Degiacomi
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

from copy import deepcopy
import numpy as np
import scipy.signal
import pandas as pd

class Structure(object):
    '''
    A Structure consists of an ensemble of points in 3D space, and metadata associated to each of them.
    '''

    def __init__(self, p=None, r=1.0):
        '''
        Point coordinates and properties data structures are first initialized.
        properties is a dictionary initially containing an entry for 'center' (center of geometry) and 'radius' (average radius of points).

        :param p: coordinates data structure as a mxnx3 numpy array (alternative conformation x atom x 3D coordinate). nx3 numpy array can be supplied, in case a single conformation is present. If not provided, the Structure is empty.
        :param r: radius of every point in dataset (float), or radius of each point (list or numpy array, one value per point)
        '''
        if p is None:
            self.coordinates = np.empty((0, 0, 3))
            '''numpy array containing an ensemble of alternative coordinates in 3D space'''

        elif p.ndim == 3:
            self.coordinates = p

        elif p.ndim == 2:
            self.coordinates = np.array([p])
        else:
            raise Exception("ERROR: expected numpy array with 2 or three dimensions, but %s dimensions were found" %p.ndim)

        self.current = 0
        '''index of currently selected conformation'''

        self._point_to_current()
        '''pointer to currently selected conformation'''

        self.properties = {}
        '''collection of properties. By default, 'center' (geometric center of the Structure) is defined'''

        self.properties['center'] = self.get_center()

        if np.ndim(r) == 0:
            rad = float(r) * np.ones(len(self.points))
        else:
            rad = np.asarray(r, dtype=float)
            if len(rad) != len(self.points):
                raise Exception("ERROR: %s radii provided for %s points" % (len(rad), len(self.points)))

        self.data = pd.DataFrame(rad, index=np.arange(len(self.points)), columns=["radius"])
        ''' metadata about each atom (pandas Dataframe)'''

    def _point_to_current(self):
        '''
        point self.points to the current frame (an empty array if there are no frames).
        '''
        if len(self.coordinates) == 0:
            self.points = np.empty((0, 3))
        else:
            self.points = self.coordinates.view()[self.current]

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
            self._point_to_current()
            self.properties['center'] = self.get_center()
        else:
            raise Exception("ERROR: position %s requested, but only %s conformations available" %(pos, self.coordinates.shape[0]))

    def get_xyz(self, indices=[]):
        '''
        get points coordinates.

        :param indices: indices of points to select. If none is provided, all points coordinates are returned.
        :returns: coordinates of all points indexed by the provided indices list, or all of them if no list is provided.
        '''
        if len(indices) == 0:
            return self.points
        else:
            return self.points[indices]

    def set_xyz(self, coords):
        '''
        set point coordinates.

        :param coords: array of 3D points
        '''
        self.coordinates[self.current] = deepcopy(coords)
        self._point_to_current()

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
            first = len(self.coordinates)
            self.coordinates = np.concatenate((self.coordinates, coords))
            # set new frame to the first of the newly inserted ones
            self.set_current(first)

        elif self.coordinates.size > 0 and coords.ndim == 2:
            first = len(self.coordinates)
            self.coordinates = np.concatenate((self.coordinates, np.array([coords])))
            # set new frame to the first of the newly inserted ones
            self.set_current(first)

        else:
            raise Exception("ERROR: expected numpy array with 2 or three dimensions, but %s dimensions were found" %coords.ndim)


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
        self.coordinates = np.empty((0, 0, 3))
        self.current = 0
        self._point_to_current()
        self.data = pd.DataFrame(index=[], columns=[])

    def translate(self, x, y, z):
        '''
        translate the current conformation by a given amount. Other conformations are not moved.

        :param x: translation around x axis
        :param y: translation around y axis
        :param z: translation around z axis
        '''

        # if center has not been defined yet (may happen when using
        # subclasses), compute it
        if 'center' not in self.properties:
            self.get_center()

        # translate the points of the current conformation
        self.properties['center'] = self.properties['center'] + np.array([x, y, z], dtype=float)
        self.coordinates[self.current] += np.array([x, y, z], dtype=float)
        self._point_to_current()

    def rotate(self, x, y, z):
        '''
        rotate the current conformation provided angles of rotation around x, y and z axes (in degrees). Other conformations are not moved.

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
        apply a 3x3 transformation matrix to the current conformation, as points multiplied on the right (p' = p M). Other conformations are not moved.

        :param M: 3x3 transformation matrix (2D numpy array)
        '''
        self.coordinates[self.current] = np.dot(self.coordinates[self.current], M)
        self._point_to_current()
        self.get_center()

    def get_center(self):
        '''
        compute protein center of geometry (also assigns it to self.properties["center"] key).
        '''
        if len(self.points) > 0:
            self.properties['center'] = np.mean(self.points, axis=0)
        else:
            self.properties['center'] = np.array([0.0, 0.0, 0.0])

        return self.properties['center'].copy()

    def center_to_origin(self):
        '''
        move the current conformation so that its center of geometry is at the origin. Other conformations are not moved.
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
        compute Structure's principal axes, from the inertia tensor of the current frame about its center of geometry.

        The sign of the first two axes is chosen so that their largest component is positive, and the third axis is their cross product, so that the three axes form a right-handed frame (a rotation matrix).

        :returns: 3x3 numpy array, containing the 3 principal axes as rows, ranked from smallest to biggest moment of inertia.
        '''
        # compute moment of inertia tensor (unit masses) about the center of geometry
        pts = self.points - np.mean(self.points, axis=0)
        I0 = np.identity(3) * np.sum(pts * pts) - np.dot(pts.T, pts)

        # Calculate and return the principal moments of inertia and corresponding
        # principal axes for the current geometry. The inertia tensor is symmetric, so eigh
        # applies, and its results are real (from NumPy 2.5, eig always returns complex arrays)
        e_values, e_vectors = np.linalg.eigh(I0)

        indices = np.argsort(e_values)
        e_values = e_values[indices]
        e_vectors = e_vectors.T[indices]

        # the sign of an eigenvector is arbitrary and depends on the linear algebra library.
        # Fix it (largest component positive), so that align_axes is reproducible across
        # platforms, and complete a right-handed frame with the cross product
        largest = e_vectors[np.arange(2), np.argmax(np.abs(e_vectors[:2]), axis=1)]
        e_vectors[:2] = e_vectors[:2] * np.sign(largest)[:, np.newaxis]
        e_vectors[2] = np.cross(e_vectors[0], e_vectors[1])

        return e_vectors

    def align_axes(self):
        '''
        Align the current conformation on its principal axes. Other conformations are not moved.

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

        idx_val = [self._hybrid36(i + 1) for i in range(self.coordinates.shape[1])]

        for f in frames:

            for i in range(0, len(self.coordinates[0]), 1):

                # occupancy 1, radius in the beta factor column
                l = (idx_val[i], "SPH", "SPH", "A", np.mod(i, 9999),
                     self.coordinates[f, i, 0],
                     self.coordinates[f, i, 1],
                     self.coordinates[f, i, 2],
                     1.0,
                     self.data['radius'].values[i], "Z")
                L = 'ATOM  %5s  %-4s%-4s%1s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % l
                fout.write(L)

            fout.write("END\n")

        fout.close()

    @staticmethod
    def _hybrid36(value, width=5):
        '''
        encode a positive integer in hybrid-36, the PDB convention for numbers too large for their field.

        Numbers that fit the field are written in decimal, larger ones in base 36 starting with a letter (e.g. 100000 is A0000 for width 5).

        :param value: integer to encode
        :param width: width of the field
        :returns: string of at most width characters
        '''
        if value < 10**width:
            return str(value)

        block = 26 * 36**(width - 1)
        value -= 10**width
        for digits in ["0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789abcdefghijklmnopqrstuvwxyz"]:
            if value < block:
                value += 10 * 36**(width - 1)
                code = ""
                while value > 0:
                    value, r = divmod(value, 36)
                    code = digits[r] + code
                return code
            value -= block

        raise Exception("ERROR: %s is too large for a hybrid-36 field of width %s" % (value, width))

    def convex_hull(self):
        '''
        compute the convex hull of the current frame using the QuickHull algorithm.

        :returns: :func:`Structure <biobox.classes.structure.Structure>` object, containing the coordinates of vertices composing the convex hull
        '''
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.points)
        return Structure(self.points[hull.vertices])

    def get_density(self, step=1.0, sigma=1.0, kernel_half_width=5, buff=3):
        '''
        generate density map from points

        :param step: size of cubic voxels, in Angstrom
        :param sigma: gaussian kernel sigma
        :param kernel_half_width: kernel half width, in voxels
        :param buff: padding to add at points cloud boundaries
        :returns: :func:`Density <biobox.classes.density.Density>` object, containing a simulated density map
        '''
        axes = self._grid_axes(self.points, step, buff)
        b = self._density_on_grid(self.points, axes, step, sigma, kernel_half_width)

        # prepare density data structure
        from biobox.classes.density import Density
        D = Density()
        D.properties['density'] = b
        D.properties['size'] = np.array(b.shape)
        D.properties['origin'] = np.array([ax[0] for ax in axes])
        D.properties['delta'] = np.identity(3) * step
        D.properties['format'] = 'dx'
        D.properties['filename'] = ''
        D.properties["sigma"] = np.std(b)

        return D

    @staticmethod
    def _grid_axes(pts, step, buff):
        '''
        coordinates of the grid points of a regular grid enclosing a points cloud.

        :param pts: points the grid encloses
        :param step: size of cubic voxels, in Angstrom
        :param buff: padding to add at points cloud boundaries
        :returns: list of three arrays, the coordinates of grid points along x, y and z
        '''
        return [np.arange(np.min(pts[:, i]) - buff, np.max(pts[:, i]) + buff + step, step) for i in range(3)]

    @staticmethod
    def _grid_indices(pts, axes, step):
        '''
        indices of the grid points closest to each point.

        :param pts: points to place on the grid
        :param axes: grid axes, as returned by _grid_axes
        :param step: size of cubic voxels, in Angstrom
        :returns: tuple of three arrays of indices, along x, y and z
        '''
        return tuple(np.clip(np.rint((pts[:, i] - axes[i][0]) / step).astype(int), 0, len(axes[i]) - 1) for i in range(3))

    @staticmethod
    def _density_on_grid(pts, axes, step, sigma, kernel_half_width):
        '''
        density map of points on a grid, convolved with a gaussian kernel and scaled to a maximum of 1.

        :param pts: points to place on the grid
        :param axes: grid axes, as returned by _grid_axes
        :param step: size of cubic voxels, in Angstrom
        :param sigma: gaussian kernel sigma, in voxels
        :param kernel_half_width: kernel half width, in voxels
        :returns: 3D numpy array
        '''
        # count points in their closest grid point
        d = np.zeros([len(ax) for ax in axes])
        np.add.at(d, Structure._grid_indices(pts, axes, step), 1)

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

        return b

    def rmsf(self, indices=-1):
        '''
        compute Root Mean Square Fluctuation (RMSF) of selected atoms over all conformations: the square root of the mean squared displacement of each point from its mean position.

        No superposition is performed, so conformations should be aligned beforehand (e.g. with :func:`rmsd_one_vs_all <biobox.classes.structure.Structure.rmsd_one_vs_all>` and align=True).

        :param indices: indices of points for which RMSF will be calculated. If no indices list is provided, RMSF of all points will be calculated.
        :returns: numpy aray with RMSF of all provided indices, in the same order
        '''

        if self.coordinates.shape[0] < 2:
            raise Exception("ERROR: to compute RMSF several conformations must be available!")

        # if no index is provided, compute RMSF of all points
        if np.ndim(indices) == 0 and indices == -1:
            indices = np.arange(self.coordinates.shape[1])

        means = np.mean(self.coordinates[:, indices], axis=0)

        # cumulate all squared distances with respect of mean
        d = []
        for i in range(0, self.coordinates.shape[0], 1):
            d.append(np.sum((self.coordinates[i, indices] - means)**2, axis=1))

        # compute square root of sum of mean squared distances
        dist = np.array(d)
        return np.sqrt(np.sum(dist, axis=0) / float(self.coordinates.shape[0]))

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
        if not (np.ndim(indices) == 0 and indices == -1):
            X = self.coordinates[:, indices].reshape(
                     (len(self.coordinates), -1))
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
            raise Exception("ERROR: index %s requested, but only %s exist in database" %(ref_index, len(self.coordinates)))

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

                # if V*Wt is improper (determinant -1) it is a rotation combined with a
                # reflection, and would turn the structure into its mirror image. This must be
                # corrected before the alignment uses it, not only for the RMSD value
                if np.linalg.det(V) * np.linalg.det(Wt) < 0.0:
                    S[-1] = -S[-1]
                    V[:, -1] = -V[:, -1]

                # if alignement is required, rotate frame i about its center and move it
                # onto the center of the reference frame
                if align:
                    rotation = np.dot(V, Wt)
                    self.coordinates[i] = np.dot(self.coordinates[i] - COM2, rotation) + COM1

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

        if reflect < 0.0:
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
