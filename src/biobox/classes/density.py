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

import os
import warnings

import scipy.ndimage
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

from biobox.classes.structure import Structure

class Density(Structure):
    '''
    Subclass of :class:`Structure <biobox.classes.structure.Structure>`, allows importing density maps and representing them as a collection of spheres placed on the map's high density regions.
    '''

    def __init__(self):
        '''
        Create an empty density map. A density map is described by the following keys of the self.properties dictionary:

        * density: density map, as a 3D numpy array indexed as [x, y, z]
        * delta: voxel shape, as a 3x3 matrix (voxel size in Angstrom on the diagonal)
        * size: dimensions in voxels
        * origin: coordinates of the bottom-left-front corner of the map, in Angstrom
        * radius: radius of points composing the density map, in Angstrom (default 1.9)
        * sigma: standard deviation of the map intensities (set when a map is loaded)
        * format: format of the loaded file (dx, ccp4, mrc or imod)
        * filename: name of the loaded file
        * scan: rows of [sigma, volume, CCS] filled by :func:`threshold_vol_ccs <biobox.classes.density.Density.threshold_vol_ccs>`
        '''

        super(Density, self).__init__()
        self._reset_info()

    def _reset_info(self, r=1.9):
        '''
        reset all properties related to a density map, and remove all points.

        :param r: radius of points composing the density map, in Angstrom
        '''

        self.properties['density'] = np.array([])  # density map

        self.properties['radius'] = r
        # scaling factor for voxels (default is [1, 1, 1] Angstrom)
        self.properties['delta'] = np.array([])
        self.properties['size'] = np.array([])  # dimensions in voxels
        # bottom-left-front corner of the cube
        self.properties['origin'] = np.array([])
        self.properties['format'] = ""  # file format
        self.properties['scan'] = np.array([])
        self.properties['filename'] = ""

        self.clear()

    def return_density_map(self):
        '''
        :returns: density map as 3D numpy array, indexed as [x, y, z]
        '''
        return self.properties['density']

    def import_map(self, filename, fileformat='dx'):
        '''
        Import density map and fill up the points and properties data structures.

        Points are placed with :func:`place_points <biobox.classes.density.Density.place_points>` default parameters, and no points are placed if that fails.
        On failure to load, properties are reset and a ValueError is raised.

        :param filename: name of density file to load
        :param fileformat: dx (default), ccp4, mrc or imod
        :raises FileNotFoundError: if filename does not exist
        :raises ValueError: if fileformat is not supported, or if the file cannot be loaded
        '''

        if not os.path.exists(filename):
            raise FileNotFoundError("%s not found!" % filename)

        # call format-specific loading functions.
        # function should fill up all required properties in _reset_info(), and
        # load the map as a 3D array, containing intensity values.
        if fileformat not in ('dx', 'ccp4', 'mrc', 'imod'):
            raise ValueError("sorry, format %s is not supported" % fileformat)

        try:
            if fileformat == 'dx':
                self._import_dx(filename)
            else:
                self._import_mrc(filename, fileformat)

            # if any error went undetected during loading (missing information),
            # data structures may be inconsistent
            for key, message in [('density', "density map could not be correctly loaded"),
                                 ('size', "density map information missing"),
                                 ('origin', "map origin information missing"),
                                 ('delta', "voxel size information missing")]:
                if len(self.properties[key]) == 0:
                    raise ValueError(message)

        except Exception as e:
            self._reset_info()
            raise ValueError("could not load %s: %s" % (filename, e)) from e

        self.properties['format'] = fileformat
        self.properties['filename'] = filename
        self.properties["sigma"] = np.std(self.properties['density'])

        # place points instead of voxels (a map with nothing above threshold has no points)
        try:
            self.place_points()
        except ValueError:
            pass

    def import_numpy(self, data, origin=[0, 0, 0], delta=np.identity(3)):
        '''
        import a numpy 3D array to allow manipulation as a density map. Points are not placed.

        The points radius is set to that of a sphere having the volume of one voxel.

        :param data: numpy 3D array
        :param origin: coordinates of bottom left corner of the map, in Angstrom
        :param delta: voxels' shape, as a 3x3 matrix with voxel sizes in Angstrom on the diagonal (default is a cubic voxel of 1 Angstrom-long sides).
        '''

        if len(data.shape) != 3:
            raise ValueError("a 3D numpy array is expected")

        self.properties['density'] = data
        self.properties['origin'] = np.array(origin)
        self.properties['size'] = np.array(data.shape)
        self.properties['delta'] = delta

        # sphere size corresponding to the volume of one voxel
        voxel_volume = self.properties['delta'][0, 0] * self.properties['delta'][1, 1] * self.properties['delta'][2, 2]
        self.properties['radius'] = (voxel_volume * 3 / (4 * np.pi))**(1 / 3.0)
        self.properties["sigma"] = np.std(self.properties['density'])


    def get_oversampled_points(self, sigma=0):
        '''
        return points obtained by oversampling the map (double points on every axis). The points stored in the structure are not changed.

        Each voxel above threshold (border voxels excluded) fills its oversampled voxel and six oversampled neighbours. The arrangement is shrunk by the sphere radius via ``_shrink_scale``.

        :param sigma: place points only on voxels having intensity greater than this threshold, in multiples of the map standard deviation
        :returns: points placed on oversampled voxels above threshold, as an (n, 3) numpy array
        :returns: radius of produced points (2/3 of the map points radius), in Angstrom
       '''

        thresh = self.get_thresh_from_sigma(sigma)

        oversampled_data = np.zeros((self.properties['size'][0] * 2,
                                     self.properties['size'][1] * 2,
                                     self.properties['size'][2] * 2))

        for x in range(1, self.properties['size'][0] - 1, 1):
            for y in range(1, self.properties['size'][1] - 1, 1):
                for z in range(1, self.properties['size'][2] - 1, 1):
                    if self.properties['density'][x, y, z] > thresh:
                        oversampled_data[x * 2, y * 2, z * 2] = self.properties['density'][x, y, z]
                        oversampled_data[x * 2 + 1, y * 2, z * 2] = self.properties['density'][x, y, z]
                        oversampled_data[x * 2, y * 2 + 1, z * 2] = self.properties['density'][x, y, z]
                        oversampled_data[x * 2, y * 2, z * 2 + 1] = self.properties['density'][x, y, z]
                        oversampled_data[x * 2 - 1, y * 2, z * 2] = self.properties['density'][x, y, z]
                        oversampled_data[x * 2, y * 2 - 1, z * 2] = self.properties['density'][x, y, z]
                        oversampled_data[x * 2, y * 2, z * 2 - 1] = self.properties['density'][x, y, z]

        delta = self.properties['delta'] / 2
        radius = self.properties['radius'] * 2 / 3

        # define scaling to shrink everything by a size equal to spheres radius
        idx = np.transpose(np.where(oversampled_data > thresh))
        scaling = 8.0 / 3.0
        scale = self._shrink_scale(idx, np.diag(delta), radius * scaling, unit_voxel_branch=False)

        # create structure data (ensemble of points and their center, and store
        # its geometric center)
        points = idx * scale + self.properties['origin'] + np.ones(3) * self.properties['radius'] * 2 / 3
        return points, radius

    @staticmethod
    def _shrink_scale(idx, delta, shrink, unit_voxel_branch=True):
        '''
        per-axis scaling of voxel indices that shrinks the points arrangement by the size of its spheres.

        Along an axis, the scaling is (delta - 1) / (scaled_size - size) * (scaled_size - shrink), where size is the extent of the indices and scaled_size the same extent in Angstrom.
        With unit_voxel_branch, an axis whose voxel size is 1 is not scaled. An axis along which all indices are equal has no extent to shrink, and is scaled by its voxel size.

        :param idx: voxel indices, as an (n, 3) array
        :param delta: voxel size along each axis
        :param shrink: length the arrangement is shrunk by, in Angstrom
        :param unit_voxel_branch: if True, axes with voxel size 1 are left unscaled
        :returns: scaling along each axis
        '''
        size = np.max(idx, axis=0) - np.min(idx, axis=0)
        scaled_size = np.max(idx * delta, axis=0) - np.min(idx * delta, axis=0)
        scale = np.zeros(3)
        for a in range(3):
            if unit_voxel_branch and delta[a] == 1:
                scale[a] = 1.0
            elif scaled_size[a] - size[a] == 0:
                scale[a] = delta[a]
            else:
                scale[a] = (delta[a] - 1.0) / (scaled_size[a] - size[a]) * (scaled_size[a] - shrink)
        return scale

    def get_thresh_from_sigma(self, val):
        '''
        convert cutoff value from sigma multiples into actual threshold

        :param val: threshold in multiples of the map standard deviation
        :returns: threshold in map intensity units
        '''

        return self.properties["sigma"] * val

    def get_sigma_from_thresh(self, t):
        '''
        convert cutoff value from actual threshold to sigma multiple

        :param t: threshold in map intensity units
        :returns: threshold in multiples of the map standard deviation
        '''

        return t / float(self.properties["sigma"])

    def place_points(self, sigma=0, noise_filter=0.01):
        '''
        given density information, place points on every voxel above a given threshold. Existing points are removed, and every point gets the map points radius.

        The arrangement is shrunk by the sphere radius via ``_shrink_scale``.
        :param sigma: intensity threshold, in multiples of the map standard deviation (default 0)
        :param noise_filter: launch DBSCAN clustering algorithm (eps equal to the voxel diagonal, 10 minimum samples) to detect connected regions in density map. Regions representing less than noise_filter of the total, and points labelled as noise, are removed. This is a ratio, value should be between 0 and 1 (default 0.01). If 0, no filtering is done.
        :raises ValueError: if no voxel is above threshold, if noise_filter is out of range, or if the noise filter removes every point
        '''

        thresh = self.get_thresh_from_sigma(sigma)

        if not np.any(self.properties['density'] > thresh):
            raise ValueError("selected threshold leads to empty point ensemble")

        if noise_filter >= 1 or noise_filter < 0:
            raise ValueError("noise_filter should be between 0 and 1")

        # define scaling to shrink everything by a size equal to spheres radius
        idx = np.transpose(np.where(self.properties['density'] > thresh))
        scale = self._shrink_scale(idx, np.diag(self.properties['delta']), self.properties['radius'])

        # create structure data (ensemble of points and their center, and store
        # its geometric center)
        points = idx * scale + self.properties['origin'] + np.ones(3) * self.properties['radius']

        # remove previous points arrangement, and create new one (necessary,
        # since the amount of points will change, and cannot therefore be
        # considered a new conformation)
        self.clear()

        # apply DBSCAN noise filter
        if noise_filter != 0:
            step = self.properties['delta'][0, 0] * np.sqrt(3)
            db = DBSCAN(eps=step, min_samples=10).fit(points)
            keep = np.zeros(len(points), dtype=bool)
            for i in np.unique(db.labels_):
                if np.sum(db.labels_ == i) / float(len(points)) > noise_filter and i != -1:
                    keep = np.logical_or(keep, db.labels_ == i)

            if not np.any(keep):
                raise ValueError("noise filter removed every point")

            self.add_xyz(points[keep])

        else:
            self.add_xyz(points)

        # update radii list (useful for instance for CCS calculation)
        idx = np.arange(len(self.points))
        self.data = pd.DataFrame(self.properties["radius"], index=idx, columns=["radius"])

        # self.get_center()

    def predict_ccs_from_mass(self, resolution, mass, density=0.84, x0=2.51911893, y0=-1.06481492, c=2.7018764, k=0.44084821):

        '''
        given target mass and map resolution, predict CCS. Mass threshold is rescaled using the fitting function c / (1 + exp(-k*(resolution-x0))) + y0.

        Requires a scan produced by :func:`threshold_vol_ccs <biobox.classes.density.Density.threshold_vol_ccs>`.

        :param resolution: map resolution in Angstrom
        :param mass: protein mass in kDa
        :param density: protein density in Da/A3
        :param x0: sigmoid parameter
        :param y0: sigmoid parameter
        :param c: sigmoid parameter
        :param k: sigmoid parameter
        :returns: CCS estimated from mass and density map resolution, in A^2
        :returns: threshold the CCS corresponds to, in multiples of the map standard deviation
        :raises RuntimeError: if the stored scan is empty
        '''

        if np.size(self.properties.get('scan', [])) == 0:
            raise RuntimeError("no threshold to volume to CCS relationship loaded yet. Please execute threshold_vol_ccs method.")

        data=self.properties['scan'].copy()
        data[:,1]*=density
        data[:,1]/=1000.0

        #get mass threshold
        dtest1=np.argmin(np.abs(data[:,1]-mass))
        thresh=data[dtest1,0]

        #rescale mass threshold
        scaling= c / (1 + np.exp(-k*(resolution-x0))) + y0
        calibrated_thresh=thresh/scaling

        #get associated CCS
        dtest1=np.argmin(np.abs(data[:,0]-calibrated_thresh))
        return data[dtest1,2], data[dtest1,0]



    def predict_mass_from_ccs(self, resolution, ccs, density=0.84, x0=2.51911893, y0=-1.06481492, c=2.7018764, k=0.44084821):
        '''
        given target CCS and map resolution, predict mass. The CCS threshold is rescaled using the fitting function c / (1 + exp(-k*(resolution-x0))) + y0.

        Requires a scan produced by :func:`threshold_vol_ccs <biobox.classes.density.Density.threshold_vol_ccs>`.

        :param resolution: map resolution in Angstrom
        :param ccs: target CCS in A^2
        :param density: protein density in Da/A3
        :param x0: sigmoid parameter
        :param y0: sigmoid parameter
        :param c: sigmoid parameter
        :param k: sigmoid parameter
        :returns: mass estimated from CCS and density map resolution, in kDa
        :returns: threshold the mass corresponds to, in multiples of the map standard deviation
        :raises RuntimeError: if the stored scan is empty
        '''

        if np.size(self.properties.get('scan', [])) == 0:
            raise RuntimeError("no threshold to volume to CCS relationship loaded yet. Please execute threshold_vol_ccs method.")

        data=self.properties['scan'].copy()
        data[:,1]*=density
        data[:,1]/=1000.0

        #get mass threshold
        dtest1=np.nanargmin(np.abs(data[:,2]-ccs))
        thresh=data[dtest1,0]

        #rescale mass threshold
        scaling= c / (1 + np.exp(-k*(resolution-x0))) + y0
        calibrated_thresh=thresh*scaling

        #get associated CCS
        dtest1=np.argmin(np.abs(data[:,0]-calibrated_thresh))
        return data[dtest1,1], data[dtest1,0]


    def scan_threshold(self, mass, density=0.782878356, sampling_points=1000, verbose=False):
        '''
        if mass and density of object are known, filter the map on a linear scale of threshold values, and compare the obtained mass to the experimental one.
        Points are finally placed at the threshold giving the smallest absolute mass error.

        .. note:: in proteins, an average value of 1.3 g/cm^3 (0.782878356 Da/A^3) can be assumed. Alternatively, the relation density=1.410+0.145*exp(-mass(kDa)/13) (in g/cm^3) can be used.

        .. note:: 1 g/cm^3=0.602214120 Da/A^3

        :param mass: target mass in Da
        :param density: target density in Da/A^3
        :param sampling_points: number of measures to perform between min and max intensity in density map
        :param verbose: if True, print each tested value and its mass error
        :returns: array of shape (sampling_points, 2) reporting tested values and error on mass ([sigma, model_mass-target_mass]), with mass in Da
        '''

        low = self.get_sigma_from_thresh(np.min(self.properties['density']))
        high = self.get_sigma_from_thresh(np.max(self.properties['density']))

        result = []
        for thresh in np.linspace(low, high, num=sampling_points):

            try:
                self.place_points(thresh)
                vol = self.get_volume()
            except Exception as e:
                vol = 0.0

            if verbose:
                print("threshold=%s, error=%s" % (thresh, vol * density - mass))
            result.append([thresh, vol * density - mass])

        r = np.array(result)
        bestthresh = r[np.argmin(np.abs(r[:, 1]) - mass), 0]

        try:
            self.place_points(bestthresh)
        except Exception as ex:
            pass

        return r

    def _volume_and_ccs(self, sigma, noise_filter):
        '''
        place points at a threshold and measure the volume and CCS of the result.

        A threshold leaving no points gives volume and CCS 0. A CCS that cannot be computed (e.g. IMPACT unavailable) is NaN, with a UserWarning, and the volume is kept.

        :param sigma: density threshold, in multiples of the map standard deviation
        :param noise_filter: see :func:`place_points <biobox.classes.density.Density.place_points>`
        :returns: volume, in A^3
        :returns: CCS, in A^2
        '''
        import biobox as bb

        try:
            self.place_points(sigma, noise_filter=noise_filter)
        except ValueError:
            return 0.0, 0.0

        vol = self.get_volume()
        try:
            ccs = bb.ccs(self)
        except Exception as ex:
            warnings.warn("CCS not computed at sigma %s: %s" % (sigma, ex), stacklevel=3)
            ccs = np.nan

        return vol, ccs

    def _append_scan(self, rows):
        '''
        add rows of [sigma, volume, CCS] to the scan already stored.

        :param rows: a row or an (n, 3) array of rows
        '''
        rows = np.atleast_2d(rows)
        if self.properties['scan'].size == 0:
            self.properties['scan'] = rows
        else:
            self.properties['scan'] = np.vstack((self.properties['scan'], rows))

    def find_data_from_sigma(self, sigma, exact=True, append=False, noise_filter=0.01):
        '''
        map experimental data to given threshold

        :param sigma: density threshold, in multiples of the map standard deviation
        :param exact: if True, measure volume and CCS at this threshold (points are placed at this threshold). Otherwise, return the row of the scan already stored having the closest sigma.
        :param append: if True and exact is True, add the measurement to the stored scan
        :param noise_filter: launch DBSCAN clustering algorithm to detect connected regions in density map. Regions representing less than noise_filter of the total will be removed. This is a ratio, value should be between 0 and 1. Used only if exact is True.
        :returns: array [sigma, volume (A^3), CCS (A^2)]
        '''

        if exact:
            vol, ccs = self._volume_and_ccs(sigma, noise_filter)
            res = np.array([sigma, vol, ccs])

            if append:
                self._append_scan(res)

            return res

        else:
            return self.properties['scan'][
                np.argmin(np.abs(self.properties['scan'][:, 0] - sigma))]


    def find_data_from_volume(self, vol):
        '''
        map experimental data to given volume, using the scan already stored

        :param vol: target volume, in A^3
        :returns: row of the stored scan having the closest volume, as array [sigma, volume (A^3), CCS (A^2)]
        '''

        return self.properties['scan'][
            np.argmin(np.abs(self.properties['scan'][:, 1] - vol))]

    def find_data_from_ccs(self, ccs):
        '''
        map experimental data to given ccs, using the scan already stored (NaN CCS are ignored)

        :param ccs: target CCS (in A^2)
        :returns: row of the stored scan having the closest CCS, as array [sigma, volume (A^3), CCS (A^2)]
        '''

        return self.properties['scan'][
            np.nanargmin(np.abs(self.properties['scan'][:, 2] - ccs))]

    def threshold_vol_ccs(self, low="", high="", sampling_points=1000, append=False, noise_filter=0.01, verbose=False):
        '''
        return the volume to threshold to CCS relationship, and store it in self.properties['scan']. Points are left placed at the last tested threshold.

        :param low: lowest threshold tested, in multiples of the map standard deviation (default: the minimum map intensity, expressed in the same units)
        :param high: highest threshold tested, in multiples of the map standard deviation (default: the maximum map intensity, expressed in the same units)
        :param sampling_points: number of thresholds tested, linearly spaced between low and high
        :param append: if True, add the results to the scan already stored, otherwise replace it
        :param noise_filter: see :func:`place_points <biobox.classes.density.Density.place_points>`
        :param verbose: if True, print each measure
        :returns: array of shape (sampling_points, 3) reporting tested values, volumes and CCS ([sigma, volume (A^3), CCS (A^2)]). A threshold leaving no points gives volume and CCS 0, and a CCS that could not be computed is NaN.
        '''

        if low == "":
            low = self.get_sigma_from_thresh(np.min(self.properties['density']))
        if high == "":
            high = self.get_sigma_from_thresh(np.max(self.properties['density']))

        result = []
        for thresh in np.linspace(low, high, num=sampling_points):
            vol, ccs = self._volume_and_ccs(thresh, noise_filter)

            if verbose:
                print("thresh: %s, vol=%s, ccs=%s (%s points)" % (thresh, vol, ccs, len(self.points)))

            result.append([thresh, vol, ccs])

        r = np.array(result)

        if append:
            self._append_scan(r)

        else:
            self.properties['scan'] = r

        return r

    def best_threshold(self, mass, density=0.782878356):
        '''
        If mass and density of object are known, try to filter the map so that the mass is best matched.

        search for best threshold using bisection method, until the mass error repeats or is zero. Points are finally placed at the tested threshold giving the smallest absolute mass error.

        .. note:: in proteins, an average value of 1.3 g/cm^3 (0.782878356 Da/A^3) can be assumed. Alternatively, the relation density=1.410+0.145*exp(-mass(kDa)/13) (in g/cm^3) can be used.

        .. note:: 1 Da/A^3=1.660538946 g/cm^3
        .. note:: 1 g/cm^3=0.602214120 Da/A^3

        :param mass: target mass in Da
        :param density: target density in Da/A^3
        :returns: array reporting tested values and error on mass ([sigma, model_mass-target_mass]), with mass in Da, one row per bisection step
        '''

        high = self.get_sigma_from_thresh(np.max(self.properties['density']))
        low = self.get_sigma_from_thresh(np.min(self.properties['density']))

        high_val = 1000000000
        low_val = -1000000000
        error = high_val

        result = []
        while True:
            thresh = (high + low) / 2.0

            try:
                self.place_points(thresh)
                vol = self.get_volume()
            except Exception as ex:
                vol = 0

            mass_model = vol * density
            error = mass_model - mass
            result.append([thresh, error])

            if error == high_val or error == low_val:
                break

            if error < 0:
                high = thresh
                high_val = error
            elif error > 0:
                low = thresh
                low_val = error
            else:
                break

        r = np.array(result)
        bestthresh = r[np.argmin(np.abs(r[:, 1])), 0]

        try:
            self.place_points(bestthresh)
        except Exception as ex:
            pass

        return r

    def blur(self, dimension=5, sigma=0.5):
        '''
        blur density applying a cubic gaussian kernel of given kernel dimension (cubic grid size), and update the map standard deviation. Points are not updated.

        .. warning:: cannot be undone

        :param dimension: size of the kernel grid, in voxels.
        :param sigma: standard deviation of gaussian kernel, in voxels.
        '''

        shape = (dimension, dimension, dimension)

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

        dens = scipy.ndimage.convolve(self.properties['density'], h, mode='constant')
        self.properties['density'] = dens
        self.properties["sigma"] = np.std(self.properties['density'])

    def write_dx(self, filename='dens.dx'):
        '''
        Write a density map in DX format

        :param filename: output file name
        '''

        dens = self.properties['density']
        origin = self.properties['origin']
        delta = self.properties['delta']

        fout = open(filename, "w")
        fout.write("# density generated with SBT\n#\n#\n#\n")
        fout.write("object 1 class gridpositions counts %s %s %s\n" % (dens.shape[0], dens.shape[1], dens.shape[2]))
        fout.write("origin %s %s %s\n"%(origin[0], origin[1], origin[2]))
        fout.write("delta %s %s %s\n"%(delta[0, 0], delta[0, 1], delta[0, 2]))
        fout.write("delta %s %s %s\n" % (delta[1, 0], delta[1, 1], delta[1, 2]))
        fout.write("delta %s %s %s\n" %(delta[2, 0], delta[2, 1], delta[2, 2]))
        fout.write("object 2 class gridconnections counts %s %s %s\n"%(dens.shape[0], dens.shape[1], dens.shape[2]))
        fout.write("object 3 class array type double rank 0 items %i data follows\n"%(dens.shape[0] * dens.shape[1] * dens.shape[2]))

        cnt = 0
        for xpos in range(0, dens.shape[0], 1):
            for ypos in range(0, dens.shape[1], 1):
                for zpos in range(0, dens.shape[2], 1):
                    fout.write("%s " % dens[xpos, ypos, zpos])
                    cnt += 1
                    if cnt % 3 == 0:
                        fout.write("\n")

        fout.close()

    def export_as_pdb(self, filename, step, threshold=0.1):
        '''
        Write a pdb file with points where the density exceeds a threshold. A voxel of index i is written at coordinate i/step + origin.

        :param filename: output file name
        :param step: number of voxels per Angstrom
        :param threshold: density (in map intensity units) to be exceeded to generate a point in pdb
        '''

        # @todo assign spheres beta factor to associated density value

        dens = self.properties['density']
        origin = self.properties['origin']

        fout = open(filename, 'w')

        cnt = 1
        identifier = 'ATOM'  # atom to be used to mimick density
        symbol = 'H'  # element for atom

        for xpos in range(0, dens.shape[0], 1):
            for ypos in range(0, dens.shape[1], 1):
                for zpos in range(0, dens.shape[2], 1):
                    if dens[xpos, ypos, zpos] > threshold:
                        x_coord = float(xpos / step) + origin[0]
                        y_coord = float(ypos / step) + origin[1]
                        z_coord = float(zpos / step) + origin[2]
                        L = '%-6s%5s  %-4s%-4s  DIS    %8.3f%8.3f%8.3f  1.00  0.00            \n'%(identifier, cnt, symbol, symbol, x_coord, y_coord, z_coord)
                        fout.write(L)
                        cnt += 1

        fout.close()

    def _import_dx(self, filename):
        '''
        import density map in dx format, and fill up the density, size, origin and delta properties.

        :param filename: name of dx file to load
        '''

        try:
            fin = open(filename, "r")
        except Exception as e:
            raise FileNotFoundError('opening of file %s failed!' % filename) from e

        d = []
        dlt = []
        for line in fin:
            w = line.split()
            if len(w) == 0 or w[0] == "#":
                continue
            elif len(w) <= 3: #and np.array(list(w)).dtype == ('float'):
                try:
                    w = np.array(w).astype(float)
                    for i in range(len(w)):
                        d.append(w[i])
                except Exception as e:
                    continue
                # get coordinates
            elif len(w) > 2 and w[0] == "object" and w[3] == "gridpositions":
                self.properties['size'] = np.array([w[-3], w[-2], w[-1]]).astype(int)
            # get scaling factor
            elif len(w) > 2 and w[0] == "delta":
                dlt.append(w[1:4])
            # get position of origin
            elif len(w) > 2 and w[0] == "origin":
                self.properties['origin'] = np.array(w[1:4]).astype(float)
        fin.close()

        # scaling factor with respect of unit cell voxels
        self.properties['delta'] = np.array(dlt).astype(float)

        try:
            self.properties['density'] = np.reshape(
                np.array(d).astype(float),
                (self.properties['size'][0],
                 self.properties['size'][1],
                 self.properties['size'][2]))
        except Exception as ex:
            raise ValueError("reshaping of dx data failed! Dimensions and dataset size are inconsistent!") from ex

    def _import_mrc(self, filename, fileformat):
        '''
        import density map in MRC, CCP4 or IMOD format, and fill up the density, origin, size, delta and radius properties.

        MRC format here: www2.mrc-lmb.cam.ac.uk/image2000.html

        CCP4 format here: www.ccp4.ac.uk/html/maplib.html

        :param filename: name of MRC or CCP4 file to load
        :param fileformat: can be mrc, imod or ccp4
        '''

        import biobox.classes.density_MRC as MRC
        [density, data] = MRC.read_density(filename, fileformat)
        self.properties['density'] = density
        self.properties['origin'] = np.array(data.origin)
        self.properties['size'] = np.array(density.shape)
        self.properties['delta'] = np.identity(3) * data.mrc_data.data_step

        # sphere size corresponding to the volume of one voxel
        voxel_volume = self.properties['delta'][0, 0] * self.properties['delta'][1, 1] * self.properties['delta'][2, 2]
        self.properties['radius'] = (voxel_volume * 3 / (4 * np.pi))**(1 / 3.0)


    def get_volume(self):
        '''
        compute density map volume. This is done by counting the points, and multiplying that by voxels' volume.

        .. warning:: can be called only after :func:`place_points <biobox.classes.density.Density.place_points>` has been called.

        .. warning:: supposes unskewed voxels.

        :returns: volume in A^3
        '''
        return self.properties['delta'][0, 0] * self.properties['delta'][1, 1] * self.properties['delta'][2, 2] * len(self.points)



if __name__ == "__main__":

    import biobox as bb

    print("loading density...")
    D = bb.Density()
    D.import_map("..%stest%sEMD-1080.mrc"%(os.sep, os.sep), "mrc")

    print("origin: %s"%np.array(D.properties["origin"]))
    print("shape: %s"%np.array(D.properties["density"].shape))
    print("delta: %s"%(D.properties["delta"]))

    # test points placement
    D.place_points(4)

    # test CCS calculation and prediction
    D.threshold_vol_ccs(sampling_points=50, append=False)#, noise_filter=1)
    ccs = D.predict_ccs_from_mass(11.5, 801)
    print(ccs)
