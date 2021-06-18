# Copyright (c) 2014-2021 Matteo Degiacomi
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

import os

import scipy.ndimage.filters
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

from biobox.classes.structure import Structure

class Density(Structure):
    '''
    Subclass of :func:`Structure <structure.Structure>`, allows importing density map, and transform them in a PDB file containing a collection of spheres placed on the map's high density regions.
    '''

    def __init__(self):
        '''
        A density map is fully described by the following attributes, stored in the self.properties dictionary:
        
        :param density: density map
        :param delta:   scaling factor for voxels (default is [1, 1, 1] Angstrom)
        :param size:    dimensions in voxels
        :param origin:  bottom-left-front corner of the cube
        :param radius: radius of points composing the density map
        :param format: format name (only dx supported at the moment)
        '''

        super(Density, self).__init__()
        self._reset_info()

    def _reset_info(self, r=1.9):
        '''
        reset all properties related to a density map (clean)
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
        :returns: density map as 3D numpy array
        '''
        return self.properties['density']

    def import_map(self, filename, fileformat='dx'):
        '''
        Import density map and fill up the points and properties data structures.

        :param  filename: name of density file to load
        :param  fileformat: at the moment supports dx, ccp4, mrc and imod
        '''

        if not os.path.exists(filename):
            raise Exception("%s not found!" % filename)

        # call format-specific loading functions.
        # function should fill up all required properties in _reset_info(), and
        # load the map as a 3D array, containing intensity values.
        try:
            if fileformat == 'dx':
                self._import_dx(filename)
            elif fileformat == 'ccp4':
                self._import_mrc(filename, 'ccp4')
            elif fileformat == 'mrc':
                self._import_mrc(filename, 'mrc')
            elif fileformat == 'imod':
                self._import_mrc(filename, 'imod')
            else:
                raise Exception("sorry, format %s is not supported" % fileformat)

        except Exception as e:
            self._reset_info()
            Exception("ERROR: %s" % e)

        # if any error went undetected during loading (missing information),
        # data structures may be inconsistent. Call cleaning procedure!
        if len(self.properties['density']) == 0:
            print("density map could not be correctly loaded!")
            self._reset_info(self)
        elif len(self.properties['size']) == 0:
            print("density map information missing!")
            self._reset_info(self)
        elif len(self.properties['origin']) == 0:
            print("map origin information missing!")
            self._reset_info(self)
        elif len(self.properties['delta']) == 0:
            print("voxel size information missing!")
            self._reset_info(self)

        # if all required information is present, place points instead of
        # voxels
        else:
            try:
                self.place_points()
            except Exception as e:
                pass

            self.properties['format'] = format
            self.properties['filename'] = filename

        self.properties["sigma"] = np.std(self.properties['density'])

    def import_numpy(self, data, origin=[0, 0, 0], delta=np.identity(3)):
        '''
        import a numpy 3D array to allow manipulation as a density map

        :param data: numpy 3D array
        :param origin: coordinates of bottom left corner of the map
        :param delta: voxels' shape (default is a cubic voxel of 1 Angstrom-long sides).
        '''
                
        if len(data.shape) != 3:
            raise Exception("ERROR: a 3D numpy array is expected")
        
        self.properties['density'] = data
        self.properties['origin'] = np.array(origin)
        self.properties['size'] = np.array(data.shape)
        self.properties['delta'] = delta

        # sphere size corresponding to the volume of one voxel
        voxel_volume = self.properties['delta'][0, 0] * self.properties['delta'][1, 1] * self.properties['delta'][2, 2]
        self.properties['radius'] = (voxel_volume * 3 / (4 * np.pi))**(1 / 3.0)


    def get_oversampled_points(self, sigma=0):
        '''
        return points obtained by oversampling the map (doule points on every axis)

        :param sigma: place points only on voxels having intensity greater than threshold
        :returns: points 3D points placed on voxels having value higher than threshold
        :returns: radius radius of produced points
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
        size = np.max(np.transpose(np.where(oversampled_data > thresh)), axis=0) - np.min(np.transpose(np.where(oversampled_data > thresh)), axis=0)
        scaled_size = np.max(np.transpose(np.where(oversampled_data > thresh)) * np.diag(delta), axis=0) - np.min(np.transpose(np.where(oversampled_data > thresh)) * np.diag(delta), axis=0)

        scaling = 8.0 / 3.0
        scale_x = (delta[0, 0] - 1.0) / (scaled_size[0] - size[0]) * (scaled_size[0] - radius * scaling)
        scale_y = (delta[1, 1] - 1.0) / (scaled_size[1] - size[1]) * (scaled_size[1] - radius * scaling)
        scale_z = (delta[2, 2] - 1.0) / (scaled_size[2] - size[2]) * (scaled_size[2] - radius * scaling)

        # create structure data (ensemble of points and their center, and store
        # its geometric center)
        points = np.transpose(np.where(oversampled_data > thresh)) * np.array([scale_x, scale_y, scale_z]) + self.properties['origin'] + np.ones(3) * self.properties['radius'] * 2 / 3
        return points, radius

    def get_thresh_from_sigma(self, val):
        '''
        convert cutoff value from sigma multiples into actual threshold

        :param val: sigma scaling
        :returns: cutoff
        '''

        return self.properties["sigma"] * val

    def get_sigma_from_thresh(self, t):
        '''
        convert cutoff value from actual threshold to sigma multiple

        :param threshold value
        :returns sigma multiple
        '''

        return t / float(self.properties["sigma"])

    def place_points(self, sigma=0, noise_filter=0.01):
        '''
        given density information, place points on every voxel above a given threshold.

        :param sigma: intensity threshold value.
        :param noise_filter: launch DBSCAN clustering algorithm to detect connected regions in density map. Regions representing less than noise_filter of the total will be removed. This is a ratio, value should be between 0 and 1.
        '''

        thresh = self.get_thresh_from_sigma(sigma)

        if not np.any(self.properties['density'] > thresh):
            raise IOError("selected threshold leads to empty point ensemble")

        if noise_filter >= 1 or noise_filter < 0:
            raise IOError("noise_filter should be between 0 and 1")

        # define scaling to shrink everything by a size equal to spheres radius
        size = np.max(np.transpose(np.where(self.properties['density'] > thresh)), axis=0) - np.min(np.transpose(np.where(self.properties['density'] > thresh)), axis=0)
        scaled_size = np.max(np.transpose(np.where(self.properties['density'] > thresh)) * np.diag(self.properties['delta']), axis=0)-np.min(np.transpose(np.where(self.properties['density'] > thresh)) * np.diag(self.properties['delta']), axis=0)

        if self.properties['delta'][0, 0] != 1:
            scale_x = (self.properties['delta'][0, 0] - 1.0) / (scaled_size[0] - size[0]) * (scaled_size[0] - self.properties['radius'])
        else:
            scale_x = 1.0

        if self.properties['delta'][1, 1] != 1:
            scale_y = (self.properties['delta'][1, 1] - 1.0) / (scaled_size[1] - size[1]) * (scaled_size[1] - self.properties['radius'])
        else:
            scale_y = 1.0

        if self.properties['delta'][2, 2] != 1:
            scale_z = (self.properties['delta'][2, 2] - 1.0) / (scaled_size[2] - size[2]) * (scaled_size[2] - self.properties['radius'])
        else:
            scale_z = 1
        # create structure data (ensemble of points and their center, and store
        # its geometric center)
        points = np.transpose(np.where(self.properties['density'] > thresh)) * np.array([scale_x, scale_y, scale_z]) + self.properties['origin'] + np.ones(3) * self.properties['radius']

        # remove previous points arrangement, and create new one (necessary,
        # since the amount of points will change, and cannot therefore be
        # considered a new conformation)
        self.clear()

        # apply DBSCAN noise filter
        if noise_filter != 0:
            step = self.properties['delta'][0, 0] * np.sqrt(3)
            db = DBSCAN(eps=step, min_samples=10).fit(points)
            pts2 = []
            for i in np.unique(db.labels_):
                if np.sum(db.labels_ == i) / float(len(points)) > 0.01 and i != -1:
                    if len(pts2) == 0:
                        pts2 = points[db.labels_ == i]
                    else:
                        pts2 = np.concatenate((pts2, points[db.labels_ == i]))

            self.add_xyz(pts2)

        else:
            self.add_xyz(points)

        # update radii list (useful for instance for CCS calculation)
        idx = np.arange(len(self.points))
        self.data = pd.DataFrame(self.properties["radius"], index=idx, columns=["radius"])

        # self.get_center()

    def predict_ccs_from_mass(self, resolution, mass, density=0.84, x0=2.51911893, y0=-1.06481492, c=2.7018764, k=0.44084821):

        '''
        given target mass and map resolution, predict CCS. Mass threshold is rescaled using the fitting function c / (1 + exp(-k*(resolution-x0))) + y0.
        
        :param resolution: map resolution in 1/Angstrom
        :param mass: protein mass in kDa
        :param density: protein density in Da/A3
        :param x0: sigmoid parameter
        :param y0: sigmoid parameter
        :param c: sigmoid parameter
        :param k: sigmoid parameter
        :returns CCS estimated from mass and density map resolution
        '''        

        if 'scan' not in list(self.properties):
            raise IOError("no threshold to volume to CCS relationship loaded yet. Please execute thresh_vol_ccs method.")
        
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
        given target mass and map resolution, predict CCS. Mass threshold is rescaled using the fitting function c / (1 + exp(-k*(resolution-x0))) + y0.
    
        :param resolution: map resolution in 1/Angstrom
        :param mass: protein mass in kDa
        :param density: protein density in Da/A3
        :param x0: sigmoid parameter
        :param y0: sigmoid parameter
        :param c: sigmoid parameter
        :param k: sigmoid parameter
        :returns CCS estimated from mass and density map resolution
        '''
        
        if 'scan' not in list(self.properties):
            raise IOError("no threshold to volume to CCS relationship loaded yet. Please execute thresh_vol_ccs method.")
        
        data=self.properties['scan'].copy()
        data[:,1]*=density
        data[:,1]/=1000.0

        #get mass threshold
        dtest1=np.argmin(np.abs(data[:,2]-ccs))
        thresh=data[dtest1,0]
        
        #rescale mass threshold
        scaling= c / (1 + np.exp(-k*(resolution-x0))) + y0
        calibrated_thresh=thresh*scaling
        
        #get associated CCS
        dtest1=np.argmin(np.abs(data[:,0]-calibrated_thresh))
        return data[dtest1,1], data[dtest1,0]


    def scan_threshold(self, mass, density=0.782878356, sampling_points=1000):
        '''
        if mass and density of object are known, filter the map on a linear scale of threshold values, and compare the obtained mass to the experimental one.

        .. note:: in proteins, an average value of 1.3 g/cm^3 (0.782878356 Da/A^3) can be assumed. Alternatively, the relation density=1.410+0.145*exp(-mass(kDa)/13) can be used.

        .. note:: 1 Da/A^3=0.602214120 g/cm^3

        :param mass: target mass in Da
        :param density: target density in Da/A^3
        :param sampling_points: number of measures to perform between min and max intensity in density map
        :returns: array reporting tested values and error on mass ([threshold, model_mass-target_mass])
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

            print("threshold=%s, error=%s" % (thresh, vol * density - mass))
            result.append([thresh, vol * density - mass])

        r = np.array(result)
        bestthresh = r[np.argmin(np.abs(r[:, 1]) - mass), 0]

        try:
            self.place_points(bestthresh)
        except Exception as ex:
            pass

        return r

    def find_data_from_sigma(self, sigma, exact=True, append=False, noise_filter=0.01):
        '''
        map experimental data to given threshold

        :param sigma: density threshold
        :param noise_filter: launch DBSCAN clustering algorithm to detect connected regions in density map. Regions representing less than noise_filter of the total will be removed. This is a ratio, value should be between 0 and 1.        '''

        thresh = self.get_sigma_from_thresh(sigma)

        if exact:
            import biobox as bb

            try:
                self.place_points(thresh, noise_filter)
                vol = self.get_volume()
                ccs = bb.ccs(self)
            except Exception as ex:
                vol = 0
                ccs = 0

            res = np.array([thresh, vol, ccs])

            if append:
                self.properties['scan'] = np.concatenate((self.properties['scan'], res))

            return res

        else:
            return self.properties['scan'][
                np.argmin(np.abs(self.properties['scan'][:, 0] - sigma))]


    def find_data_from_volume(self, vol):
        '''
        map experimental data to given volume

        :param vol: volume
        '''

        return self.properties['scan'][
            np.argmin(np.abs(self.properties['scan'][:, 1] - vol))]

    def find_data_from_ccs(self, ccs):
        '''
        map experimental data to given ccs

        :param ccs: target CCS (in A^2)
        '''

        return self.properties['scan'][
            np.argmin(np.abs(self.properties['scan'][:, 2] - ccs))]

    def threshold_vol_ccs(self, low="", high="", sampling_points=1000, append=False, noise_filter=0.01):
        '''
        return the volume to threshold to CCS relationship

        :param sampling_points: number of measures to perform between min and max intensity in density map
        :returns: array reporting tested values and error on mass ([threshold, model_mass-target_mass])
        '''
        
        import biobox as bb

        if low == "":
            low = self.get_sigma_from_thresh(np.min(self.properties['density']))
        if high == "":
            high = self.get_sigma_from_thresh(np.max(self.properties['density']))

        result = []
        for thresh in np.linspace(low, high, num=sampling_points):
            try:
                self.place_points(thresh, noise_filter=noise_filter)
                vol = self.get_volume()
                ccs = bb.ccs(self)
            except Exception as ex:
                vol = 0
                ccs = 0

            print("thresh: %s, vol=%s, ccs=%s (%s points)" % (thresh, vol, ccs, len(self.points)))
            result.append([thresh, vol, ccs])

        r = np.array(result)

        if append:
            self.properties['scan'] = np.concatenate((self.properties['scan'], r))

        else:
            self.properties['scan'] = r

        return r

    def best_threshold(self, mass, density=0.782878356):
        '''
        If mass and density of object are known, try to filter the map so that the mass is best matched.

        search for best threshold using bissection method.

        .. note:: in proteins, an average value of 1.3 g/cm^3 (0.782878356 Da/A^3) can be assumed. Alternatively, the relation density=1.410+0.145*exp(-mass(kDa)/13) can be used.

        .. note:: 1 Da/A^3=1.660538946 g/cm^3
        .. note:: 1 g/cm^3=0.602214120 Da/A^3

        :param mass: target mass in Da
        :param density: target density in Da/A^3
        :returns: array reporting tested values and error on mass ([sigma, model_mass-target_mass])
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
                if np.abs(high_val) < np.abs(low_val):
                    thresh = high
                else:
                    thresh = low
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
        bestthresh = r[-1, 0]

        try:
            self.place_points(bestthresh)
        except Exception as ex:
            pass

        return r

    def blur(self, dimension=5, sigma=0.5):
        '''
        blur density applying a cubic gaussian kernel of given kernel dimension (cubic grid size).

        .. warning:: cannot be undone

        :param dimension: size of the kernel grid.
        :param sigma: standard deviation of gaussian kernel.
        '''

        shape = (dimension, dimension, dimension)

        m, n, k = [(ss - 1.) / 2. for ss in shape]

        x_ = np.arange(-m, m + 1, 1).astype(int)
        y_ = np.arange(-n, n + 1, 1).astype(int)
        z_ = np.arange(-k, k + 1, 1).astype(int)
        x, y, z = np.meshgrid(x_, y_, z_)

        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh

        dens = scipy.ndimage.filters.convolve(self.properties['density'], h, mode='constant')
        self.properties['density'] = dens
        self.properties["sigma"] = np.std(self.properties['density'])

    def write_dx(self, fname='dens.dx'):
        '''
        Write a density map in DX format

        :param fname: output file name
        '''

        dens = self.properties['density']
        origin = self.properties['origin']
        delta = self.properties['delta']

        fout = open(fname, "w")
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

    def export_as_pdb(self, outname, step, threshold=0.1):
        '''
        Write a pdb file with points where the density exceeds a threshold

        :param outname: output file name
        :param step: stepsize used to generate the density map
        :param threshold: density to be exceeded to generate a point in pdb
        '''

        # @todo assign spheres beta factor to associated density value

        dens = self.properties['density']
        origin = self.properties['origin']

        fout = open(outname, 'w')

        cnt = 1
        identifier = 'ATOM'  # atom to be used to mimick density
        symbol = 'H'  # element for atom

        print('exporting density greater than %s to pdb' % threshold)

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
        import density map and fill up the points and properties data structures.

        :param filename: name of dx file to load
        '''

        try:
            fin = open(filename, "r")
        except Exception as e:
            raise Exception('opening of file %s failed!' % filename)

        d = []
        dlt = []
        for line in fin:
            w = line.split()
            if w[0] == "#":
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


        # scaling factor with respect of unit cell voxels
        self.properties['delta'] = np.array(dlt).astype(float)

        try:
            self.properties['density'] = np.reshape(
                np.array(d).astype(float),
                (self.properties['size'][0],
                 self.properties['size'][1],
                 self.properties['size'][2]))
        except Exception as ex:
            raise Exception("reshaping of dx data failed! Dimensions and dataset size are inconsistent!")

    def _import_mrc(self, filename, fileformat):
        '''
        import density map in MRC, CCP4 or IMOD format.

        MRC format here: www2.mrc-lmb.cam.ac.uk/image2000.html

        CCP4 format here: www.ccp4.ac.uk/html/maplib.html

        :param filename name of MRC or CCP4 file to load
        :param fileformat: can be mrc, imod or ccp4
        '''

        import biobox.classes.density_MRC as MRC
        try:
            [density, data] = MRC.read_density(filename, fileformat)
            self.properties['density'] = density
            self.properties['origin'] = np.array(data.origin)
            self.properties['size'] = np.array(density.shape)
            self.properties['delta'] = np.identity(3) * data.mrc_data.data_step

            # sphere size corresponding to the volume of one voxel
            voxel_volume = self.properties['delta'][0, 0] * self.properties['delta'][1, 1] * self.properties['delta'][2, 2]
            self.properties['radius'] = (voxel_volume * 3 / (4 * np.pi))**(1 / 3.0)

        except Exception as e:
            raise Exception("%s" % e)


    def get_volume(self):
        '''
        compute density map volume. This is done by counting the points, and multiplying that by voxels' volume.

        .. warning:: can be called only after :func:`place_points <density.Density.place_points>` has been called.

        .. warning:: supposes unskewed voxels.
        '''
        return self.properties['delta'][0, 0] * self.properties['delta'][1, 1] * self.properties['delta'][2, 2] * len(self.points)



if __name__ == "__main__":

    import os
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
