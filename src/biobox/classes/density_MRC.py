# Read 'mrc' or 'ccp4' or 'imod' map file format electron microscope data.
# Byte swapping will be done if needed.
#
# NOTE: the code below is a Python 3 translation of CHIMERA files
# chimera/share/VolumeData/mrc, readarray.py and griddata.py (2014 version, in Python 2)

import numpy as np
import os.path
from functools import reduce, cmp_to_key
#from density import Structure

def cmp(a, b):
    '''
    three-way comparison (Python 2 cmp).

    :param a: first value
    :param b: second value
    :returns: 1 if a > b, -1 if a < b, 0 otherwise
    '''
    return (a > b) - (a < b)

# -----------------------------------------------------------------------------
# Maintain a cache of data objects using a limited amount of memory.
# The least recently accessed data is released first.
# -----------------------------------------------------------------------------
#
class Data_Cache:
    '''
    cache of data objects using a limited amount of memory. The least recently accessed data, not referenced elsewhere, is released first.
    '''

    def __init__(self, size):
        '''
        :param size: maximum memory to use, in bytes
        '''
        self.size = size
        self.used = 0
        self.time = 1
        self.data = {}
        self.groups = {}


    # ---------------------------------------------------------------------------
    #
    def cache_data(self, key, value, size, description, groups = []):
        '''
        store a value in the cache (replacing any value having the same key), then release data if the memory limit is exceeded.

        :param key: key identifying the value
        :param value: data to store
        :param size: memory used by the value, in bytes
        :param description: text describing the value
        :param groups: list of groups the value belongs to
        '''

        self.remove_key(key)
        d = Cached_Data(key, value, size, description, self.time_stamp(), groups)
        self.data[key] = d

        for g in groups:
            gtable = self.groups
            if not g in gtable:#.has_key(g):
                gtable[g] = []
            gtable[g].append(d)

        self.used = self.used + size
        self.reduce_use()

    # ---------------------------------------------------------------------------
    #
    def lookup_data(self, key):
        '''
        retrieve a value from the cache and update its access time, then release data if the memory limit is exceeded.

        :param key: key identifying the value
        :returns: cached value, or None if the key is not in the cache
        '''

        data = self.data
        if key in data: #data.has_key(key):
            d = data[key]
            d.last_access = self.time_stamp()
            v = d.value
        else:
            v = None

        self.reduce_use()
        return v

    # ---------------------------------------------------------------------------
    #
    def remove_key(self, key):
        '''
        remove the value having the given key (if present), then release data if the memory limit is exceeded.

        :param key: key identifying the value
        '''

        data = self.data
        if key in data: #.has_key(key):
            self.remove_data(data[key])
        self.reduce_use()

    # ---------------------------------------------------------------------------
    #
    def group_keys_and_data(self, group):
        '''
        :param group: group identifier
        :returns: list of (key, value) pairs of the values belonging to the group (empty list if the group is unknown)
        '''

        groups = self.groups
        if not group in groups: #.has_key(group):
            return []

        kd = [(d.key, d.value) for d in groups[group]]
        return kd

    # ---------------------------------------------------------------------------
    #
    def resize(self, size):
        '''
        change the memory limit, and release data if it is exceeded.

        :param size: maximum memory to use, in bytes
        '''

        self.size = size
        self.reduce_use()

    # ---------------------------------------------------------------------------
    #
    def reduce_use(self):
        '''
        if the memory limit is exceeded, release values not referenced outside the cache, least recently accessed first, until the memory used is within the limit.
        '''

        if self.used <= self.size:
            return

        data = self.data
        dlist = list(data.values())
        dlist.sort(key=cmp_to_key(lambda d1, d2: cmp(d1.last_access, d2.last_access)))
        import sys
        for d in dlist:
            if sys.getrefcount(d.value) == 2:
                self.remove_data(d)
                if self.used <= self.size:
                    break

    # ---------------------------------------------------------------------------
    #
    def remove_data(self, d):
        '''
        remove a cached entry from the cache and from its groups.

        :param d: :class:`Cached_Data <biobox.classes.density_MRC.Cached_Data>` entry to remove
        '''

        del self.data[d.key]
        self.used = self.used - d.size
        d.value = None

        for g in d.groups:
            dlist = self.groups[g]
            dlist.remove(d)
            if len(dlist) == 0:
                del self.groups[g]

    # ---------------------------------------------------------------------------
    #
    def time_stamp(self):
        '''
        :returns: current value of the access counter (incremented at each call)
        '''

        t = self.time
        self.time = t + 1
        return t

###############################################################################

class Cached_Data:
    '''
    entry of a :class:`Data_Cache <biobox.classes.density_MRC.Data_Cache>`.
    '''

    def __init__(self, key, value, size, description, time_stamp, groups):
        '''
        :param key: key identifying the value
        :param value: cached data
        :param size: memory used by the value, in bytes
        :param description: text describing the value
        :param time_stamp: access counter value at last access
        :param groups: list of groups the value belongs to
        '''

        self.key = key
        self.value = value
        self.size = size
        self.description = description
        self.last_access = time_stamp
        self.groups = groups

###############################################################################

class MRC_Grid:
    '''
    grid data read from an MRC, CCP4 or IMOD file, and its placement in space.
    '''

    def __init__(self, path, file_type = 'mrc'):
        '''
        read the file header and set up the grid size, value type, origin, voxel size, cell angles, rotation and index to coordinate transforms.

        :param path: file name (or list or tuple whose first element is the file name)
        :param file_type: mrc (default), ccp4 or imod
        '''

        d = MRC_Data(path, file_type)
        self.mrc_data = d

        # Path, file_type used for reloading data sets.
        self.path = path
        self.file_type = file_type    # 'mrc', 'ccp4', ....

        name = self.name_from_path(path)
        self.name = name

        self.size = tuple(d.data_size)

        if not isinstance(d.element_type, np.dtype):
            d.element_type = np.dtype(d.element_type)
        self.value_type = d.element_type                # numpy dtype.


        # Parameters defining how data matrix is positioned in space
        self.origin = tuple(d.data_origin)
        self.original_origin = self.origin
        self.step = tuple(d.data_step)
        self.original_step = self.step
        self.cell_angles = tuple(d.cell_angles)
        self.rotation = tuple(map(tuple, d.rotation))
        self.symmetries = ()
        self.ijk_to_xyz_transform = None
        self.xyz_to_ijk_transform = None

        #self.rgba = default_color                        # preferred color for displaying data

        global data_cache
        self.data_cache = data_cache

        self.writable = False
        self.change_callbacks = []

        self.update_transform()

    # ---------------------------------------------------------------------------
    #
    def name_from_path(self, path):
        '''
        :param path: file name (or list or tuple whose first element is the file name)
        :returns: base name of the file
        '''

        if isinstance(path, (list,tuple)):
                p = path[0]
        else:
                p = path

        name = os.path.basename(p)
        return name

    # ---------------------------------------------------------------------------
    # Compute 3 by 4 matrices encoding rotation and translation.
    #
    def update_transform(self):
        '''
        compute the 3x4 matrices mapping matrix indices to xyz coordinates and back (from origin, step, cell angles and rotation), and call the change callbacks if they changed.
        '''

        saxes = skew_axes(self.cell_angles)
        rsaxes = [apply_rotation(self.rotation, a) for a in saxes]
        tf, tf_inv = transformation_and_inverse(self.origin, self.step, rsaxes)
        if tf != self.ijk_to_xyz_transform or tf_inv != self.xyz_to_ijk_transform:
            self.ijk_to_xyz_transform = tf
            self.xyz_to_ijk_transform = tf_inv
            self.coordinates_changed()

    # ---------------------------------------------------------------------------
    # A matrix ijk corresponds to a point in xyz space.
    # This function maps the xyz point to the matrix index.
    # The returned matrix index need not be integers.
    #
    def xyz_to_ijk(self, xyz):
        '''
        map an xyz point to the (not necessarily integer) matrix index.

        :param xyz: point coordinates, in Angstrom
        :returns: matrix index (i, j, k), as a tuple of floats
        '''

        return map_point(xyz, self.xyz_to_ijk_transform)

    # ---------------------------------------------------------------------------
    # A matrix ijk corresponds to a point in xyz space.
    # This function maps the matrix index to the xyz point.
    #
    def ijk_to_xyz(self, ijk):
        '''
        map a matrix index to an xyz point.

        :param ijk: matrix index (i, j, k)
        :returns: point coordinates, in Angstrom, as a tuple
        '''

        return map_point(ijk, self.ijk_to_xyz_transform)

    # ---------------------------------------------------------------------------
    #
    def matrix(self, ijk_origin = (0,0,0), ijk_size = None,
                         ijk_step = (1,1,1), progress = None, from_cache_only = False):
        '''
        return a submatrix of the data, from the cache if available, otherwise read from file (and cached).

        :param ijk_origin: index of the first element along x, y and z
        :param ijk_size: number of elements along x, y and z (default: the full grid)
        :param ijk_step: step between elements along x, y and z
        :param progress: progress reporter, or None
        :param from_cache_only: if True, do not read the file when the data is not cached
        :returns: numpy array where element ijk is accessed as m[k,j,i], or None if from_cache_only is True and the data is not cached
        '''

        if ijk_size == None:
            ijk_size = self.size

        m = self.cached_data(ijk_origin, ijk_size, ijk_step)
        if m is None and not from_cache_only:
            m = self.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
            self.cache_data(m, ijk_origin, ijk_size, ijk_step)

        return m

    # ---------------------------------------------------------------------------
    # NumPy matrix.    The returned matrix has size ijk_size and
    # element ijk is accessed as m[k,j,i].    It is an error if the requested
    # submatrix does not lie completely within the full data matrix.    It is
    # also an error for the size to be <= 0 in any dimension.    These invalid
    # inputs might throw an exception or might return garbage.    It is the
    # callers responsibility to make sure the arguments are valid.
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        '''
        read a submatrix from file, see :func:`MRC_Data.read_matrix <biobox.classes.density_MRC.MRC_Data.read_matrix>`.

        :param ijk_origin: index of the first element along x, y and z
        :param ijk_size: number of elements along x, y and z
        :param ijk_step: step between elements along x, y and z
        :param progress: progress reporter, or None
        :returns: numpy array where element ijk is accessed as m[k,j,i]
        '''

        return self.mrc_data.read_matrix(ijk_origin, ijk_size, ijk_step, progress)


    def cached_data(self, origin, size, step):
        '''
        look for the requested submatrix in the cache, either stored as such or contained in a larger cached matrix.

        :param origin: index of the first element along x, y and z
        :param size: number of elements along x, y and z
        :param step: step between elements along x, y and z
        :returns: numpy array where element ijk is accessed as m[k,j,i], or None if not found (or if there is no cache)
        '''

        dcache = self.data_cache
        if dcache is None:
            return None

        key = (self, tuple(origin), tuple(size), tuple(step))
        m = dcache.lookup_data(key)
        if not m is None:
            return m

        # Look for a matrix containing the desired matrix
        group = self
        kd = dcache.group_keys_and_data(group)
        for k, matrix in kd:
            orig, sz, st = k[1:]
            if (step[0] < st[0] or step[1] < st[1] or step[2] < st[2] or
                    step[0] % st[0] or step[1] % st[1] or step[2] % st[2]):
                continue                # Step sizes not compatible
            if (origin[0] < orig[0] or origin[1] < orig[1] or origin[2] < orig[2] or
                    origin[0] + size[0] > orig[0] + sz[0] or
                    origin[1] + size[1] > orig[1] + sz[1] or
                    origin[2] + size[2] > orig[2] + sz[2]):
                continue                # Doesn't cover.
            dstep = [a // b for a, b in zip(step, st)]
            offset = [a - b for a, b in zip(origin, orig)]
            if offset[0] % st[0] or offset[1] % st[1] or offset[2] % st[2]:
                continue                # Offset stagger.
            moffset = [o // s for o, s in zip(offset, st)]
            msize = [(s + t - 1) // t for s, t in zip(size, st)]
            m = matrix[moffset[2]:moffset[2]+msize[2]:dstep[2],
                                 moffset[1]:moffset[1]+msize[1]:dstep[1],
                                 moffset[0]:moffset[0]+msize[0]:dstep[0]]
            dcache.lookup_data(key) # update access time
            return m

        return None

    # ---------------------------------------------------------------------------
    #
    def cache_data(self, m, origin, size, step):
        '''
        store a submatrix in the cache (if there is one).

        :param m: numpy array to store
        :param origin: index of the first element along x, y and z
        :param size: number of elements along x, y and z
        :param step: step between elements along x, y and z
        '''

        dcache = self.data_cache
        if dcache is None:
            return

        key = (self, tuple(origin), tuple(size), tuple(step))
        elements = reduce(lambda a,b: a*b, m.shape, 1)
        bytes1 = elements * m.itemsize
        groups = [self]
        descrip = self.data_description(origin, size, step)
        dcache.cache_data(key, m, bytes1, descrip, groups)

    # ---------------------------------------------------------------------------
    #
    def data_description(self, origin, size, step):
        '''
        :param origin: index of the first element along x, y and z
        :param size: number of elements along x, y and z
        :param step: step between elements along x, y and z
        :returns: text describing a submatrix: grid name, index bounds and step (if not 1)
        '''

        description = self.name

        if origin == (0,0,0):
            bounds = ' (%d,%d,%d)' % tuple(size)
        else:
            region = (origin[0], origin[0]+size[0]-1,
                                origin[1], origin[1]+size[1]-1,
                                origin[2], origin[2]+size[2]-1)
            bounds = ' (%d-%d,%d-%d,%d-%d)' % region
        description += bounds

        if step != (1,1,1):
            description += ' step (%d,%d,%d)' % tuple(step)

        return description

    # ---------------------------------------------------------------------------
    #
    def clear_cache(self):
        '''
        remove all cached submatrices of this grid.
        '''

        dcache = self.data_cache
        if dcache is None:
            return

        for k,d in dcache.group_keys_and_data(self):
            dcache.remove_key(k)

    # ---------------------------------------------------------------------------
    #
    def add_change_callback(self, cb):
        '''
        :param cb: function to call, with a reason string, when values or coordinates change
        '''

        self.change_callbacks.append(cb)

    # ---------------------------------------------------------------------------
    #
    def remove_change_callback(self, cb):
        '''
        :param cb: callback function to remove
        '''

        self.change_callbacks.remove(cb)

    # ---------------------------------------------------------------------------
    # Code has modified matrix elements, or the value type has changed.
    #
    def values_changed(self):
        '''
        call the change callbacks with reason 'values changed'.
        '''

        self.call_callbacks('values changed')

    # ---------------------------------------------------------------------------
    # Mapping of array indices to xyz coordinates has changed.
    #
    def coordinates_changed(self):
        '''
        call the change callbacks with reason 'coordinates changed'.
        '''

        self.call_callbacks('coordinates changed')

    # ---------------------------------------------------------------------------
    #
    def call_callbacks(self, reason):
        '''
        :param reason: string passed to every change callback
        '''

        for cb in self.change_callbacks:
            cb(reason)

###############################################################################

class MRC_Data:
    '''
    header information of an MRC, CCP4 or IMOD file, and access to its data.
    '''

    def __init__(self, path, file_type):
        '''
        read the file header, detecting byte order, and derive value type, axes order, size, voxel size, cell angles, origin and rotation.
        Raises SyntaxError if the header is invalid or the file is too small, and Exception if cell angles are invalid.

        :param path: file name
        :param file_type: mrc, ccp4 or imod
        '''


        self.path = path
        self.name = os.path.basename(path)

        file1 = open(path, 'rb')
        try:

            file1.seek(0,2)                                                            # go to end of file
            file_size = file1.tell()
            file1.seek(0,0)                                                          # go to beginning of file

            # Infer file byte order from column axis size nc.    Requires nc < 2**16
            # Was using mode value but 0 is allowed and does not determine byte order.
            self.swap_bytes = 0
            nc = self.read_values(file1, np.int32, 1)
            self.swap_bytes = not (nc > 0 and nc < 65536)
            file1.seek(0,0)

            v = self.read_header_values(file1, file_size, file_type)

            if v.get('imodStamp') == 1146047817:
                unsigned_8_bit = (v['imodFlags'] & 0x1 == 0)
            else:
                unsigned_8_bit = (file_type == 'imod' or v['type'] == 'mrc')
            self.element_type = self.value_type(v['mode'], unsigned_8_bit)

            self.check_header_values(v, file_size, file1)
            self.header = v                         # For dumpmrc.py standalone program.

            self.data_offset = file1.tell()
        finally:
            file1.close()

        # Axes permutation.
        # Names c,r,s refer to fast, medium, slow file matrix axes.
        # Names i,j,k refer to x,y,z spatial axes.
        mapc, mapr, maps = v['mapc'], v['mapr'], v['maps']
        if (1 in (mapc, mapr, maps) and
                2 in (mapc, mapr, maps) and
                3 in (mapc, mapr, maps)):
            crs_to_ijk = (mapc-1,mapr-1,maps-1)
            ijk_to_crs = [None,None,None]
            for a in range(3):
                ijk_to_crs[crs_to_ijk[a]] = a
        else:
            crs_to_ijk = ijk_to_crs = (0, 1, 2)
        self.crs_to_ijk = crs_to_ijk
        self.ijk_to_crs = ijk_to_crs

        crs_size = v['nc'], v['nr'], v['ns']
        self.matrix_size = [int(s) for s in crs_size]
        self.data_size = [int(crs_size[a]) for a in ijk_to_crs]

        mx, my, mz = v['mx'], v['my'], v['mz']
        xlen, ylen, zlen = v['xlen'], v['ylen'], v['zlen']
        if mx > 0 and my > 0 and mz > 0 and xlen > 0 and ylen > 0 and zlen > 0:
            self.data_step = (xlen/mx, ylen/my, zlen/mz)
        else:
            self.data_step = (1.0, 1.0, 1.0)

        alpha, beta, gamma = (v['alpha'], v['beta'], v['gamma'])
        if not valid_cell_angles(alpha, beta, gamma, path):
            alpha = beta = gamma = 90
        self.cell_angles = (alpha, beta, gamma)

        if (v['type'] == 'mrc2000' and
                (v['zorigin'] != 0 or v['xorigin'] != 0 or v['yorigin'] != 0)):
            #
            # This is a new MRC 2000 format file.    The xyz origin header parameters
            # are used instead of using ncstart, nrstart nsstart for new style files,
            # provided the xyz origin specified is not zero.    It turns out the
            # xorigin, yorigin, zorigin values are zero in alot of new files while
            # the ncstart, nrstart, nsstart give the correct (non-zero) origin. So in
            # cases where the xyz origin parameters and older nrstart, ncstart,
            # nsstart parameters specify different origins the one that is non-zero
            # is preferred.    And if both are non-zero, the newer xorigin, yorigin,
            # zorigin are used.
            #
            self.data_origin = (v['xorigin'], v['yorigin'], v['zorigin'])
        else:
            crs_start = v['ncstart'], v['nrstart'], v['nsstart']
            ijk_start = [crs_start[a] for a in ijk_to_crs]
            # Check if ijk_start values appear to be uninitialized.
            limit = 10*max(max(mx,my,mz), max(self.data_size))
            if [s for s in ijk_start if abs(s) > limit]:
                self.data_origin = (0., 0., 0.)
            else:
                self.data_origin = scale_and_skew(ijk_start, self.data_step, self.cell_angles)

        r = ((1,0,0),(0,1,0),(0,0,1))
        for lbl in v['labels']:
            if lbl.startswith(b'Chimera rotation: '):
                ax,ay,az,angle = map(float, lbl.rstrip(b'\0').split()[2:])

                #S = Structure()
                r = self.rotation_matrix(np.array([ax, ay, az]), np.radians(angle))
                #r = Matrix.rotation_from_axis_angle((ax,ay,az), angle)

        self.rotation = r

        self.min_intensity = v['amin']
        self.max_intensity = v['amax']



    def rotation_matrix(self, axis, theta):
        '''
        compute matrix needed to rotate the system around an arbitrary axis (using Euler-Rodrigues formula).

        :param axis: 3d vector (numpy array), representing the axis around which to rotate
        :param theta: desired rotation angle, in radians
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




    # Format derived from C header file mrc.h.
    #
    def read_header_values(self, file1, file_size, file_type):
        '''
        read the header (format derived from C header file mrc.h), leaving the file positioned after the symmetry operators.
        Raises SyntaxError if the size of the symmetry operators block is invalid.

        :param file1: file object open in binary mode, positioned at the beginning of the file
        :param file_size: file size, in bytes
        :param file_type: mrc, ccp4 or imod (ccp4 reads the CCP4 header layout, others the MRC one)
        :returns: dictionary of header values
        '''

        MRC_USER = 29
        CCP4_USER = 15
        MRC_NUM_LABELS = 10
        MRC_LABEL_SIZE = 80
        MRC_HEADER_LENGTH = 1024

        i32 = np.int32
        f32 = np.float32

        v = {}
        v['nc'], v['nr'], v['ns'] = self.read_values(file1, i32, 3)
        v['mode'] = self.read_values(file1, i32, 1)
        v['ncstart'], v['nrstart'], v['nsstart'] = self.read_values(file1, i32, 3)
        v['mx'], v['my'], v['mz'] = self.read_values(file1, i32, 3)
        v['xlen'], v['ylen'], v['zlen'] = self.read_values(file1, f32, 3)
        v['alpha'], v['beta'], v['gamma'] = self.read_values(file1, f32, 3)
        v['mapc'], v['mapr'], v['maps'] = self.read_values(file1, i32, 3)
        v['amin'], v['amax'], v['amean'] = self.read_values(file1, f32, 3)
        v['ispg'], v['nsymbt'] = self.read_values(file1, i32, 2)
        if file_type == 'ccp4':
            v['lskflg'] = self.read_values(file1, i32, 1)
            v['skwmat'] = self.read_values(file1, f32, 9)
            v['skwtrn'] = self.read_values(file1, f32, 3)
            v['user'] = self.read_values(file1, i32, CCP4_USER)
            v['map'] = file1.read(4)     # Should be 'MAP '.
            v['machst'] = self.read_values(file1, i32, 1)
            v['rms'] = self.read_values(file1, f32, 1)
            v['type'] = 'ccp4'
        else:
            # MRC file
            user = file1.read(4*MRC_USER)
            if user[-4:] == b'MAP ':
                # New style MRC 2000 format file with xyz origin
                v['user'] = self.read_values_from_string(user, i32, MRC_USER)[:-4]
                xyz_origin = self.read_values_from_string(user[-16:-4], f32, 3)
                v['xorigin'], v['yorigin'], v['zorigin'] = xyz_origin
                v['imodStamp'] = self.read_values_from_string(user[56:60], i32, 1)
                v['imodFlags'] = self.read_values_from_string(user[60:64], i32, 1)
                v['machst'] = self.read_values(file1, i32, 1)
                v['rms'] = self.read_values(file1, f32, 1)
                v['type'] = 'mrc2000'
            else:
                # Old style MRC has xy origin instead of machst and rms.
                v['user'] = self.read_values_from_string(user, i32, MRC_USER)
                v['xorigin'], v['yorigin'] = self.read_values(file1, f32, 2)
                v['type'] = 'mrc'

        v['nlabl'] = self.read_values(file1, i32, 1)
        labels = []
        for i in range(MRC_NUM_LABELS):
            labels.append(file1.read(MRC_LABEL_SIZE))
        v['labels'] = labels

        # Catch incorrect nsymbt value.
        if v['nsymbt'] < 0 or v['nsymbt'] + MRC_HEADER_LENGTH > file_size:
            raise SyntaxError('MRC header value nsymbt (%d) is invalid'
                                                    % v['nsymbt'])
        v['symop'] = file1.read(v['nsymbt'])

        return v

    #
    def value_type(self, mode, unsigned_8_bit):
        '''
        :param mode: MRC data mode (0: 8 bit integers, 1: 16 bit integers, 2: 32 bit floats, 6: unsigned 16 bit integers)
        :param unsigned_8_bit: if True, mode 0 is read as unsigned
        :returns: numpy dtype of data values. Raises SyntaxError for other modes.
        '''

        MODE_char     = 0
        MODE_short    = 1
        MODE_float    = 2
        MODE_ushort   = 6

        if mode == MODE_char:
            if unsigned_8_bit:
                t = np.dtype(np.uint8)
            else:
                t = np.dtype(np.int8)                # CCP4 or MRC2000
        elif mode == MODE_short:
            t = np.dtype(np.int16)
        elif mode == MODE_float:
            t = np.dtype(np.float32)
        elif mode == MODE_ushort:
            t = np.dtype(np.uint16)
        else:
            raise SyntaxError('MRC data value type (%d) ' % mode +
                                                    'is not 8 or 16 bit integers or 32 bit floats')

        return t

    #
    def check_header_values(self, v, file_size, file1):
        '''
        check that grid size is positive and that the file is large enough to contain the data.
        If the header declares symmetry operators that are missing, the file is moved back to the data start and v['symop'] is emptied.
        Raises SyntaxError otherwise.

        :param v: dictionary of header values
        :param file_size: file size, in bytes
        :param file1: file object, positioned at the end of the header
        '''

        if v['nc'] <= 0 or v['nr'] <= 0 or v['ns'] <= 0:
            raise SyntaxError('Bad MRC grid size (%d,%d,%d)'
                                                    % (v['nc'],v['nr'],v['ns']))

        esize = self.element_type.itemsize
        data_size = int(v['nc']) * int(v['nr']) * int(v['ns']) * esize
        header_end = file1.tell()
        if header_end + data_size > file_size:
            if v['nsymbt'] and (header_end - v['nsymbt']) + data_size == file_size:
                # Sometimes header indicates symmetry operators are present but
                # they are not.    This error occurs in macromolecular structure database
                # entries emd_1042.map, emd_1048.map, emd_1089.map, ....
                # This work around code allows the incorrect files to be read.
                file1.seek(-v['nsymbt'], 1)
                v['symop'] = ''
            else:
                msg = ('File size %d too small for grid size (%d,%d,%d)'
                             % (file_size, v['nc'],v['nr'],v['ns']))
                if v['nsymbt']:
                    msg += ' and %d bytes of symmetry operators' % (v['nsymbt'],)
                raise SyntaxError(msg)

    #
    def read_values(self, file1, etype, count):
        '''
        read binary values from file, swapping bytes if needed. Raises SyntaxError if the file is truncated.

        :param file1: file object open in binary mode
        :param etype: numpy type of the values
        :param count: number of values to read
        :returns: a single value if count is 1, otherwise a numpy array
        '''

        esize = np.array((), etype).itemsize
        string = file1.read(esize * count)
        if len(string) < esize * count:
            raise SyntaxError('MRC file is truncated.    Failed reading %d values, type %s' % (count, etype.__name__))
        values = self.read_values_from_string(string, etype, count)
        return values

    #
    def read_values_from_string(self, string, etype, count):
        '''
        convert bytes into values, swapping bytes if needed.

        :param string: bytes to convert
        :param etype: numpy type of the values
        :param count: if 1, return only the first value
        :returns: a single value if count is 1, otherwise a numpy array
        '''

        values = np.frombuffer(string, etype)
        if self.swap_bytes:
            values = values.byteswap()
        if count == 1:
            return values[0]
        return values

    # Reads a submatrix from a the file.
    # Returns 3d numpy matrix with zyx index order.
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        '''
        read a submatrix from the file, accounting for the file axes order.

        :param ijk_origin: index of the first element along x, y and z
        :param ijk_size: number of elements along x, y and z
        :param ijk_step: step between elements along x, y and z
        :param progress: progress reporter, or None
        :returns: numpy array with zyx index order
        '''

        # ijk correspond to xyz.    crs refers to fast,medium,slow matrix file axes.
        crs_origin = [ijk_origin[a] for a in self.crs_to_ijk]
        crs_size = [ijk_size[a] for a in self.crs_to_ijk]
        crs_step = [ijk_step[a] for a in self.crs_to_ijk]

        matrix = read_array(self.path, self.data_offset,
                                                crs_origin, crs_size, crs_step,
                                                self.matrix_size, self.element_type, self.swap_bytes,
                                                progress)
        if not matrix is None:
            matrix = self.permute_matrix_to_xyz_axis_order(matrix)

        return matrix

    #
    def permute_matrix_to_xyz_axis_order(self, matrix):
        '''
        :param matrix: numpy array with indices in file axes order (slow, medium, fast)
        :returns: numpy array with zyx index order
        '''

        if self.ijk_to_crs == (0,1,2):
            return matrix

        kji_to_src = [2-self.ijk_to_crs[2-a] for a in (0,1,2)]
        m = matrix.transpose(kji_to_src)

        return m
###############################################################################
###############################################################################

def valid_cell_angles(alpha, beta, gamma, path):
    '''
    check cell angles. Raises Exception if any angle is not between 0 and 180, if their sum is not less than 360, or if the largest is not less than the sum of the other two.

    :param alpha: alpha angle, in degrees
    :param beta: beta angle, in degrees
    :param gamma: gamma angle, in degrees
    :param path: file name, used in the error message
    :returns: True if angles are valid
    '''

    err = None

    for a in (alpha, beta, gamma):
        if a <= 0 or a >= 180:
            err = 'must be between 0 and 180'

    if alpha + beta + gamma >= 360 and err is None:
        err = 'sum must be less than 360'

    if max((alpha, beta, gamma)) >= 0.5 * (alpha + beta + gamma) and err is None:
        err = 'largest angle must be less than sum of other two'

    if err:
        raise ValueError('%s: invalid cell angles %.5g,%.5g,%.5g %s.\n'
                                 % (path, alpha, beta, gamma, err))
        return False

    return True

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Return 3 by 4 matrix where first 3 columns give rotation and last column
# is translation.
#
def transformation_and_inverse(origin, step, axes):
    '''
    :param origin: xyz coordinates of the grid origin, in Angstrom
    :param step: voxel size along each axis, in Angstrom
    :param axes: unit vectors of the three grid axes
    :returns: 3x4 matrix mapping indices to xyz coordinates (first 3 columns give rotation and scaling, last column the translation), as nested tuples
    :returns: 3x4 matrix of the inverse transform, as nested tuples
    '''

    ox, oy, oz = origin
    d0, d1, d2 = step
    ax, ay, az = axes

    tf = ((d0*ax[0], d1*ay[0], d2*az[0], ox),
                (d0*ax[1], d1*ay[1], d2*az[1], oy),
                (d0*ax[2], d1*ay[2], d2*az[2], oz))

    full = np.vstack([np.array(tf, dtype=float), [0.0, 0.0, 0.0, 1.0]])
    tf_inv = np.linalg.inv(full)[:3]

    # Replace array by tuples
    tf_inv = tuple(map(tuple, tf_inv))

    return tf, tf_inv

# -----------------------------------------------------------------------------
# Apply scaling and skewing transformations.
#
def skew_axes(cell_angles):
    '''
    unit vectors along the cell axes a, b and c, for the given cell angles (in degrees), with a along x and b in the xy plane.

    :param cell_angles: alpha, beta and gamma angles, in degrees
    :returns: three unit vectors, as a tuple of 3-tuples
    '''
    alpha, beta, gamma = map(lambda a: a * np.pi / 180, cell_angles)

    cg = np.cos(gamma)
    sg = np.sin(gamma)
    cb = np.cos(beta)
    ca = np.cos(alpha)
    c1 = (ca - cb*cg)/sg
    c2 = np.sqrt(1 - cb*cb - c1*c1)

    return ((1.0, 0.0, 0.0), (cg, sg, 0.0), (cb, c1, c2))

# -----------------------------------------------------------------------------
#
def scale_and_skew(ijk, step, cell_angles):
    '''
    convert a grid index into xyz coordinates, applying voxel size and cell skewing.

    :param ijk: grid index (i, j, k)
    :param step: voxel size along each axis, in Angstrom
    :param cell_angles: alpha, beta and gamma angles, in degrees
    :returns: xyz coordinates, as a tuple
    '''

    # Convert to radians
    alpha, beta, gamma = map(lambda a: a * np.pi / 180, cell_angles)

    cg = np.cos(gamma)
    sg = np.sin(gamma)
    cb = np.cos(beta)
    ca = np.cos(alpha)
    c1 = (ca - cb*cg)/sg
    c2 = np.sqrt(1 - cb*cb - c1*c1)

    i, j, k = ijk
    d0, d1, d2 = step

    xyz = (d0*i + d1*cg*j + d2*cb*k, d1*sg*j + d2*c1*k, d2*c2*k)
    return xyz


# -----------------------------------------------------------------------------
#
def apply_rotation(r, v):
    '''
    :param r: 3x3 rotation matrix
    :param v: 3D vector
    :returns: rotated vector, as a tuple
    '''

    rv = [r[a][0]*v[0] + r[a][1]*v[1] + r[a][2]*v[2] for a in (0,1,2)]
    return tuple(rv)

# -----------------------------------------------------------------------------
#
def map_point(p, tf):
    '''
    :param p: 3D point
    :param tf: 3x4 transformation matrix (rotation in the first 3 columns, translation in the last)
    :returns: transformed point, as a tuple
    '''

    tfp = [0,0,0]
    for r in range(3):
        tfr = tf[r]
        tfp[r] = tfr[0]*p[0] +tfr[1]*p[1] + tfr[2]*p[2] + tfr[3]
    tfp = tuple(tfp)
    return tfp

# -----------------------------------------------------------------------------

def read_array(path, byte_offset, ijk_origin, ijk_size, ijk_step,
                             full_size, type1, byte_swap, progress = None):
        '''
        read a submatrix from a binary file, where the first axis varies fastest.

        :param path: file name
        :param byte_offset: position of the data start in the file, in bytes
        :param ijk_origin: index of the first element along each file axis
        :param ijk_size: number of elements along each file axis
        :param ijk_step: step between elements along each file axis
        :param full_size: size of the full data along each file axis
        :param type1: numpy type of the values
        :param byte_swap: if True, swap bytes after reading
        :param progress: progress reporter, or None
        :returns: numpy array with indices in reverse order with respect to the file axes
        '''

        if (tuple(ijk_origin) == (0,0,0) and
                tuple(ijk_size) == tuple(full_size) and
                tuple(ijk_step) == (1,1,1)):
                m = read_full_array(path, byte_offset, full_size,
                                                        type1, byte_swap, progress)
                return m

        matrix = allocate_array(ijk_size, type1, ijk_step, progress)

        file1 = open(path, 'rb')

        if progress:
                progress.close_on_cancel(file1)

        # Seek in file to read needed 1d slices.
        io, jo, ko = ijk_origin
        isize, jsize, ksize = ijk_size
        istep, jstep, kstep = ijk_step
        element_size = matrix.itemsize
        jbytes = full_size[0] * element_size
        kbytes = full_size[1] * jbytes
        ibytes = isize * element_size
        ioffset = io * element_size
        from numpy import frombuffer
        for k in range(ko, ko+ksize, kstep):
            if progress:
                progress.plane((k-ko)/kstep)
            kbase = byte_offset + k * kbytes
            for j in range(jo, jo+jsize, jstep):
                offset = kbase + j * jbytes + ioffset
                file1.seek(offset)
                data = file1.read(ibytes)
                slice1 = frombuffer(data, type1)
                matrix[int((k-ko)/kstep), int((j-jo)/jstep), :] = slice1[::istep]

        file1.close()

        if byte_swap:
            matrix.byteswap(True)

        return matrix

# -----------------------------------------------------------------------------
# Read an array from a binary file making at most one copy of array in memory.
#
def read_full_array(path, byte_offset, size, type1, byte_swap,
                                        progress = None, block_size = 2**20):
        '''
        read a full array from a binary file making at most one copy of array in memory.

        :param path: file name
        :param byte_offset: position of the data start in the file, in bytes
        :param size: size of the data along each file axis
        :param type1: numpy type of the values
        :param byte_swap: if True, swap bytes after reading
        :param progress: progress reporter, or None
        :param block_size: number of elements read at a time when a progress reporter is given
        :returns: numpy array with indices in reverse order with respect to the file axes
        '''

        a = allocate_array(size, type1)

        file1 = open(path, 'rb')
        file1.seek(byte_offset)

        if progress:
                progress.close_on_cancel(file1)
                a_1d = a.ravel()
                n = len(a_1d)
                nf = float(n)
                for s in range(0,n,block_size):
                        b = a_1d[s:s+block_size]
                        file1.readinto(b)
                        progress.fraction(s/nf)
                progress.done()
        else:
                file1.readinto(a)

        file1.close()

        if byte_swap:
                a.byteswap(True)

        return a

# -----------------------------------------------------------------------------
#
def allocate_array(size, value_type = np.float32, step = None, progress = None,
                                     reverse_indices = True, zero_fill = False):
        '''
        allocate an array able to hold data of the given size, subsampled with the given step.

        :param size: number of elements along each axis
        :param value_type: numpy type of the values
        :param step: step between elements along each axis (default: 1). Each axis gets 1 + (size - 1) // step elements.
        :param progress: not used
        :param reverse_indices: if True, the array shape is in reverse axes order
        :param zero_fill: if True, the array is filled with zeros, otherwise it is not initialised
        :returns: numpy array
        '''




        if step is None:
                msize = size
        else:
                msize = [1+(sz-1)/st for sz,st in zip(size, step)]

        msize = np.array(msize).astype(int)

        shape = list(msize)
        if reverse_indices:
                shape.reverse()

        if zero_fill:
                from numpy import zeros as alloc
        else:
                from numpy import empty as alloc

        m = alloc(shape, value_type)
        return m

# -----------------------------------------------------------------------------
#
def closest_mrc2000_type(type1):
        '''
        :param type1: numpy type of the data values
        :returns: numpy type to convert the data to (float32, int16 or int8). Raises TypeError if the type is not recognised.
        '''

        if type1 in (np.float32, np.float64, np.int32, np.uint32, np.uint, np.uint16):
                ctype = np.float32
        elif type1 in (np.int16, np.uint8):
                ctype = np.int16
        elif type1 in (np.int8, np.intp, np.character):
                ctype = np.int8
        else:
                raise TypeError('Volume data has unrecognized type %s' % type1)

        return ctype

# -----------------------------------------------------------------------------
#can read 'mrc' or 'ccp4' or 'imod'
def read_density(filename,extension):
    '''
    read a density map in MRC, CCP4 or IMOD format, one z section at a time.

    :param filename: name of the file to read
    :param extension: file format: mrc, ccp4 or imod
    :returns: density map as a float64 numpy array, indexed as [x, y, z]
    :returns: :class:`MRC_Grid <biobox.classes.density_MRC.MRC_Grid>` object, holding header information (e.g. origin and voxel size)
    '''

    try:
        grid_data= MRC_Grid(filename, extension)
    except Exception as e:
        raise ValueError('cannot load density map %s: %s'%(filename, e)) from e

    mtype = grid_data.value_type.type
    type1 = closest_mrc2000_type(mtype)

    isz, jsz, ksz = grid_data.size
    m=[]
    for k in range(ksz):
        matrix = grid_data.matrix((0,0,k), (isz,jsz,1))
        if type1 != mtype:
            matrix = matrix.astype(type1)

        m.append(matrix)

    # every section has shape (1, ny, nx), so the stack is (nz, ny, nx)
    data = np.concatenate(m, axis=0).astype(float)

    #return data, grid_data
    return np.swapaxes(data,0,2), grid_data


data_cache = Data_Cache(size = 0)

if __name__=="__main__":

    import os
    filename = "..%stest%sEMD-1080.mrc"%(os.sep, os.sep)

    #try:
    [density,data] = read_density(filename, 'mrc')
    print("origin: %s"%np.array(data.origin))
    print("shape: %s"%np.array(density.shape))
    print("delta: %s"%(np.identity(3)*np.array(data.mrc_data.data_step)))

    #except Exception as e:
    #    print("%s"%e)
