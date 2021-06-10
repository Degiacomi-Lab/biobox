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

'''
Functions to measure characteristics of any BiobOx object
'''

import subprocess
import os
import sys
import random
import string

import numpy as np
from ctypes import cdll, c_int, c_float, byref

import biobox.lib.fastmath as FM  # cython routines


def sasa_c(M, targets=[], probe=1.4, n_sphere_point=960, threshold=0.05):
    '''
    compute the accessible surface area using the Shrake-Rupley algorithm ("rolling ball method")

    :param M: any biobox object
    :param targets: indices to be used for surface estimation. By default, all indices are kept into account.
    :param probe: radius of the "rolling ball"
    :param n_sphere_point: number of mesh points per atom
    :param threshold: fraction of points in sphere, above which structure points are considered as exposed
    :returns: accessible surface area in A^2
    :returns: mesh numpy array containing the found points forming the accessible surface mesh
    :returns: IDs of surface points
    '''

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()

    elif this_inst in ["Assembly", "Polyhedra"]:
        M = M.make_structure()

    # getting radii associated to every atom
    radii = M.data['radius'].values

    if threshold < 0.0 or threshold > 1.0:
        raise Exception("ERROR: threshold should be a floating point between 0 and 1!")

    if len(targets) == 0:
        return FM.c_get_surface(M.points, radii, probe, n_sphere_point, threshold)
    else:
        return FM.c_get_surface(M.points[targets], radii, probe, n_sphere_point, threshold)

def sasa(M, targets=[], probe=1.4, n_sphere_point=960, threshold=0.05):
    '''
    compute the accessible surface area using the Shrake-Rupley algorithm ("rolling ball method")

    :param M: any biobox object
    :param targets: indices to be used for surface estimation. By default, all indices are kept into account.
    :param probe: radius of the "rolling ball"
    :param n_sphere_point: number of mesh points per atom
    :param threshold: fraction of points in sphere, above which structure points are considered as exposed
    :returns: accessible surface area in A^2
    :returns: mesh numpy array containing the found points forming the accessible surface mesh
    :returns: IDs of surface points
    '''

    import biobox.measures.interaction as I

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()

    elif this_inst in ["Assembly", "Polyhedra"]:
        M = M.make_structure()

    if len(targets) == 0:
        targets = range(0, len(M.points), 1)

    # getting radii associated to every atom
    radii = M.data['radius'].values

    if threshold < 0.0 or threshold > 1.0:
        raise Exception("ERROR: threshold should be a floating point between 0 and 1!")

    # create unit sphere points cloud (using golden spiral)
    pts = []
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2 / float(n_sphere_point)
    for k in range(int(n_sphere_point)):
        y = k * offset - 1 + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        pts.append([np.cos(phi) * r, y, np.sin(phi) * r])

    sphere_points = np.array(pts)
    const = 4.0 * np.pi / len(sphere_points)

    contact_map = I.distance_matrix(M.points, M.points)

    asa = 0.0
    surface_atoms = []
    mesh_pts = []
    # compute accessible surface for every atom
    for i in targets:

        # place mesh points around atom of choice
        mesh = sphere_points * (radii[i] + probe) + M.points[i]

        # compute distance matrix between mesh points and neighboring atoms
        test = np.where(contact_map[i, :] < radii.max() + probe * 2)[0]
        neigh = M.points[test]
        dist = I.distance_matrix(neigh, mesh) - radii[test][:, np.newaxis]

        # lines=atoms, columns=mesh points. Count columns containing values greater than probe*2
        # i.e. allowing sufficient space for a probe to fit completely
        cnt = 0
        for m in range(dist.shape[1]):
            if not np.any(dist[:, m] < probe):
                cnt += 1
                mesh_pts.append(mesh[m])

        # calculate asa for current atom, if a sufficient amount of mesh
        # points is exposed (NOTE: to verify)
        if cnt > n_sphere_point * threshold:
            surface_atoms.append(i)
            asa += const * cnt * (radii[i] + probe)**2

    return asa, np.array(mesh_pts), np.array(surface_atoms)

def rgyr(M):
    '''
    compute radius of gyration.
    
    :param M: any biobox object
    :returns: radius of gyration
    '''

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()

    elif this_inst in ["Assembly", "Polyhedra"]:
        M = M.make_structure()
    
    d_square = np.sum((M.points - M.get_center())**2, axis=1)
    return np.sqrt(np.sum(d_square) / d_square.shape[0])


def saxs(M, crysol_path='', crysol_options="-lm 20 -ns 500", pdbname=""):
    '''
    compute SAXS curve using crysol (from ATSAS suite)

    :param M: any biobox object
    :param crysol_path: path to crysol executable. If not provided, the environment variable ATSASPATH is sought instead. This allows redirecting to a specific ATSAS bin folder.
    :param crysol_options: flags to be passes to impact executable
    :param pdbname: if a file has been already written, crysol can be asked to analyze it
    :returns: SAXS curve (nx2 numpy array)
    '''

    if crysol_path == '':
        try:
            crysol_path = os.environ['ATSASPATH']
        except KeyError:
            raise Exception("ATSASPATH environment variable undefined")

    if pdbname == "":
        # write temporary pdb file of current structure on which to launch
        # SAXS calculation
        pdbname = "%s.pdb" % random_string(32)
        while os.path.exists(pdbname):
            pdbname = "%s.pdb" % random_string(32)

        M.write_pdb(pdbname, [M.current])

    else:
        # if file was already provided, verify its existence first!
        if os.path.isfile(pdbname) != 1:
            raise Exception("ERROR: %s not found!" % pdbname)

    # get basename for output
    outfile = os.path.basename(pdbname).split('.')[0]

    call_line = os.path.join(crysol_path, "crysol")
    #try:
    subprocess.check_call('%s %s %s > /dev/null' %(call_line, crysol_options, pdbname), shell=True)
    #except Exception as e:
    #    raise Exception("ERROR: crysol calculation failed!")

    data = np.loadtxt("%s00.int" % outfile, skiprows=1)
    try:
        os.remove("%s00.alm" % outfile)
        os.remove("%s00.int" % outfile)
        os.remove("%s00.log" % outfile)
        os.remove("%s.pdb" % outfile)
    except Exception as ex:
        pass

    return data[:, 0:2]


def ccs(M, use_lib=True, impact_path='', impact_options="-Octree -nRuns 32 -cMode sem -convergence 0.01", pdbname="", tjm_scale=False, proberad=1.0):
    '''
    compute CCS calling either impact.

    :param M: any biobox object
    :param use_lib: if true, impact library will be used, if false a system call to impact executable will be performed instead
    :param impact_path: by default, the environment variable IMPACTPATH is sought. This allows redirecting to a specific impact root folder. 
    :param impact_options: flags to be passes to impact executable
    :param pdbname: if a file has been already written, impact can be asked to analyze it
    :param tjm_scale: if True, CCS value calculated with PA method is scaled to better match trajectory method.
    :param proberad: radius of probe. Do find out if your impact library already adds this value by default or not (old ones do)!
    :returns: CCS value in A^2. Error return: -1 = input filename not found, -2 = unknown code for CCS calculation\n
              -3 CCS calculator failed, -4 = parsing of CCS calculation results failed
    '''

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()
        M.assign_atomtype()
        M.get_atoms_ccs()

    elif this_inst in ["Assembly", "Polyhedra"]:
        M = M.make_structure()

    elif this_inst == "Molecule" and "atom_ccs" not in M.data.columns:
        M.assign_atomtype()
        M.get_atoms_ccs()

    if use_lib and pdbname == "":

        #if True:
        from biobox.measures.calculators import CCS
        try:
            if impact_path == '':

                try:
                    impact_path = os.path.join(os.environ['IMPACTPATH'], "lib")
                except KeyError:
                    raise Exception("IMPACTPATH environment variable undefined")

            if "win" in sys.platform:
                libfile = os.path.join(impact_path, "libimpact.dll")
            else:
                libfile = os.path.join(impact_path, "libimpact.so")

            C = CCS(libfile=libfile)

        except Exception as e:
            raise Exception(str(e))

        if "atom_ccs" in M.data.columns:
            radii = M.data['atom_ccs'].values + proberad
        else:
            radii = M.data['radius'].values + proberad

        if tjm_scale:
            return C.get_ccs(M.points, radii)[0]
        else:
            return C.get_ccs(M.points, radii, a=1.0, b=1.0)[0]

    # generate random file name to capture CCS software terminal output
    tmp_outfile = random_string(32)
    while os.path.exists(tmp_outfile):
        tmp_outfile = "%s.pdb" % random_string(32)

    if pdbname == "":
        # write temporary pdb file of current structure on which to launch
        # CCS calculation
        filename = "%s.pdb" % random_string(32)
        while os.path.exists(filename):
            filename = "%s.pdb" % random_string(32)

        M.write_pdb(filename, [M.current])

    else:
        filename = pdbname
        # if file was already provided, verify its existence first!
        if os.path.isfile(pdbname) != 1:
            raise Exception("ERROR: %s not found!" % pdbname)

    try:

        if impact_path == '':
                try:
                    impact_path = os.path.join(os.environ['IMPACTPATH'], "bin")
                except KeyError:
                    raise Exception("IMPACTPATH environment variable undefined")

        # if using impact, create parameterization file containing a
        # description for Z atoms (pseudoatom name used in this code)
        f = open('params', 'w')

        f.write('[ defaults ]\n H 2.2\n C 2.91\n N 2.91\n O 2.91\n P 2.91\n S 2.91\n')
        #@fix this for the general case of multiple atoms with different radius
        f.write(' Z %s' % (np.unique(M.data['radius'])[0] + proberad))
        impact_options += " -param params"

        f.close()

        if "win" in sys.platform:
            impact_name = os.path.join(impact_path, "impact.exe")
        else:
            impact_name = os.path.join(impact_path, "impact")

        subprocess.check_call('%s  %s -rProbe 0 %s > %s' % (impact_name, impact_options, filename, tmp_outfile), shell=True)

    except Exception as e:
        raise Exception(str(e))

    #parse output generated by IMPACT and written into a file
    try:
        f = open(tmp_outfile, 'r')
        for line in f:
            w = line.split()
            if len(w) > 0 and w[0] == "CCS":

                if tjm_scale:
                    v = float(w[-2])
                else:
                    v = float(w[3])

                break

        f.close()

        # clean temp files if needed
        #(if a filename is provided, don't delete it!)
        os.remove(tmp_outfile)
        if pdbname == "":
            os.remove(filename)

        return v

    except:
        # clean temp files
        os.remove(tmp_outfile)
        if pdbname == "":
            os.remove(filename)

        return -4



class CCS(object):
    '''
    CCS calculator (wrapper for C library)
    '''

    def __init__(self, libfile):
        '''
        initialize by loading IMPACT library

        :param libfile: library path
        '''

        try:
            self.libs = cdll.LoadLibrary(libfile)
            self.libs.pa2tjm.restype = c_float

        except:
            raise Exception("loading library %s failed!" % libfile)

        # declare output variables
        self.ccs = c_float()
        self.sem = c_float()
        self.niter = c_int()

    def get_ccs(self, points, radii, a=0.842611, b=1.051280):
        '''
        compute CCS using the PA method as implemented in IMPACT library.

        :param points: xyz coordinates of atoms, Angstrom (nx3 numpy array)
        :param radii: van der Waals radii associated to every point (numpy array with n elements)
        :param a: power-law factor for calibration with TJM
        :param b: power-law exponent for calibration with TJM
        :returns: TJM CCS
        :returns: standard error
        :returns: number of iterations
        '''

        # create ctypes for intput data
        unravel = np.ravel(points)
        cpoints = (c_float * len(unravel))(*unravel)
        cradii = (c_float * len(radii))(*radii)
        natoms = (c_int)(len(radii))

        # call library, and rescale obtained value using exponential law
        self.libs.ccs_from_atoms_defaults(natoms, byref(cpoints), byref(cradii), byref(self.ccs), byref(self.sem), byref(self.niter))
        ccs_tjm = self.libs.pa2tjm(c_float(a), c_float(b), self.ccs)

        return ccs_tjm, self.sem.value, self.niter.value


def random_string(length=32):
    '''
    generate a random string of arbitrary characters. Useful to generate temporary file names.

    :param length: length of random string
    '''
    return ''.join([random.choice(string.ascii_letters)
                    for n in range(length)])
