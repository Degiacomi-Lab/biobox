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

'''
Functions to measure characteristics of any Biobox object
'''

import subprocess
import os
import shlex
import sys
import random
import string

import numpy as np
from ctypes import cdll, c_int, c_float, byref

import biobox.lib.fastmath as FM  # cython routines


def sasa_c(M, targets=[], probe=1.4, n_sphere_point=960, threshold=0.05):
    '''
    compute the accessible surface area using the Shrake-Rupley algorithm ("rolling ball method").

    Alias of :func:`sasa <biobox.measures.calculators.sasa>`, kept for backwards compatibility.

    :param M: any biobox object
    :param targets: indices of the atoms whose surface is estimated. By default (empty list), all atoms are used.
    :param probe: radius of the "rolling ball", in A
    :param n_sphere_point: number of mesh points per atom
    :param threshold: fraction of mesh points that must be exposed for an atom to be listed among the surface atoms. It does not affect the area or the mesh.
    :returns: accessible surface area in A^2, summed over all target atoms
    :returns: mx3 numpy array of the exposed mesh points forming the accessible surface mesh
    :returns: numpy array of int, indices of the surface atoms, i.e. target atoms whose exposed fraction exceeds threshold
    '''
    return sasa(M, targets=targets, probe=probe, n_sphere_point=n_sphere_point, threshold=threshold)


def _golden_spiral(n_sphere_point):
    '''
    points evenly distributed on a unit sphere, placed along a golden spiral.

    :param n_sphere_point: number of points
    :returns: n_sphere_point x 3 numpy array
    '''
    k = np.arange(int(n_sphere_point))
    inc = np.pi * (3 - np.sqrt(5))
    offset = 2 / float(n_sphere_point)
    y = k * offset - 1 + (offset / 2)
    r = np.sqrt(1 - y * y)
    phi = k * inc
    return np.column_stack((np.cos(phi) * r, y, np.sin(phi) * r))


def sasa(M, targets=[], probe=1.4, n_sphere_point=960, threshold=0.05):
    '''
    compute the accessible surface area using the Shrake-Rupley algorithm ("rolling ball method").

    every target atom is surrounded by a mesh of points at distance radius+probe from its centre,
    i.e. the positions the centre of a probe touching the atom can take. A mesh point is exposed
    when it lies farther than radius+probe from every other atom, and the area of the atom is the
    exposed fraction of its sphere. All atoms of M act as occluders, whether or not they are
    targets. Atomic radii are read from the "radius" column of M.data, and only the current
    conformation is measured. A ValueError is raised if any radius is not finite, and an empty
    structure returns an area of 0.0.

    :param M: any biobox object
    :param targets: indices of the atoms whose surface is estimated. By default (empty list), all atoms are used.
    :param probe: radius of the "rolling ball", in A
    :param n_sphere_point: number of mesh points per atom
    :param threshold: fraction of mesh points (between 0 and 1) that must be exposed for an atom to be listed among the surface atoms. It does not affect the area or the mesh.
    :returns: accessible surface area in A^2, summed over all target atoms
    :returns: mx3 numpy array of the exposed mesh points forming the accessible surface mesh
    :returns: numpy array of int, indices of the surface atoms, i.e. target atoms whose exposed fraction exceeds threshold
    '''

    from scipy.spatial import cKDTree
    import biobox.measures.interaction as I

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()

    elif this_inst in ["Assembly", "Polyhedron"]:
        M = M.make_structure()

    if len(targets) == 0:
        targets = range(0, len(M.points), 1)

    # getting radii associated to every atom
    points = np.asarray(M.points, dtype=float)
    radii = np.asarray(M.data['radius'].values, dtype=float)

    if threshold < 0.0 or threshold > 1.0:
        raise Exception("ERROR: threshold should be a floating point between 0 and 1!")

    if len(points) == 0:
        return 0.0, np.empty((0, 3)), np.array([], dtype=int)

    n_missing = int(np.count_nonzero(~np.isfinite(radii)))
    if n_missing > 0:
        raise ValueError("%s atoms have no finite radius" % n_missing)

    sphere_points = _golden_spiral(n_sphere_point)
    const = 4.0 * np.pi / len(sphere_points)

    # a KD-tree rather than a full contact map, so that memory grows with the number of atoms
    # rather than with its square
    tree = cKDTree(points)
    max_radius = radii.max()

    asa = 0.0
    surface_atoms = []
    mesh_pts = []
    # compute accessible surface for every atom
    for i in targets:

        # place mesh points around atom of choice
        mesh = sphere_points * (radii[i] + probe) + points[i]

        # atom j can cover a mesh point of atom i only if their centres are closer than
        # radii[i] + radii[j] + 2*probe. The atom itself is excluded: its mesh lies at exactly
        # radii[i] + probe from its centre, and rounding would otherwise flag about half of the
        # points as buried by the atom they belong to
        candidates = np.asarray(tree.query_ball_point(points[i], radii[i] + max_radius + probe * 2), dtype=int)
        d_ij = np.linalg.norm(points[candidates] - points[i], axis=1)
        test = candidates[(d_ij < radii[i] + radii[candidates] + probe * 2) & (candidates != i)]

        # lines=neighbours, columns=mesh points. A mesh point is exposed when no neighbour
        # surface is closer to it than the probe radius, i.e. a probe centred there fits
        if len(test) == 0:
            exposed = np.ones(len(mesh), dtype=bool)
        else:
            dist = I.distance_matrix(points[test], mesh) - radii[test][:, np.newaxis]
            exposed = ~np.any(dist < probe, axis=0)

        cnt = int(np.count_nonzero(exposed))
        mesh_pts.extend(mesh[exposed])
        asa += const * cnt * (radii[i] + probe)**2

        # an atom counts as a surface atom if a sufficient amount of its mesh points is exposed
        if cnt > n_sphere_point * threshold:
            surface_atoms.append(i)

    return asa, np.array(mesh_pts).reshape(-1, 3), np.array(surface_atoms, dtype=int)

def rgyr(M):
    '''
    compute the radius of gyration of the current conformation, unweighted (every atom counts equally) and relative to the center of geometry.

    :param M: any biobox object
    :returns: radius of gyration (float), in the units of the coordinates (A)
    '''

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()

    elif this_inst in ["Assembly", "Polyhedron"]:
        M = M.make_structure()

    d_square = np.sum((M.points - M.get_center())**2, axis=1)
    return np.sqrt(np.sum(d_square) / d_square.shape[0])


def saxs(M, crysol_path='', crysol_options="-lm 20 -ns 500", pdbname=""):
    '''
    compute SAXS curve using crysol (from ATSAS suite)

    Unless pdbname is given, the current conformation of M is written to a temporary PDB file in the
    working directory, deleted afterwards together with the crysol output files.

    :param M: any biobox object
    :param crysol_path: folder containing the crysol executable. If not provided, the environment variable ATSASPATH is sought instead. This allows redirecting to a specific ATSAS bin folder.
    :param crysol_options: flags to be passed to crysol executable
    :param pdbname: if a file has been already written, crysol analyzes it instead of M
    :returns: SAXS curve (nx2 numpy array), i.e. the first two columns of the crysol .int file: scattering vector and intensity in solution
    '''

    if crysol_path == '':
        try:
            crysol_path = os.environ['ATSASPATH']
        except KeyError:
            raise Exception("ATSASPATH environment variable undefined")

    temporary_pdb = pdbname == ""
    if temporary_pdb:
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

    # crysol names its output after the input file, in the working directory
    outfile = os.path.splitext(os.path.basename(pdbname))[0]

    call_line = [os.path.join(crysol_path, "crysol")] + shlex.split(crysol_options) + [pdbname]
    try:
        subprocess.check_call(call_line, stdout=subprocess.DEVNULL)
        data = np.loadtxt("%s00.int" % outfile, skiprows=1)
    finally:
        for ext in ["00.alm", "00.int", "00.log"]:
            if os.path.exists(outfile + ext):
                os.remove(outfile + ext)
        if temporary_pdb:
            os.remove(pdbname)

    return data[:, 0:2]


def ccs(M, use_lib=True, impact_path='', impact_options="-Octree -nRuns 32 -cMode sem -convergence 0.01", pdbname="", tjm_scale=False, proberad=1.0):
    '''
    compute CCS with IMPACT, either via its library or via a system call to its executable.

    The library is used when use_lib is True and pdbname is not given. Otherwise, the executable is called on
    pdbname or on a temporary PDB file of the current conformation, and a "params" file is written in the working directory.
    If M is a Molecule without an "atom_ccs" column, atom types and CCS radii are assigned to it first.

    :param M: any biobox object
    :param use_lib: if true, impact library will be used, if false a system call to impact executable will be performed instead
    :param impact_path: folder containing libimpact (library mode) or the impact executable (executable mode). By default, the "lib" or "bin" subfolder of the environment variable IMPACTPATH is used.
    :param impact_options: flags to be passed to impact executable (executable mode only)
    :param pdbname: if a file has been already written, impact executable analyzes it instead of M
    :param tjm_scale: if True, CCS value calculated with PA method is scaled to better match trajectory method.
    :param proberad: radius of probe in A, added to the atomic radii. Do find out if your impact library already adds this value by default or not (old ones do)!
    :returns: CCS value in A^2, or -4 if parsing the output of impact executable fails
    '''

    #make sure that everything is collected as a Structure object, and radii are available
    this_inst = type(M).__name__
    if this_inst == "Multimer":
        M = M.make_molecule()
        M.assign_atomtype()
        M.get_atoms_ccs()

    elif this_inst in ["Assembly", "Polyhedron"]:
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

            if sys.platform.startswith("win"):
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

        if sys.platform.startswith("win"):
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
        :returns: PA CCS rescaled by IMPACT's pa2tjm power law with parameters a and b, in A^2
        :returns: standard error of the PA CCS
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
    generate a random string of ASCII letters. Useful to generate temporary file names.

    :param length: length of random string
    :returns: random string
    '''
    return ''.join([random.choice(string.ascii_letters)
                    for n in range(length)])
