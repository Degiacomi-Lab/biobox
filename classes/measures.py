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

import subprocess
import os
import sys
import random
import string

import numpy as np
from ctypes import cdll, c_int, c_float, byref


def saxs(M, crysol_path='', crysol_options="-lm 20 -ns 500", pdbname=""):
    '''
    compute SAXS curve using crysol (from ATSAS suite)

    :param crysol_path: path to crysol executable. By default, the environment variable ATSASPATH is sought. This allows redirecting to a specific impact root folder.
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

    call_line = os.path.join(crysol_path,"crysol")
    try:
        subprocess.check_call('%s %s %s >& /dev/null' %(call_line, crysol_options, pdbname), shell=True)
    except Exception as e:
        raise Exception("ERROR: crysol calculation failed!")

    data = np.loadtxt("%s00.int" % outfile, skiprows=1)
    try:
        os.remove("%s00.alm" % outfile)
        os.remove("%s00.int" % outfile)
        os.remove("%s00.log" % outfile)
        os.remove("%s.pdb" % outfile)
    except Exception, ex:
        pass

    return data[:, 0:2]


def ccs(M, use_lib=True, impact_path='', impact_options="-Octree -nRuns 32 -cMode sem -convergence 0.01", pdbname="", tjm_scale=False, proberad=1.0):
    '''
    compute CCS calling either impact.

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

    elif this_inst == "Molecule" and "atoms_ccs" not in M.data.columns:
        M.assign_atomtype()
        M.get_atoms_ccs()
        
    if use_lib and pdbname == "":

        #if True:
        from biobox.classes.measures import CCS
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

    def __init__(self, libfile="lib/libimpact.dll"):
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
                    for n in xrange(length)])


if __name__ == "__main__":

    import sys
    from biobox import Molecule, Multimer  # , Density

    filename = sys.argv[1]

    # load molecule
    M = Molecule()
    M.import_pdb(filename)
    M.get_atoms_ccs()

    # EXAMPLE 1: using the library called through BiobOx
    print "lib through BiobOx: %s A2" % M.ccs()

    # EXAMPLE 2: calling the library directly
    # extract atomic radii, and add probe to atom radius.
    # note: in BiobOx, radii are based on atomtype.
    radii = M.get_atoms_ccs() + 1.0
    C = CCS()
    ccs, sem, niter = C.get_ccs(M.points, radii)
    print "lib called directly: %s A2" % ccs

    # EXAMPLE 3: without library:
    # temporary PDB is written, and submitted to impact executable.
    print "exe through BiobOx: %s" % M.ccs(use_lib=False)

    # EXAMPLE 4: CCS of a protein assembly
    A = Multimer()
    A.load(M, 3)
    A.make_circular_symmetry(30)
    print "assembly CCS: %s" % A.ccs()
