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


import numpy as np
from ctypes import cdll, c_int, c_float, byref


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


if __name__ == "__main__":

    from biobox import Molecule, Multimer  # , Density

    filename = "test/MsAcr2_IXI_merge.align.pdb"  # sys.argv[1]

    # load molecule
    M = Molecule()
    M.import_pdb(filename)
    M.get_atoms_ccs()

    # EXAMPLE 1: using the library called through SBT
    print "lib through SBT: %s A2" % M.ccs()

    # EXAMPLE 2: calling the library directly
    # extract atomic radii, and add probe to atom radius.
    # note: in SBT, radii are based on atomtype.
    radii = M.get_atoms_ccs() + 1.0
    C = CCS()
    ccs, sem, niter = C.get_ccs(M.points, radii)
    print "lib called directly: %s A2" % ccs

    # EXAMPLE 3: without library:
    # temporary PDB is written, and submitted to impact executable.
    print "exe through SBT: %s" % M.ccs(use_lib=False)

    # EXAMPLE 4: CCS of a protein assembly
    A = Multimer()
    A.load(M, 3)
    A.make_circular_symmetry(30)
    print "assembly CCS: %s" % A.ccs()
