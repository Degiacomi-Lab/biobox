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

import os
from copy import deepcopy
import numpy as np
import scipy.signal
import pandas as pd

from biobox.classes.structure import Structure

class Molecule(Structure):
    '''
    Subclass of :func:`Structure <structure.Structure>`, allows reading, manipulating and analyzing molecular structures.
    '''

    chain_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd',
                   'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                   'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    def __init__(self):
        '''
        At instantiation, properties associated to every individual atoms are stored in a pandas Dataframe self.data.
        The columns of the self.data have the following names:
        atom, index, name, resname, chain, resid, beta, occupancy, atomtype, radius, charge.

        self.knowledge contains a knowledge base about atoms and residues properties. Default values are:

        * 'residue_mass' property stores the average mass for most common aminoacids (values from Expasy website)
        * 'atom_vdw' vdw radius of common atoms
        * 'atom_mass' mass of common atoms

        The knowledge base can be edited. For instance, to add information about residue "TST" mass in molecule M type: M.knowledge['mass_residue']["TST"]=142.42
        '''

        super(Molecule, self).__init__(r=np.array([]))

        # knowledge base about atoms and residues properties (entry keys:
        # 'residue_mass', 'atom_vdw', 'atom_, mass' can be edited)
        self.knowledge = {}
        self.knowledge['residue_mass'] = {"ALA": 71.0788, "ARG": 156.1875, "ASN": 114.1038, "ASP": 115.0886, "CYS": 103.1388, "CYX": 103.1388, "GLU": 129.1155, "GLN": 128.1307, "GLY": 57.0519,
                                          "HIS": 137.1411, "HSE": 137.1411, "HSD": 137.1411, "HSP": 137.1411, "HIE": 137.1411, "HID": 137.1411, "HIP": 137.1411, "ILE": 113.1594, "LEU": 113.1594,
                                          "LYS": 128.1741, "MET": 131.1926, "MSE": 131.1926, "PHE": 147.1766, "PRO": 97.1167, "SER": 87.0782, "THR": 101.1051, "TRP": 186.2132, "TYR": 163.1760, "VAL": 99.1326}
        self.knowledge['atom_vdw'] = {'H': 1.20, 'N': 1.55, 'NA': 2.27, 'CU': 1.40, 'CL': 1.75, 'C': 1.70, 'O': 1.52, 'I': 1.98, 'P': 1.80, 'B': 1.85, 'BR': 1.85, 'S': 1.80, 'SE': 1.90,
                                      'F': 1.47, 'FE': 1.80, 'K': 2.75, 'MN': 1.73, 'MG': 1.73, 'ZN': 1.39, 'HG': 1.8, 'XE': 1.8, 'AU': 1.8, 'LI': 1.8, '.': 1.8}
        self.knowledge['atom_ccs'] = {'H': 1.2, 'C': 1.91, 'N': 1.91, 'O': 1.91, 'P': 1.91, 'S': 1.91, '.': 1.91}
        self.knowledge['atom_mass'] = {"H": 1.00794, "D": 2.01410178, "HE": 4.00, "LI": 6.941, "BE": 9.01, "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.998403, "NE": 20.18, "NA": 22.989769,
                                       "MG": 24.305, "AL": 26.98, "SI": 28.09, "P": 30.973762, "S": 32.065, "CL": 35.453, "AR": 39.95, "K": 39.0983, "CA": 40.078, "SC": 44.96, "TI": 47.87, "V": 50.94,
                                       "CR": 51.9961, "MN": 54.938045, "FE": 55.845, "CO": 58.93, "NI": 58.6934, "CU": 63.546, "ZN": 65.409, "GA": 69.72, "GE": 72.64, "AS": 74.9216, "SE": 78.96,
                                       "BR": 79.90, "KR": 83.80, "RB": 85.47, "SR": 87.62, "Y": 88.91, "ZR": 91.22, "NB": 92.91, "MO": 95.94, "TC": 98.0, "RU": 101.07, "RH": 102.91, "PD": 106.42,
                                       "AG": 107.8682, "CD": 112.411, "IN": 114.82, "SN": 118.71, "SB": 121.76, "TE": 127.60, "I": 126.90447, "XE": 131.29, "CS": 132.91, "BA": 137.33, "PR": 140.91,
                                       "EU": 151.96, "GD": 157.25, "TB": 158.93, "W": 183.84, "IR": 192.22, "PT": 195.084, "AU": 196.96657, "HG": 200.59, "PB": 207.2, "U": 238.03}
        self.knowledge['atomtype'] = {"C": "C", "CA": "C", "CB": "C", "CG": "C", "CG1": "C", "CG2": "C", "CZ": "C", "CD1": "C", "CD2": "C",
                                      "CD": "C", "CE": "C", "CE1": "C", "CE2": "C", "CE3": "C", "CZ2": "C", "CZ3": "C", "CH2": "C",
                                      "N": "N", "NH1": "N", "NH2": "N", "NZ": "N", "NE": "N", "NE1": "N", "NE2": "N", "ND1": "N", "ND2": "N",
                                      "O": "O", "OG": "O", "OG1": "O", "OG2": "O", "OD1": "O", "OD2": "O", "OE1": "O", "OE2": "O", "OH": "O", "OXT": "O",
                                      "SD": "S", "SG": "S"}

    def know(self, prop):
        '''
        return information from knowledge base

        :param prop: desired property to extract from knowledge base
        :returns: value associated to requested property, or nan if failed
        '''
        if str(prop) in self.knowledge:
            return self.knowledge[str(prop)]
        else:
            raise Exception("entry %s not found in knowledge base!" % prop)

    def import_pdb(self, pdb, include_hetatm=False):
        '''
        read a pdb (possibly containing containing multiple models).

        Models are split according to ENDMDL and END statement.
        All alternative coordinates are expected to have the same atoms.
        After loading, the first model (M.current_model=0) will be set as active.

        :param pdb: PDB filename
        :param include_hetatm: if True, HETATM will be included (they get skipped if False)
        '''

        try:
            f_in = open(pdb, "r")
        except Exception, ex:
            raise Exception('ERROR: file %s not found!' % pdb)

        # store filename
        self.properties["filename"] = pdb

        data_in = []
        p = []
        r = []
        e = []
        alternative = []
        biomt = []
        symm = []
        for line in f_in:
            record = line[0:6].strip()

            # load biomatrix, if any is present
            if "REMARK 350   BIOMT" in line:
                try:
                    biomt.append(line.split()[4:8])
                except Exception, ex:
                    raise Exception("ERROR: biomatrix format seems corrupted")

            # load symmetry matrix, if any is present
            if "REMARK 290   SMTRY" in line:
                try:
                    symm.append(line.split()[4:8])
                except Exception, ex:
                    raise Exception("ERROR: symmetry matrix format seems corrupted")

            # if a complete model was parsed store all the saved data into
            # self.data entries (if needed) and temporary alternative
            # coordinates list
            if record == "ENDMDL" or record == "END":

                if len(alternative) == 0:

                    # load all the parsed data in superclass data (Dataframe)
                    # and points data structures
                    try:
                        #building dataframe
                        data = np.array(data_in).astype(str)
                        cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
                        idx = np.arange(len(data))
                        self.data = pd.DataFrame(data, index=idx, columns=cols)

                    except Exception, ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pdb)

                    # saving vdw radii
                    try:
                        self.data['radius'] = np.array(r)
                    except Exception, ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pdb)

                    # save default charge state
                    self.data['charge'] = np.array(e)

                # save 3D coordinates of every atom and restart the accumulator
                try:
                    if len(p) > 0:
                        alternative.append(np.array(p))
                    p = []
                except Exception, ex:
                    raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' % pdb)

            if record == 'ATOM' or (include_hetatm and record == 'HETATM'):

                # extract xyz coordinates (save in list of point coordinates)
                p.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                # if no complete model has been yet parsed, load also
                # information about atoms(resid, resname, ...)
                if len(alternative) == 0:
                    w = []
                    # extract ATOM/HETATM statement
                    w.append(line[0:6].strip())
                    w.append(line[6:11].strip())  # extract atom index
                    w.append(line[12:17].strip())  # extract atomname
                    w.append(line[17:20].strip())  # extract resname
                    w.append(line[21].strip())  # extract chain name
                    w.append(line[22:26].strip())  # extract residue id

                    # extract occupancy
                    try:
                        w.append(float(line[54:60]))
                    except Exception, ex:
                        w.append(1.0)

                    # extract beta factor
                    try:
                        # w.append("{0.2f}".format(float(line[60:66])))
                        w.append(float(line[60:66]))
                    except Exception, ex:
                        w.append(0.0)

                    # extract atomtype
                    try:
                        w.append(line[76:78].strip())
                    except Exception, ex:
                        w.append("")

                    # use atomtype to extract vdw radius
                    try:
                        r.append(self.know('atom_vdw')[line[76:78].strip()])
                    except Exception, ex:
                        r.append(self.know('atom_vdw')['.'])

                    # assign default charge state of 0
                    e.append(0.0)

                    data_in.append(w)

        f_in.close()

        # if p list is not empty, that means that the PDB file does not finish with an END statement (like the ones generated by SBT, for instance).
        # In this case, dump all the remaining stuff into alternate coordinates
        # array and (if needed) into properties dictionary.
        if len(p) > 0:

            # if no model has been yet loaded, save also information in
            # properties dictionary.
            if len(alternative) == 0:

                # load all the parsed data in superclass properties['data'] and
                # points data structures
                try:
                    #building dataframe
                    data = np.array(data_in).astype(str)
                    cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
                    idx = np.arange(len(data))
                    self.data = pd.DataFrame(data, index=idx, columns=cols)

                except Exception, ex:
                    raise Exception('ERROR: something went wrong when saving data in %s!\nERROR: are all the columns separated?' %pdb)

                try:
                    self.data['radius'] = np.array(r)
                except Exception, ex:
                    raise Exception('ERROR: something went wrong when saving van der Waals radii in %s!\nERROR: are all the columns separated?' % pdb)

                # save default charge state
                self.properties['charge'] = np.array(e)

            # save 3D coordinates of every atom and restart the accumulator
            try:
                if len(p) > 0:
                    alternative.append(np.array(p))
                p = []
            except Exception, ex:
                raise Exception('ERROR: something went wrong when saving coordinates in %s!\nERROR: are all the columns separated?' %pdb)

        # transform the alternative temporary list into a nice multiple
        # coordinates array
        if len(alternative) > 0:
            try:
                alternative_xyz = np.array(alternative).astype(float)
            except Exception, e:
                alternative_xyz = np.array([alternative[0]]).astype(float)
                print 'WARNING: found %s models, but their atom count differs' % len(alternative)
                print 'WARNING: treating only the first model in file %s' % pdb
                #raise Exception('ERROR: models appear not to have the same amount of atoms')

            self.add_xyz(alternative_xyz)
        else:
            raise Exception('ERROR: something went wrong when saving alternative coordinates in %s!\nERROR: no model was loaded... are ENDMDL statements there?' % pdb)

        # if biomatrix information is provided, creat
        if len(biomt) > 0:

            # test whether there are enough lines to create biomatrix
            # statements
            if np.mod(len(biomt), 3):
                raise Exception('ERROR: found %s BIOMT entries. A multiple of 3 is expected'%len(biomt))

            b = np.array(biomt).astype(float).reshape((len(biomt) / 3, 3, 4))
            self.properties["biomatrix"] = b

        # if symmetry information is provided, create entry in properties
        if len(symm) > 0:

            # test whether there are enough lines to create biomatrix
            # statements
            if np.mod(len(symm), 3):
                raise Exception('ERROR: found %s SMTRY entries. A multiple of 3 is expected'%len(symm))

            b = np.array(symm).astype(float).reshape((len(symm) / 3, 3, 4))
            self.properties["symmetry"] = b

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["index"] = self.data["index"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)


    def import_pqr(self, pqr, include_hetatm=False):
        '''
        Read a pqr (possibly containing containing multiple models).

        models are split according to ENDMDL and END statement.
        All alternative coordinates are expected to have the same atoms.
        After loading, the first model (M.current_model=0) will be set as active.

        :param pqr: PQR filename
        :param include_hetatm: if True, HETATM will be included (they get skipped if False)
        '''

        try:
            f_in = open(pqr, "r")
        except Exception, ex:
            raise Exception('ERROR: file %s not found!' % pqr)

        # store filename
        self.properties["filename"] = pqr

        data_in = []
        p = []  # collects coordinates for every model
        r = []  # vdW radii
        e = []  # electrostatics
        alternative = []
        for line in f_in:
            record = line[0:6].strip()
            # if a complete model was parsed store all the saved data into
            # self.properties entries (if needed) and temporary alternative
            # coordinates list
            if record == "ENDMDL" or record == "END":
                if len(alternative) == 0:
                    # load all the parsed data in superclass properties['data']
                    # and points data structures
                    try:
                        #building dataframe
                        data = np.array(data_in).astype(str)
                        cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
                        idx = np.arange(len(data))
                        self.data = pd.DataFrame(data, index=idx, columns=cols)

                    except Exception, ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pqr)

                    # saving vdw radii
                    try:
                        self.data['radius'] = np.array(r)
                    except Exception, ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pqr)

                    # saving electrostatics
                    try:
                        self.data['charge'] = np.array(e)
                    except Exception, ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' % pqr)

                # save 3D coordinates of every atom and restart the accumulator
                try:
                    if len(p) > 0:
                        alternative.append(np.array(p))
                    p = []
                except Exception, ex:
                    raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pqr)

            if record == 'ATOM' or (include_hetatm and record == 'HETATM'):

                # extract xyz coordinates (save in list of point coordinates)
                p.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                # if no complete model has been yet parsed, load also
                # information about atoms(resid, resname, ...)
                if len(alternative) == 0:

                    # extract charge
                    try:
                        # 54 is separator, 55 is plus/minus
                        e.append(float(line[54:62]))
                    except Exception, ex:
                        e.append(0.0)

                    # extract vdW radius
                    try:
                        r.append(float(line[62:69]))
                    except Exception, ex:
                        r.append(self.know('atom_vdw')['.'])

                    # initialize list
                    w = []

                    # extract ATOM/HETATM statement
                    w.append(line[0:6].strip())
                    w.append(line[6:11].strip())  # extract atom index
                    w.append(line[12:17].strip())  # extract atomname
                    w.append(line[17:20].strip())  # extract resname
                    w.append(line[21].strip())  # extract chain name
                    w.append(line[22:26].strip())  # extract residue id

                    # extract occupancy
                    w.append('1')

                    # extract beta factor
                    w.append('0')

                    # extract atomtype from atomname in BMRB notation
                    # http://www.bmrb.wisc.edu/ref_info/atom_nom.tbl
                    w.append(line[12:17].strip()[0])
                    # w.append(line[76:78].strip())

                    data_in.append(w)

        f_in.close()

        # if p list is not empty, that means that the pqr file does not finish with an END statement (like the ones generated by SBT, for instance).
        # In this case, dump all the remaining stuff into alternate coordinates
        # array and (if needed) into properties dictionary.
        if len(p) > 0:

            # if no model has been yet loaded, save also information in
            # properties dictionary.
            if len(alternative) == 0:

                # load all the parsed data in superclass properties['data'] and
                # points data structures
                try:
                    #building dataframe
                    data = np.array(data_in).astype(str)
                    cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
                    idx = np.arange(len(data))
                    self.data = pd.DataFrame(data, index=idx, columns=cols)

                except Exception, ex:
                    raise Exception('ERROR: something went wrong when saving data in %s!\nERROR: are all the columns separated?' % pqr)

                try:
                    self.data['radius'] = np.array(r)
                except Exception, ex:
                    raise Exception('ERROR: something went wrong when saving van der Waals radii in %s!\nERROR: are all the columns separated?' %pqr)

            # save 3D coordinates of every atom and restart the accumulator
            try:
                if len(p) > 0:
                    alternative.append(np.array(p))
                p = []
            except Exception, ex:
                raise Exception('ERROR: something went wrong when saving coordinates in %s!\nERROR: are all the columns separated?' %pqr)

        # transform the alternative temporary list into a nice multiple
        # coordinates array
        if len(alternative) > 0:
            try:
                alternative_xyz = np.array(alternative).astype(float)
            except Exception, ex:
                alternative_xyz = np.array([alternative[0]]).astype(float)
                print 'WARNING: found %s models, but their atom count differs' % len(alternative)
                print 'WARNING: treating only the first model in file %s' % pqr
                #raise Exception('ERROR: models appear not to have the same amount of atoms')

            self.add_xyz(alternative_xyz)
        else:
            raise Exception('ERROR: something went wrong when saving alternative coordinates in %s!\nERROR: no model was loaded... are ENDMDL statements there?' % pqr)

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["index"] = self.data["index"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)

    def assign_atomtype(self):
        '''
        guess atomtype from atom names
        '''
        
        a_type = []
        for i in xrange(0, len(self.data), 1):
            atom = self.data["name"][i]
            try:
                a_type.append(self.knowledge["atomtype"][atom])
            except Exception, ex:
                a_type.append("")

        self.data["name"] = a_type

    def get_vdw_density(self, buff=3, step=0.5, kernel_half_width=10):
        '''
        generate density map from points based on the van der Waals readius of the atoms

        :param buff: Buffer used to create the boundaries of the density map
        :param step: Stepsize for creating the density object
        :param kernel_half_width: Kernel half width of the gaussian kernel, will be scaled by atom specific sigma
        :returns: :func:`Density <density.Density>` object, containing a density map
        '''
        atomdata = [["C", 1.7, 1.455, 0.51], ["H", 1.2, 0.72, 0.25],
                    ["O", 1.52, 1.15, 0.42], ["S", 1.8, 1.62, 0.54],
                    ["N", 1.55, 1.2, 0.44]]

        dens = []
        for d in atomdata:
            # select atomtype and point only to those coordinates
            pts = self.points[[self.properties["data"][:, 8] == d[0]]]
            # print self.properties['data'][:, 8]
            # if there are atoms from a certain type
            if len(pts) > 0:
                # use the standard density calculation with a atom-type
                # specific sigma value
                D = self.get_density(buff=buff, step=step, kernel_half_width=kernel_half_width, sigma=d[2])
                # export the np-array density map for summing
                d_tmp = D.properties["density"]

            # print 'atom %s has a maximum density of %s, occurs %s times in the protein and the density map has a shape of %s'%(d[0], np.max(d_tmp), len(pts), d_tmp.shape)
            # print 'one entry in d_tmp: \n%s'%d_tmp[35][27][20]
            # print np.unravel_index(d_tmp.argmax(), d_tmp.shape)
            # initialize the dens-list if it's empty
            if len(dens) == 0:
                dens = deepcopy(d_tmp)
            # sum up the densities from all atom types
            else:
                dens += d_tmp  # dens is a 3d numpy array with point intensities

        # print 'one entry in dens is: \n%s'%dens[35][27][20]

        from biobox.classes.density import Density
        D = Density()
        D.properties['density'] = dens
        D.properties['size'] = np.array(dens.shape)
        D.properties['origin'] = np.mean(pts, axis=0) - step * np.array(dens.shape) / 2.0
        D.properties['delta'] = np.identity(3) * step
        D.properties['format'] = 'dx'
        D.properties['filename'] = ''
        D.properties["sigma"] = np.std(dens)

    def get_electrostatics(self, step=1.0, buff=3, threshold=0.01, vdw_kernel_half_width=5, elect_kernel_half_width=12, chain='*', clear_mass=True):
        '''
        generate electrostatics map from points

        :param step: size of cubic voxels, in Angstrom
        :param buff: padding to add at points cloud boundaries
        :param threshold: Threshold used for removing mass occupied space from the electron density map
        :param vdw_kernel_half_width: kernel half width, in voxels
        :param elect_kernel_half_width: kernel half width, in voxels
        :param chain: select chain to use, default all chains
        :returns: positive :func:`Density <density.Density>` object
        :returns: negative :func:`Density <density.Density>` object
        :returns: mass density object
        '''

        pts, idx = self.atomselect(chain, '*', '*', get_index=True)

        try:
            # numpy array of charges [c1, c2, c3, ...]
            charges = self.data['charge'][idx]
        except Exception, e:
            raise Exception('ERROR: No charges associated with %s' % self)

        charges = np.reshape(charges, (len(charges), 1))

        # numpy 2d-array of shape (:, 4): [[x, y, z, charge], [x, y, z,
        # charge], ...]
        c_atoms = np.hstack((pts, charges))

        k = 8.9875517873681764  # Coulomb's constant in nN

        # rectangular box boundaries
        bnds = np.array([[np.min(pts[:, 0]) - buff, np.max(pts[:, 0]) + buff],
                         [np.min(pts[:, 1]) - buff, np.max(pts[:, 1]) + buff],
                         [np.min(pts[:, 2]) - buff, np.max(pts[:, 2]) + buff]])

        xax = np.arange(bnds[0, 0], bnds[0, 1] + step, step)
        yax = np.arange(bnds[1, 0], bnds[1, 1] + step, step)
        zax = np.arange(bnds[2, 0], bnds[2, 1] + step, step)

        # create empty box
        d = np.zeros((len(xax), len(yax), len(zax)))

        # place Kronecker deltas in mesh grid -> discretes point coordinates
        # into mesh grid with an intensity of charge
        for atom in c_atoms:
            xpos = np.argmin(np.abs(xax - atom[0]))
            ypos = np.argmin(np.abs(yax - atom[1]))
            zpos = np.argmin(np.abs(zax - atom[2]))
            d[xpos, ypos, zpos] = atom[3]

        # initialize 3d kernel_box
        # how many steps are needed to reach kernel half width in angstroms
        l_kernel = 2 * elect_kernel_half_width / step
        # the kernel is an empty box...
        kernel = np.zeros((l_kernel, l_kernel, l_kernel))
        it = np.nditer(kernel, flags=['multi_index'])
        while not it.finished:
            x_index = it.multi_index[0]  # indices
            y_index = it.multi_index[1]
            z_index = it.multi_index[2]
            x_coord = x_index * step - (l_kernel / 2)  # real space coordinates
            y_coord = y_index * step - (l_kernel / 2)
            z_coord = z_index * step - (l_kernel / 2)
            distance = np.sqrt(x_coord *x_coord + y_coord *y_coord + z_coord *z_coord)
            if distance > 0.9:  # ... where a hyperbola will be created only outside of H-vdW and inside of relevant kernel-half-width distance
                kernel[x_index, y_index, z_index] = k / distance

            it.iternext()

        # convolve point mesh with 3d coulomb hyperbola
        e = scipy.signal.fftconvolve(d, kernel, mode='same')

        # define mass-occupied space
        # mass_density is Density-object, buff=buff makes shure that the
        # density maps have the same dimensions
        mass_density = self.get_vdw_density(
            buff=buff, step=step, kernel_half_width=vdw_kernel_half_width)

        if clear_mass:
            d_map = mass_density.return_density_map()

        # initialize counter to count the amount of removed points
        i = 0
        # initialize the iterator for said function
        it = np.nditer(e, flags=['multi_index'])
        while not it.finished:
            x_index = it.multi_index[0]  # indices
            y_index = it.multi_index[1]
            z_index = it.multi_index[2]
            x_coord = xax[x_index]  # real space coordinates
            y_coord = yax[y_index]
            z_coord = zax[z_index]
            if d_map[x_index, y_index, z_index] > threshold:
                i += 1
                e[x_index, y_index, z_index] = 0

            it.iternext()

        print 'removed %s points due to van der Waals clashing' % i

        # split the density into two maps
        e_pos = deepcopy(e)
        e_neg = deepcopy(e)
        e_pos[np.where(e_pos < 0)] = 0
        e_neg[np.where(e_neg >= 0)] = 0
        # changes sign of negative array for better visualization
        e_neg = np.negative(e_neg)

        # prepare density data structure for both positive and negative maps at
        # once
        from biobox.classes.density import Density
        D_pos = Density()
        D_neg = Density()
        D_pos.properties['density'] = e_pos
        D_neg.properties['density'] = e_neg
        D_pos.properties['size'] = D_neg.properties['size'] = np.array(e.shape)
        D_pos.properties['origin'] = D_neg.properties['origin'] = np.mean(self.points, axis=0) - step * np.array(e.shape) / 2.0
        D_pos.properties['delta'] = D_neg.properties['delta'] = np.identity(3) * step
        D_pos.properties['format'] = D_neg.properties['format'] = 'dx'
        D_pos.properties['filename'] = D_neg.properties['filename'] = ''
        D_pos.properties["sigma"] = np.std(e_pos)
        D_neg.properties["sigma"] = np.std(e_neg)

        return D_pos, D_neg, mass_density

    def _apply_matrices(self, mats):
        '''
        replicate molecule according to list of transformation matrices
        '''

        M2 = Molecule()
        xyzall = []
        data = []

        # replicate original coordinates applying transformations
        cnt = 0
        for m in mats:
            xyz2 = self.points.copy() + m[:, 3]  # translate
            xyz2 = np.dot(xyz2, m[:, 0:3])  # rotate
            xyzall.extend(xyz2)  # merge

            #assign a specific name to the new chain
            newdata = deepcopy(self.data)
            newdata["chain"] = self.chain_names[cnt]

            if cnt == 0:
                data = [newdata]
            else:
                data.append(newdata)

            cnt += 1

        M2.coordinates = np.array([xyzall])

        # temporary vectorized hexadecimal maker, in case there are more than
        # 9999 atoms
        def dohex(number):
            return hex(number).split('x')[1]

        vhex = np.vectorize(dohex)

        newdata = np.array(data)
        indices = np.linspace(1, len(newdata), len(newdata)).astype(int)
        self.data["index"] = vhex(indices)

        return M2

    def apply_biomatrix(self):
        '''
        if biomatrix information is provided, generate a new molecule with all symmetry operators applied.

        :returns: new Molecule containing several copies of the current Molecule, arranged according to BIOMT statements contained in pdb, or -1 if no transformation matrix is provided
        '''

        # if no biomatrix statement is found, return with error
        if "biomatrix" not in self.properties:
            raise Exception("ERROR: no biomatrix found in pdb %s" %self.properties["filename"])

        return self._apply_matrices(self.properties["biomatrix"])

    def apply_symmetry(self):
        '''
        if symmetry information is provided, generate a new molecule with all symmetry operators applied.

        :returns: new Molecule containing several copies of the current Molecule, arranged according to SMTRY statements contained in pdb, or -1 if no transformation matrix is provided
        '''

        # if no symmetry statement is found, return with error
        if "symmetry" not in self.properties:
            raise Exception("ERROR: no symmetry matrix found in pdb %s" %self.properties["filename"])

        return self._apply_matrices(self.properties["symmetry"])

    def import_gro(self, filename):
        '''
        read a gro possibly containing multiple structures.

        :param filename: name of .gro file to import
        '''

        if not os.path.isfile(filename):
            raise Exception("ERROR: %s not found!" % filename)

        self.clear()

        # print "\n> loading %s..."%filename
        fin = open(filename, "r")

        line = fin.readline()

        d_data = []
        b = []
        while line:
            cnt = 0
            d = []
            atoms = int(fin.readline())
            d_data = []
            while cnt < atoms:
                w = fin.readline().split()
                resnumber = w[0][0:-3]
                resname = w[0][-3:len(w[0])]

                # read data useful for indexing (guess what is missing)
                d_data.append(["ATOM", w[2], w[1], resname, "A", resnumber, "0.0", "0.0", ""])
                d.append([w[3], w[4], w[5]])
                cnt += 1

            # add one conformation (in Angstrom) and store its box size in
            # temporary list
            self.add_xyz(np.array(d).astype(float) * 10)
            b.append(fin.readline().split())

            line = fin.readline()  # attempt to get header of next frame

        # store data information and box size for every frame
        self.properties['box'] = np.array(b).astype(float) * 10


        #building dataframe
        data = np.array(d_data).astype(str)
        cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
        idx = np.arange(len(data))
        self.data = pd.DataFrame(data, index=idx, columns=cols)

        #add additional information about van der waals radius and atoms charge
        self.data['radius'] = np.ones(len(d_data)) * self.know('atom_vdw')['.']
        self.data['charge'] = np.zeros(len(d_data))

        fin.close()


    def get_atoms_ccs(self):
        '''
        return array with atomic CCS radii of every atom in molecule

        :returns: CCS in Angstrom^2
        '''

        if "atom_ccs" in self.data.columns:
            return self.data["atom_ccs"]

        ccs = np.ones(len(self.points)) * self.know("atom_ccs")["."]
        for e in self.know("atom_ccs").keys():
            if "e" != ".":
                ccs[self.data["atomtype"].values == e] = self.knowledge["atom_ccs"][e]

        self.data["atom_ccs"] = ccs

        return ccs

    def get_data(self, indices, columns):
        '''
        Return information about atom of interest (i.e., slice the data DataFrame)

        :param indices: list of indices
        :param columns: list of columns (e.g. ["resname", "resid", "chain"])
        :returns: slice of molecule's data DataFrame
        '''

        return self.data.ix[indices, columns].values
        
    def query(self, query_text, get_index=False):
        '''
        Select specific atoms in a multimer un the basis of a text query.

        :param query_text: string selecting atoms of interest. Uses the pandas query syntax, can access all columns in the dataframe self.data.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: coordinates of the selected points (in a unique array) and, if get_index is set to true, a list of their indices in subunits' self.points array.
        '''

        idx = self.data.query(query_text).index.values

        if get_index:
            return [self.points[idx], idx]
        else:
            return self.points[idx]


    def atomselect(self, chain, res, atom, get_index=False, use_resname=False):
        '''
        Select specific atoms in the protein providing chain, residue ID and atom name.

        :param chain: selection of a specific chain name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param res: residue ID of desired atoms (accepts * as wildcard). Can also be a list or numpy array of of int.
        :param atom: name of desired atom (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.properties['data'])
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the selected points and, if get_index is set to true, their indices in self.points array.
        '''

        # chain name boolean selector
        if isinstance(chain, str):
            if chain == '*':
                chain_query = np.array([True] * len(self.points))
            else:
                chain_query = self.data["chain"].values == chain
                
        elif isinstance(chain, list) or type(chain).__module__ == 'numpy':
            chain_query = self.data["chain"].values == chain[0]
            for c in xrange(1, len(chain), 1):
                chain_query = np.logical_or(chain_query, self.data["chain"].values == chain[c])
        else:
            raise Exception("ERROR: wrong type for chain selection. Should be str, list, or numpy")

        if isinstance(res, str):
            if res == '*':
                res_query = np.array([True] * len(self.points))
            elif use_resname:
                res_query = self.data["resname"].values == res
            else:
                res_query = self.data["resid"].values == res

        elif isinstance(res, int):
            if use_resname:
                res_query = self.data["resname"].values == str(res)
            else:
                res_query = self.data["resid"].values == res

        elif isinstance(res, list) or type(res).__module__ == 'numpy':
            if use_resname:
                res_query = self.data["resname"].values == str(res[0])
            else:
                res_query = self.data["resid"].values == res[0]

            for r in xrange(1, len(res), 1):
                if use_resname:
                    res_query = np.logical_or(res_query, self.data["resname"].values == str(res[r]))
                else:
                    res_query = np.logical_or(res_query, self.data["resid"].values == res[r])

        else:
            raise Exception("ERROR: wrong type for resid selection. Should be int, list, or numpy")

        # atom name boolean selector
        if isinstance(atom, str):
            if atom == '*':
                atom_query = np.array([True] * len(self.points))
            else:
                atom_query = self.data["name"].values == atom
        elif isinstance(atom, list) or type(atom).__module__ == 'numpy':
            atom_query = self.data["name"].values == atom[0]
            for a in xrange(1, len(atom), 1):
                atom_query = np.logical_or(atom_query, self.data["name"].values == atom[a])
        else:
            raise Exception("ERROR: wrong type for atom selection. Should be str, list, or numpy")

        # slice data array and return result (colums 5 to 7 contain xyz coords)
        query = np.logical_and(np.logical_and(chain_query, res_query), atom_query)


        if get_index:
            return [self.points[query], np.where(query == True)[0]]
        else:
            return self.points[query]

    def atomignore(self, chain, res, atom, get_index=False, use_resname=False):
        '''
        Select specific atoms that do not match a specific query (chain, residue ID and atom name).
        Useful to remove from a molecule atoms unwanted for further analysis, alternative conformations, etc...

        :param chain: chain name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param res: residue ID (accepts * as wildcard). Can also be a list or numpy array of of int.
        :param atom: atom name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of atoms in self.points array (and self.properties['data'])
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the selected points not matching the query, if get_index is set to true, their indices in self.points array.
        '''

        #extract indices of atoms matching the query
        idxs = self.atomselect(chain, res, atom, get_index=True, use_resname=use_resname)[1]

        #invert the selection
        idxs2 = []
        for i in xrange(len(self.points)):
            if i not in idxs:
                idxs2.append(i)

        if get_index:
            return [self.points[idxs2], np.array(idxs2)]
        else:
            return self.points[idxs2]

    def same_residue(self, index, get_index=False):
        '''
        Select atoms having the same residue and chain as a given atom (or list of atoms)

        :param index indices: of atoms of choice (integer of list of integers)
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.properties['data'])
        :returns: coordinates of the selected points and, if get_index is set to true, their indices in self.points array.
        '''

        D = self.data.values
        l = D[index]

        if len(l.shape) == 1:
            l = l.reshape(1, len(l))

        test = np.logical_and(D[:, 4] == l[:, 4], D[:, 5] == l[:, 5])

        idxs = np.where(test)[0]
        if len(idxs) > 0:
            pts = self.points[idxs]
        else:
            pts = []

        if get_index:
            return pts, idxs
        else:
            return pts

    def same_residue_unique(self, index, get_index=False):
        '''
        Select atoms having the same residue and chain as a given atom (or list of atoms)

        :param index: indices of atoms of choice (integer of list of integers)
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.properties['data'])
        :returns: coordinates of the selected points and, if get_index is set to true, their indices in self.points array.
        '''

        try:
            test = len(index)  # this should fail if index is a number
            idlist = index
        except Exception, e:
            idlist = [index]

        D = self.data.values
        pts = []
        idxs = []
        for i in idlist:
            done = False
            j = 0  # starting from same point
            while not done:

                if i - j < 0:
                    done = True

                elif D[i, 4] == D[i - j, 4] and D[i, 5] == D[i - j, 5]:

                    if len(idxs) != 0 and i - j not in idxs:
                        pts.append(self.points[i - j])
                        idxs.append(i - j)
                    elif i - j not in idxs:
                        pts = [self.points[i - j].copy()]
                        idxs = [i - j]

                    j += 1

                else:
                    done = True

            j = 1
            done = False
            while not done:

                if i + j == len(self.points):
                    done = True

                elif D[i, 4] == D[i + j, 4] and D[i, 5] == D[i + j, 5]:

                    if len(idxs) != 0 and i + j not in idxs:
                        pts.append(self.points[i + j])
                        idxs.append(i + j)
                    elif i + j not in idxs:
                        pts = [self.points[i + j].copy()]
                        idxs = [i + j]

                    j += 1

                else:
                    done = True

        if get_index:
            return np.array(pts), np.array(idxs)
        else:
            return np.array(pts)

    def get_subset(self, idxs, conformations=[]):
        '''
        Return a :func:`Molecule <molecule.Molecule>` object containing only the selected atoms and frames

        :param ixds: atoms to extract
        :param conformations: frames to extract (by default, all)
        :returns: :func:`Molecule <molecule.Molecule>` object
        '''

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        # create molecule, and push created data information
        M = Molecule()
        postmp = self.coordinates[:, idxs]
        M.coordinates = postmp[frames]
        M.data = self.data[idxs]

        M.current = 0
        M.points = M.coordinates[M.current]

        M.properties['center'] = M.get_center()

        return M

    def guess_chain_split(self, distance=3, use_backbone=True):
        '''
        reassign chain name, using distance cutoff (cannot be undone).
        If two consecutive atoms are beyond a cutoff, a new chain is assigned.

        :param distance: distance cutoff distance
        :param use_backbone: if True, splitting will be performed considering backbone atoms (N and C), all atoms in a sequence otherwise
        '''

        # wipe current chain assignment
        self.data["chain"] = ""

        # identify different chains
        intervals = [0]

        if not use_backbone:
            for i in xrange(len(self.coordinates[0]) - 1):
                dist = np.sqrt(np.dot(self.points[i] - self.points[i + 1], self.points[i] - self.points[i + 1]))
                if dist > distance:
                    intervals.append(i + 1)

        else:
            #aminoacids start with N. Find where a C is too far from the next N.
            posN, idxN = self.atomselect("*", "*", "N", get_index=True)
            posC = self.atomselect("*", "*", "C")
            for i in xrange(len(idxN)-1):
                dist = np.sqrt(np.dot(posC[i] - posN[i+1], posC[i] - posN[i+1]))
                if dist > distance:
                    intervals.append(idxN[i+1])

        intervals.append(len(self.coordinates[0]))

        # separate chains
        for i in xrange(len(intervals) - 1):
            thepos = i % len(self.chain_names)
            self.data[intervals[i]:intervals[i + 1], 4] = self.chain_names[thepos]

        return len(intervals) - 1

    def get_pdb_data(self, index=[]):
        '''
        aggregate data and point coordinates, and return in a unique data structure

        Returned data is a list containing strings for points data and floats for point coordinates
        in the same order as a pdb file, i.e.
        ATOM/HETATM, index, name, resname, chain name, residue ID, x, y, z, occupancy, beta factor, atomtype.

        :returns: list aggregated data and coordinates for every point, as string.
        '''
        if len(index) == 0:
            index = xrange(0, len(self.points), 1)

        # create a list containing all infos contained in pdb (point
        # coordinates and properties)
        d = []
        for i in index:
            d.append([self.data["atom"][i],
                      self.data["index"][i],
                      self.data["name"][i],
                      self.data["resname"][i],
                      self.data["chain"][i],
                      self.data["resid"][i],
                      self.points[i, 0],
                      self.points[i, 1],
                      self.points[i, 2],
                      self.data["beta"][i],
                      self.data["occupancy"][i],
                      self.data["atomtype"][i]])

        return d

    def write_pdb(self, outname, conformations=[], index=[]):
        '''
        overload superclass method for writing (multi)pdb.

        :param outname: name of pdb file to be generated.
        :param index: indices of atoms to write to file. If empty, all atoms are returned. Index values obtaineable with a call like: index=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, a multipdb with all conformations will be produced.
        '''

        # store current frame, so it will be reestablished after file output is
        # complete
        currentbkp = self.current

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        f_out = open(outname, "w")

        for f in frames:

            # get all informations from PDB (for current conformation) in a
            # list
            self.set_current(f)
            d = self.get_pdb_data(index)

            for i in xrange(0, len(d), 1):
                # create and write PDB line
                L = '%-6s%5s  %-4s%-4s%1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], d[i][1], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                f_out.write(L)

            f_out.write("END\n")

        f_out.close()

        self.set_current(currentbkp)

        return

    def write_gro(self, outname, conformations=[], index=""):
        '''
        write structure(s) in .gro format.

        :param outname: name of .gro file to be generated.
        :param index: indices of atoms to write to file. If empty, all atoms are returned. Index values obtaineable with a call like: index=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, all conformations will be returned.
        '''

        # store current frame, so it will be reestablished after file output is
        # complete
        currentbkp = self.current

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        f_out = open(outname, "w")
        for f in frames:
            f_out.write("%s\n" % outname.split(".")[0])
            f_out.write("%s\n" % len(self.points))
            # get all informations from PDB (for current conformation) in a
            # list
            self.set_current(f)

            # ATOM/HETATM, index, atom name, resname, chain name, residue ID, x,
            # y, z, beta factor, occupancy, atomtype
            d = self.get_pdb_data(index)
            for i in xrange(0, len(d), 1):
                # create and write .gro line
                L = '%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n' % (d[i][5], d[i][3], d[i][2], int(d[i][1]), float(d[i][6]) / 10.0, float(d[i][7]) / 10.0, float(d[i][8]) / 10.0)
                f_out.write(L)

            if "box" in self.properties:
                b = self.properties["box"][f] / 10.0
            else:
                minpos = np.min(self.points, axis=0) / 10.0
                b = np.max(self.points, axis=0) - minpos / 10.0

            f_out.write("%10.5f%10.5f%10.5f\n" % (b[0], b[1], b[2]))

        f_out.close()
        self.set_current(currentbkp)
        return

    def beta_factor_from_rmsf(self, indices=-1, step=1):
        '''
        estimate atoms beta factor on the base of their RMSF.

        :param indices: indices of atoms of interest. If not set all atoms will be considered.
        :param step: timestep between two conformations (useful when using conformations extracted from molecular dynamics)
        '''

        try:
            rmsf = self.rmsf(indices, step)
            return 8.0 * (np.pi**2) * (rmsf**2) / 3.0
        except Exception, ex:
            raise Exception('ERROR: could not calculate RMSF!')

    def rmsf_from_beta_factor(self, indices=[]):
        '''
        calculate RMSF from atoms beta factors.

        :param indices: indices of atoms of interest. If not set all atoms will be considered.
        '''

        try:
            if len(indices) == 0:
                b = self.data["beta"].astype(float)
            else:
                b = self.data["beta"][indices].astype(float)

            return np.sqrt(b * 3 / (8 * np.pi * np.pi))

        except Exception, ex:
            raise Exception('ERROR: beta factors missing?')

    def get_mass_by_residue(self, skip_resname=[]):
        '''
        Compute protein mass using residues (i.e. account also for atoms not present in the structure)

        Sum the average mass all every residue (using a knowledge base of atom masses in Dalton)
        The knowledge base can be expanded or edited by adding entries to the molecule's mass dictionary, e.g. to add the residue "TST" mass in molecule M type: M.knowledge['mass_residue']["TST"]=142.42

        :param skip_resname: list of resnames to skip. Useful to exclude ions water or other ligands from the calculation.
        :returns: mass of molecule in Dalton
        '''
        #@todo mass of N and C termini to add for every chain

        mass = 0
        chains = np.unique(self.data["chain"])
        for chainname in chains:
            # for every chain, get a list of all its (unique) resids
            indices = self.atomselect(chainname, "*", "*", True)[1]
            resids = np.unique(self.data['resid'][indices])

            for r in resids:
                # for every residue in the chain, get the index of its first
                # atom, and extract its associated resname
                index = self.atomselect(chainname, int(r), "*", True)[1]
                resname = self.properties['data'][index[0], 3]

                if resname not in skip_resname:
                    try:
                        # add mass of residue to total mass
                        mass += self.know('residue_mass')[resname]
                    except Exception, ex:
                        #@todo: if residue is not known, why not summing constituent atoms masses, warning the user that it's an estimation?
                        raise Exception("ERROR: mass for resname %s is unknown!\nInsert a key in protein\'s masses dictionary knowledge['residue_mass'] and retry!\nex.: protein.knowledge['residue_mass'][\"TST\"]=142.42" %resname)

        return mass

    def get_mass_by_atom(self, skip_resname=[]):
        '''
        compute protein mass using atoms in pdb

        sum the mass of all atoms (using a knowledge base of atom masses in Dalton)
        The knowledge base can be expanded or edited by adding or editing entries to the molecule's mass dictionary, e.g. to add the atom "PI" mass in molecule M type: M.knowledge['atom_mass']["PI"]=3.141592

        :param skip_resname: list of resnames to skip. Useful to exclude ions water or other ligands from the calculation.
        :returns: mass of molecule in Dalton
        '''

        mass = 0
        for i in xrange(0, len(self.data), 1):
            resname = self.data["resname"][i]
            atomtype = self.data["atomtype"][i]

            if resname not in skip_resname:
                try:
                    mass += self.know('atom_mass')[atomtype]
                except Exception, e:
                    if atomtype == "":
                        raise Exception("ERROR: no atomtype found!")
                    else:
                        raise Exception("ERROR: mass for atom %s is unknown!\nInsert a key in protein\'s masses dictionary knowledge['atom_mass'] and retry!\nex.: protein.knowledge['atom_mass'][\"PI\"]=3.141592" %atomtype)

        return mass

    def s2(self, atomname1="N", atomname2="H"):
        '''
        compute s2, given two atoms defining the vector of interest.

        :param atomname1: name of the first atom
        :param atomname2: name of the second atom
        :returns: data numpy array containing information about residues for which measuring has been performed (i.e.[chain, resid])
        :returns: s2 s2 of residues for which both provided input atoms have been found
        '''

        Nidx = self.atomselect("*", "*", atomname1, get_index=True)[1]
        Hidx = self.atomselect("*", "*", atomname2, get_index=True)[1]

        if len(Nidx) == 0:
            raise Exception("ERROR: no atom name %s found!"%atomname1)

        if len(Hidx) == 0:
            raise Exception("ERROR: no atom name %s found!"%atomname2)

        Ndata = self.data.ix[Nidx, ["chain", "resid"]].values
        Hdata = self.data.ix[Hidx, ["chain", "resid"]].values

        print Ndata

        a1 = []
        a2 = []
        d = []
        for i in xrange(0, len(Ndata), 1):
            j = np.where(
                np.logical_and(
                    Hdata[:, 0] == Ndata[i, 0],
                    Hdata[:, 1] == Ndata[i, 1]))[0]

            if len(j) == 1:
                idx1 = Nidx[i]
                idx2 = Hidx[j[0]]
                a1.append(self.coordinates[:, idx1])
                a2.append(self.coordinates[:, idx2])
                d.append(Ndata[i])

        atoms1 = np.array(a1)
        atoms2 = np.array(a2)
        data = np.array(d)

        # iterate over every residue
        s2_summary = []
        for j in range(atoms1.shape[0]):

            dx = atoms2[j, :, 0] - atoms1[j, :, 0]
            dy = atoms2[j, :, 1] - atoms1[j, :, 1]
            dz = atoms2[j, :, 2] - atoms1[j, :, 2]

            # create list of unit vectors
            dnorm = np.sqrt(dx**2 + dy**2 + dz**2)
            dx /= dnorm
            dy /= dnorm
            dz /= dnorm

            d = 0
            d += (np.sum(dx * dx) / atoms1.shape[1])**2
            d += (np.sum(dx * dy) / atoms1.shape[1])**2
            d += (np.sum(dx * dz) / atoms1.shape[1])**2
            d += (np.sum(dy * dx) / atoms1.shape[1])**2
            d += (np.sum(dy * dy) / atoms1.shape[1])**2
            d += (np.sum(dy * dz) / atoms1.shape[1])**2
            d += (np.sum(dz * dx) / atoms1.shape[1])**2
            d += (np.sum(dz * dy) / atoms1.shape[1])**2
            d += (np.sum(dz * dz) / atoms1.shape[1])**2

            s2_summary.append(0.5 * (3 * d - 1))

        s2 = np.array(s2_summary)

        return data, s2

    def get_couples(self, idx, cutoff):
        '''
        given a list of indices, compute the all-vs-all distance and return only couples below a given cutoff distance

        useful for the detection of disulfide bridges or linkable sites via cross-linking (approximation, supposing euclidean distances)'

        :param idx: indices of atoms to check.
        :param cutoff: minimal distance to consider a couple as linkable.
        :returns: nx3 numpy array containing, for every valid connection, id of first atom, id of second atom and distance between the two.
        '''

        import biobox.measures.interaction as I

        points1 = self.get_xyz()[idx]

        dist = I.distance_matrix(points1, points1)
        couples = I.get_neighbors(dist, cutoff)

        res = []
        for c in couples.transpose():
            if c[0] > c[1]:
                res.append([idx[c[0]], idx[c[1]], dist[c[0], c[1]]])

        return np.array(res)