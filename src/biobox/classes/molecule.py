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
import re
import warnings
from copy import deepcopy
import numpy as np
import scipy.signal
import pandas as pd

# Definiton of constants for later calculations
epsilon0 = 8.8542 * 10**(-12) # m**-3 kg**-1 s**4 A**2, Permitivitty of free space
kB = 1.3806 * 10**(-23) # m**2 kg s**-2 K-1, Lattice Boltzmann constant
e = 1.602 * 10**(-19) # A s, electronic charge
m = 1 * 10**(-9) # number of nm in 1 m
c = 3.336 * 10**(-30) # conversion from debye to e m
Na = 6.022 * 10**(23) # Avogadro's Number

from biobox.classes.structure import Structure
from biobox.lib import e_density

class Molecule(Structure):
    '''
    Subclass of :func:`Structure <biobox.classes.structure.Structure>`, allows reading, manipulating and analyzing molecular structures.
    '''

    chain_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd',
                   'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                   'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    def __init__(self, filename=""):
        '''
        Properties associated to every individual atom are stored in a pandas Dataframe self.data.
        After loading a pdb, pqr, gro or md file, the columns of self.data have the following names:
        atom, index, name, resname, chain, resid, occupancy, beta, atomtype, radius, charge, altloc, icode, formal_charge.
        formal_charge holds integer formal charges, read from columns 79-80 of pdb files and 0 for the other formats.

        self.knowledge contains a knowledge base about atoms and residues properties. Default entries are:

        * 'residue_mass' average mass of the most common amino acids, in Dalton (values from Expasy website)
        * 'atom_vdw' vdw radius of common atoms, in Angstrom
        * 'atom_ccs' radius of common atoms used for CCS calculations, in Angstrom
        * 'atom_mass' mass of common atoms, in Dalton
        * 'atomtype' element associated to common atom names
        * 'AA_mapping' one-letter code of amino acid residue names

        The knowledge base can be edited. For instance, to add information about residue "TST" mass in molecule M type: M.knowledge['residue_mass']["TST"]=142.42

        :param filename: name of a file to load. The file is parsed according to its extension (pdb, pqr, md or gro). If empty (default), an empty Molecule is created.
        '''

        super(Molecule, self).__init__(r=np.array([]))

        # knowledge base about atoms and residues properties (entry keys:
        # 'residue_mass', 'atom_vdw', 'atom_, mass' can be edited)
        self.knowledge = {}
        self.knowledge['residue_mass'] = {"ALA": 71.0788, "ARG": 156.1875, "ASN": 114.1038, "ASP": 115.0886, "CYS": 103.1388, "CYX": 103.1388, "GLU": 129.1155, "GLN": 128.1307, "GLY": 57.0519,
                                          "HIS": 137.1411, "HSE": 137.1411, "HSD": 137.1411, "HSP": 137.1411, "HIE": 137.1411, "HID": 137.1411, "HIP": 137.1411, "ILE": 113.1594, "LEU": 113.1594,
                                          "LYS": 128.1741, "MET": 131.1926, "MSE": 131.1926, "PHE": 147.1766, "PRO": 97.1167, "SER": 87.0782, "THR": 101.1051, "TRP": 186.2132, "TYR": 163.1760, "VAL": 99.1326}
        self.knowledge['atom_vdw'] = {'H': 1.20, 'N': 1.55, 'NA': 2.27, 'CU': 1.40, 'CL': 1.75, 'C': 1.70, 'O': 1.52, 'I': 1.98, 'P': 1.80, 'B': 1.85, 'BR': 1.85, 'S': 1.80, 'SE': 1.90,
                                      'F': 1.47, 'FE': 1.80, 'K': 2.75, 'CA': 2.31, 'MN': 1.73, 'MG': 1.73, 'ZN': 1.39, 'HG': 1.8, 'XE': 1.8, 'AU': 1.8, 'LI': 1.8, '.': 1.8}
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
                                      "SD": "S", "SG": "S", "H": "H", "HA": "H", "HB1": "H", "HB2": "H", "HE1": "H", "HE2": "H", "HD1": "H", "HD2": "H",
                                      "H1": "H", "H2": "H", "H3": "H", "HH11": "H", "HH12": "H", "HH21": "H", "HH22": "H", "HG1": "H", "HG2": "H", "HE21": "H",
                                      "HE22": "H", "HD11": "H", "HD12": "H", "HD13": "H", "HD21": "H", "HD22": "H", "HG11": "H", "HG12": "H", "HG13": "H",
                                      "HG21": "H", "HG22": "H", "HG23": "H", "HZ2": "H", "HZ3": "H", "HZ": "H", "HA1": "H", "HA2": "H", "HB": "H", "HD3": "H",
                                      "HG": "H", "HZ1": "H", "HE3": "H", "HB3": "H", "HH1": "H", "HH2": "H", "HD23": "H", "HD13": "H", "HE": "H", "HH": "H",
                                      "OC1": "O", "OC2": "O", "OW": "O", "HW1": "H", "HW2": "H", "CH3" : "C", "HH31" : "H", "HH32" : "H", "HH33" : "H",
                                      "C00" : "C", "C01" : "C", "C02" : "C", "C04" : "C", "C06" : "C", "C08" : "C", "H03" : "H", "H05" : "H", "H07" : "H",
                                      "H09" : "H", "H0A" : "H", "H0B" : "H", "N01" : "N", "C03": "C", "C05": "C", "O06": "O", "H08": "H", "H0C": "H", "H0D": "H",
                                      "H0E": "H", "H0F": "H", "O03": "O", "H04": "H", "H06": "H", "OD": "O", "O02" : "O", "HO" : "H", "OT" : "O", "O1" : "O", "O2" : "O",
                                      "1H":"H", "2H":"H", "3H":"H", "1HG1":"H", "2HG1":"H", "3HG1":"H", "1HG2":"H", "2HG2":"H", "3HG2":"H", "1HB":"H", "2HB":"H", "1HG":"H", "2HG":"H",
                                      "1HE2":"H", "2HE2":"H", "1HD":"H", "2HD":"H", "1HH1":"H", "2HH1":"H", "1HH2":"H", "2HH2":"H", "1HD1":"H", "1HD2":"H",
                                      "2HD1":"H", "2HD2":"H", "3HD1":"H", "3HD2":"H", "1HZ":"H", "2HZ":"H", "3HZ":"H", "1HE":"H", "2HE":"H", "3HB":"H", "1HA":"H", "2HA":"H",
                                      "3HE":"H", "HN":"H",
                                      "SOD":"NA", "POT":"K", "CLA":"CL", "CAL":"CA", "CES":"CS"}
        self.knowledge['AA_mapping'] = {"GLY": "G", "ALA": "A", "LEU": "L", "MET": "M", "PHE": "F", "TRP": "W", "LYS": "K", "GLN": "Q", "GLU": "E", "SER": "S",
                                        "PRO": "P", "VAL": "V", "ILE": "I", "CYS": "C", "TYR": "Y", "HIS": "H", "ARG": "R", "ASN": "N", "ASP": "D", "THR": "T", "NAN" : "Z",
                                        "MSE": "M", "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
                                        "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K"}

        # if a filename is provided, attempt loading the file according to its file extension
        if filename != "":
            msg = "If you are positive Biobox can read this file, please instantiate Molecule and call the appropriate file parsing method."
            fsplit = os.path.basename(filename).split(".")
            if len(fsplit)<2:
                raise ValueError("Cannot determine file extension. %s"%msg)

            ext = fsplit[-1]
            if ext == "pdb":
                self.import_pdb(filename)
            elif ext == "pqr":
                self.import_pqr(filename)
            elif ext == "md":
                self.import_md(filename)
            elif ext == "gro":
                self.import_gro(filename)
            else:
                raise ValueError("File extension %s unknown. %s"%(ext, msg))


    def __add__(self, other):
        '''
        combine the current conformations of two molecules into a new molecule, via :class:`biobox.classes.multimer.Multimer`.

        Atoms of self come first, followed by those of other. Chain names are kept as they are, so that chains of the two molecules having the same name share it.
        The charge column is kept only if both molecules have it.

        :param other: :class:`biobox.classes.molecule.Molecule` to add to self
        :returns: new :class:`biobox.classes.molecule.Molecule` with a single conformation
        '''
        from biobox.classes.multimer import Multimer
        M = Multimer()
        M.load_list([self, other], ["A", "B"])
        M2 = M.make_molecule()
        return M2

    def addall(self, other, conformations=[]):
        '''
        Like __add__, but instead of adding Molecules self and other at their current frame, include all frames in the addition
        i.e. __add__ changes coordinates from (f, n+m, 3) -> (1, n+m, 3) at the current frame. Here we preserve f.
        Ensure the different subunits have the same number of alternate conformations.
        Use as: N = self.addall(other)

        :param other: Other molecule object to add with self
        :param conformations: List of specific conformations you wish to add together (default == all)
        :returns: New molecule object (as __add__), with one conformation per requested frame
        '''
        current_self = self.current; current_other = other.current

        f = self.coordinates.shape[0]
        if f != other.coordinates.shape[0]:
            raise ValueError("Number of frames need to be identical between two Molecule objects!")

        if len(conformations) == 0:
            start = 0
        else:
            start = conformations[0]

        self.set_current(start); other.set_current(start)
        N = self.__add__(other)

        if len(conformations) == 0:
            for i in range(1, f):
                self.set_current(i); other.set_current(i)
                N2 = self.__add__(other)
                N.add_xyz(N2.points)
        else:
            for c in conformations[1:]:
                self.set_current(c); other.set_current(c)
                N2 = self.__add__(other)
                N.add_xyz(N2.points)

        self.set_current(current_self) # reset to input
        other.set_current(current_other)
        return N

    def know(self, prop):
        '''
        return information from knowledge base

        :param prop: desired property to extract from knowledge base
        :returns: value associated to requested property
        :raises KeyError: if the property is not in the knowledge base
        '''
        if str(prop) in self.knowledge:
            return self.knowledge[str(prop)]
        else:
            raise KeyError("entry %s not found in knowledge base!" % prop)

    def _guess_element(self, name):
        '''
        guess the chemical element of an atom from its PDB atom name field (columns 13-16).

        One-letter elements are right-justified, so " CA " is a carbon, while two-letter elements start in column 13, so "CA  " is a calcium.
        A name starting in column 13 and ending in digits (e.g. "HE21") is a hydrogen, and a name starting with a digit (e.g. "1HD1") takes the element of its second character.

        :param name: atom name field, columns 13-16 of an ATOM or HETATM line
        :returns: element symbol in upper case, or "" if no known element matches
        '''
        name = name.upper().ljust(4)
        elements = self.know('atom_mass')

        stripped = name.strip()
        if stripped == "":
            return ""

        if name[0].isalpha() and not name[2:].strip().isdigit() and stripped in elements:
            return stripped

        if stripped[0].isdigit() and len(stripped) > 1:
            candidate = stripped[1]
        else:
            candidate = stripped[0]

        if candidate in elements:
            return candidate

        return ""

    def _guess_gro_element(self, name, resname):
        '''
        guess the chemical element of an atom from its gro atom and residue names.

        An atom named as its own residue (e.g. NA in residue NA) is a monatomic ion, and names listed in knowledge['atomtype'] take the element given there.
        Any other name takes the element of its first letter after leading digits (e.g. C12 is a carbon, 1HD1 a hydrogen).

        :param name: atom name
        :param resname: residue name
        :returns: element symbol in upper case, or "" if no known element matches
        '''
        name = name.strip().upper()
        resname = resname.strip().upper()
        elements = self.know('atom_mass')

        if name == resname and name in elements:
            return name

        if name in self.know('atomtype'):
            return self.know('atomtype')[name]

        stripped = name.lstrip("0123456789")
        if stripped != "" and stripped[0] in elements:
            return stripped[0]

        return ""

    def import_pdb(self, filename, include_hetatm=False):
        '''
        read a pdb (possibly containing containing multiple models).

        Models are split according to ENDMDL and END statement.
        All alternative coordinates are expected to have the same atoms: if models have different atom counts, only the first is loaded, and a UserWarning is issued.
        After loading, the first model (M.current=0) will be set as active.
        The chain name is column 22, unless the segment identifier (columns 73-76) has two characters, the first of which is that chain: the segment identifier is then the chain name, as written by :func:`write_pdb <biobox.classes.molecule.Molecule.write_pdb>`.
        Formal charges in columns 79-80 (e.g. "2+" or "1-") are loaded in the integer column formal_charge, which is 0 where they are blank. TER records are ignored.

        :param filename: PDB filename
        :param include_hetatm: if True, HETATM will be included (they get skipped if False)
        :raises FileNotFoundError: if the file cannot be opened
        :raises ValueError: if the file content cannot be parsed
        '''

        try:
            f_in = open(filename, "r")
        except Exception:
            raise FileNotFoundError('file %s not found!' % filename)

        # store filename
        self.properties["filename"] = filename

        data_in = []
        alt = []  # alternate location indicators
        ins = []  # insertion codes
        fc = []  # formal charges
        p = []
        r = []
        e = []
        alternative = []
        biomt = {}  # biomolecule id: list of [chains, BIOMT rows]
        biomolecule = None
        symm = []
        for line in f_in:
            record = line[0:6].strip()

            # load biomatrix, if any is present, together with the chains it applies to
            if line.startswith("REMARK 350"):
                text = line[10:].strip()
                if text.startswith("BIOMOLECULE:"):
                    biomolecule = int(text.split(":")[1])
                    biomt[biomolecule] = []

                elif text.startswith("APPLY THE FOLLOWING TO CHAINS:") or text.startswith("AND CHAINS:"):
                    chains = [c.strip() for c in text.split(":")[1].split(",") if c.strip() != ""]
                    if biomolecule is None:
                        biomolecule = 1
                        biomt[biomolecule] = []
                    groups = biomt[biomolecule]
                    # continuation lines extend the chain list of the current group
                    if text.startswith("AND CHAINS:") and len(groups) > 0 and len(groups[-1][1]) == 0:
                        groups[-1][0].extend(chains)
                    else:
                        groups.append([chains, []])

                elif text.startswith("BIOMT"):
                    if biomolecule is None:
                        biomolecule = 1
                        biomt[biomolecule] = []
                    groups = biomt[biomolecule]
                    # matrices given before any chain list apply to all chains
                    if len(groups) == 0:
                        groups.append([None, []])
                    try:
                        groups[-1][1].append(line.split()[4:8])
                    except Exception:
                        raise ValueError("biomatrix format seems corrupted")

            # load symmetry matrix, if any is present
            if "REMARK 290   SMTRY" in line:
                try:
                    symm.append(line.split()[4:8])
                except Exception:
                    raise ValueError("symmetry matrix format seems corrupted")

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
                        cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                        idx = np.arange(len(data))
                        self.data = pd.DataFrame(data, index=idx, columns=cols)
                        # Set the index numbers to the idx values to avoid hexadecimal counts
                        self.data["index"] = idx

                    except Exception:
                        raise ValueError('something went wrong when loading the structure %s!\nare all the columns separated?' %filename)

                    # saving vdw radii
                    try:
                        self.data['radius'] = np.array(r)
                    except Exception:
                        raise ValueError('something went wrong when loading the structure %s!\nare all the columns separated?' %filename)

                    # save default charge state
                    self.data['charge'] = np.array(e)

                # save 3D coordinates of every atom and restart the accumulator
                try:
                    if len(p) > 0:
                        alternative.append(np.array(p))
                    p = []
                except Exception:
                    raise ValueError('something went wrong when loading the structure %s!\nare all the columns separated?' % filename)

            if record == 'ATOM' or (include_hetatm and record == 'HETATM'):

                # extract xyz coordinates (save in list of point coordinates)
                p.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                # if no complete model has been yet parsed, load also
                # information about atoms(resid, resname, ...)
                if len(alternative) == 0:
                    w = []
                    # extract ATOM/HETATM statement
                    w.append(line[0:6].strip())
                    w.append(line[6:12].strip())  # extract atom index
                    w.append(line[12:16].strip())  # extract atomname
                    w.append(line[17:21].strip())  # extract resname
                    w.append(self._parse_pdb_chain(line))  # extract chain name
                    w.append(self._parse_resid(line[22:26]))  # extract residue ID
                    alt.append(line[16].strip())  # extract alternate location indicator
                    ins.append(line[26].strip())  # extract insertion code
                    fc.append(self._parse_formal_charge(line[78:80]))  # extract formal charge

                    # extract occupancy
                    try:
                        w.append(float(line[54:60]))
                    except Exception:
                        w.append(1.0)

                    # extract beta factor
                    try:
                        # w.append("{0.2f}".format(float(line[60:66])))
                        w.append(float(line[60:66]))
                    except Exception:
                        w.append(0.0)

                    # extract atomtype, guessing it from the atom name if the element column is blank
                    element = line[76:78].strip()
                    if element == "":
                        element = self._guess_element(line[12:16])
                    w.append(element)

                    # use atomtype to extract vdw radius
                    try:
                        r.append(self.know('atom_vdw')[element])
                    except Exception:
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
                    cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                    idx = np.arange(len(data))
                    self.data = pd.DataFrame(data, index=idx, columns=cols)
                    # Set the index numbers to the idx values to avoid hexadecimal counts
                    self.data["index"] = idx

                except Exception:
                    raise ValueError('something went wrong when saving data in %s!\nare all the columns separated?' %filename)

                try:
                    self.data['radius'] = np.array(r)
                except Exception:
                    raise ValueError('something went wrong when saving van der Waals radii in %s!\nare all the columns separated?' % filename)

                # save default charge state
                self.data['charge'] = np.array(e)

            # save 3D coordinates of every atom and restart the accumulator
            try:
                if len(p) > 0:
                    alternative.append(np.array(p))
                p = []
            except Exception:
                raise ValueError('something went wrong when saving coordinates in %s!\nare all the columns separated?' %filename)

        # transform the alternative temporary list into a nice multiple
        # coordinates array
        if len(alternative) > 0:
            try:
                alternative_xyz = np.array(alternative).astype(float)
            except Exception:
                alternative_xyz = np.array([alternative[0]]).astype(float)
                warnings.warn('found %s models, but their atom count differs, treating only the first model in file %s' % (len(alternative), filename), stacklevel=2)

            self.add_xyz(alternative_xyz)
        else:
            raise ValueError('something went wrong when saving alternative coordinates in %s!\nno model was loaded... are ENDMDL statements there?' % filename)

        # if biomatrix information is provided, store it as {biomolecule id: [(chains, matrices), ...]}
        if len(biomt) > 0:
            b = {}
            for bm, groups in biomt.items():
                b[bm] = []
                for chains, rows in groups:
                    # test whether there are enough lines to create biomatrix statements
                    if len(rows) == 0 or np.mod(len(rows), 3):
                        raise ValueError('found %s BIOMT entries in biomolecule %s. A multiple of 3 is expected'%(len(rows), bm))

                    mats = np.array(rows).astype(float).reshape((int(len(rows) / 3), 3, 4))
                    b[bm].append((chains, mats))

            self.properties["biomatrix"] = b

        # if symmetry information is provided, create entry in properties
        if len(symm) > 0:

            # test whether there are enough lines to create biomatrix
            # statements
            if np.mod(len(symm), 3):
                raise ValueError('found %s SMTRY entries. A multiple of 3 is expected'%len(symm))

            b = np.array(symm).astype(float).reshape((int(len(symm) / 3), 3, 4))
            self.properties["symmetry"] = b

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["index"] = self.data["index"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)
        self.data["altloc"] = alt
        self.data["icode"] = ins
        self.data["formal_charge"] = np.array(fc, dtype=int)

    def import_md(self, filename):
        '''
        Import a .md structure file, as output by CASTEP, loading one conformation per MD step.

        All atoms are assigned to residue TMP, number 0, of chain X, with occupancy 1.0 and beta factor 0.0, and their atomtype is their element name.
        Their radius is taken from knowledge['atom_vdw'] by element (elements not listed there take the '.' value), their charge and formal charge are 0, and their altloc and icode are empty.

        :param filename: name of the md file
        '''

        import itertools
        energy_skip = 7 # number of lines to skip between blocks of info.
        f_in = open(filename, "r")


        name = []
        label = []
        cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
        line_no = 0
        for line in f_in:
            if line[-2] == "R":
                name.append(line[1:3].replace(" ", ""))
                label.append(line[1:18].replace(" ", ""))
            elif line[-2] == "V":
                break
            line_no += 1

        no_atoms = len(name)
        atom = ["ATOM"] * no_atoms
        resname = ["TMP"] * no_atoms
        chain = ["X"] * no_atoms
        resid = [0] * no_atoms
        occupancy = [1.0] * no_atoms
        beta = [0.0] * no_atoms
        index = np.arange(no_atoms)
        header = line_no - no_atoms
        f_in.close()

        self.data = pd.DataFrame(np.array((np.asarray(atom), index, np.asarray(name), np.asarray(resname), np.asarray(chain), np.asarray(resid), np.asarray(occupancy), np.asarray(beta), np.asarray(name))).T, columns=cols)
        self.data["index"] = self.data["index"].astype(int)
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)

        vdw = self.know('atom_vdw')
        self.data["radius"] = np.array([vdw.get(n.upper(), vdw['.']) for n in name], dtype=float)
        self.data["charge"] = 0.0
        self.data["altloc"] = ""
        self.data["icode"] = ""
        self.data["formal_charge"] = 0

        p = []  # collects coordinates for every model
        coords = []
        with open(filename) as f_in:
            for line in itertools.islice(f_in, header, header+no_atoms):
                p.append([float(line[21:45]), float(line[48:72]), float(line[75:99])])
            coords.append(p)
            for line in itertools.islice(f_in, 0, 2*no_atoms+energy_skip):
                pass
            while len(p) != 0:
                p = []
                for line in itertools.islice(f_in, 0, no_atoms):
                    p.append([float(line[21:45]), float(line[48:72]), float(line[75:99])])
                for line in itertools.islice(f_in, 0, 2*no_atoms+energy_skip):
                    pass
                coords.append(p)

        coords = coords[:-1]
        coords_xyz = np.array(coords).astype(float)
        self.add_xyz(coords_xyz)

    def import_pqr(self, filename, include_hetatm=False):
        '''
        Read a pqr (possibly containing containing multiple models).

        models are split according to ENDMDL and END statement.
        All alternative coordinates are expected to have the same atoms: if models have different atom counts, only the first is loaded, and a UserWarning is issued.
        After loading, the first model (M.current=0) will be set as active.
        Charges are read from columns 55-62 and radii from columns 63-69, the chain name is column 22,
        occupancy is set to 1, beta factor to 0, and atomtype to the first letter of the atom name.

        :param filename: PQR filename
        :param include_hetatm: if True, HETATM will be included (they get skipped if False)
        :raises FileNotFoundError: if the file cannot be opened
        :raises ValueError: if the file content cannot be parsed
        '''

        try:
            f_in = open(filename, "r")
        except Exception:
            raise FileNotFoundError('file %s not found!' % filename)

        # store filename
        self.properties["filename"] = filename

        data_in = []
        alt = []  # alternate location indicators
        ins = []  # insertion codes
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
                        cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                        idx = np.arange(len(data))
                        self.data = pd.DataFrame(data, index=idx, columns=cols)
                        self.data["index"] = idx # convert to internal numbering system

                    except Exception:
                        raise ValueError('something went wrong when loading the structure %s!\nare all the columns separated?' %filename)

                    # saving vdw radii
                    try:
                        self.data['radius'] = np.array(r)
                    except Exception:
                        raise ValueError('something went wrong when loading the structure %s!\nare all the columns separated?' %filename)

                    # saving electrostatics
                    try:
                        self.data['charge'] = np.array(e)
                    except Exception:
                        raise ValueError('something went wrong when loading the structure %s!\nare all the columns separated?' % filename)

                # save 3D coordinates of every atom and restart the accumulator
                try:
                    if len(p) > 0:
                        alternative.append(np.array(p))
                    p = []
                except Exception:
                    raise ValueError('something went wrong when loading the structure %s!\nare all the columns separated?' %filename)

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
                    except Exception:
                        e.append(0.0)

                    # extract vdW radius
                    try:
                        r.append(float(line[62:69]))
                    except Exception:
                        r.append(self.know('atom_vdw')['.'])

                    # initialize list
                    w = []

                    # extract ATOM/HETATM statement
                    w.append(line[0:6].strip())
                    w.append(line[6:11].strip())  # extract atom index
                    w.append(line[12:16].strip())  # extract atomname
                    w.append(line[17:21].strip())  # extract resname
                    w.append(line[21].strip())  # extract chain name
                    w.append(self._parse_resid(line[22:26]))  # extract residue ID
                    alt.append(line[16].strip())  # extract alternate location indicator
                    ins.append(line[26].strip())  # extract insertion code

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
                    cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                    idx = np.arange(len(data))
                    self.data = pd.DataFrame(data, index=idx, columns=cols)
                    self.data["index"] = idx # convert to internal numbering system

                except Exception:
                    raise ValueError('something went wrong when saving data in %s!\nare all the columns separated?' % filename)

                try:
                    self.data['radius'] = np.array(r)
                except Exception:
                    raise ValueError('something went wrong when saving van der Waals radii in %s!\nare all the columns separated?' %filename)

                try:
                    self.data['charge'] = np.array(e)
                except Exception:
                    raise ValueError('something went wrong when saving charges in %s!\nare all the columns separated?' %filename)

            # save 3D coordinates of every atom and restart the accumulator
            try:
                if len(p) > 0:
                    alternative.append(np.array(p))
                p = []
            except Exception:
                raise ValueError('something went wrong when saving coordinates in %s!\nare all the columns separated?' %filename)

        # transform the alternative temporary list into a nice multiple
        # coordinates array
        if len(alternative) > 0:
            try:
                alternative_xyz = np.array(alternative).astype(float)
            except Exception:
                alternative_xyz = np.array([alternative[0]]).astype(float)
                warnings.warn('found %s models, but their atom count differs, treating only the first model in file %s' % (len(alternative), filename), stacklevel=2)

            self.add_xyz(alternative_xyz)
        else:
            raise ValueError('something went wrong when saving alternative coordinates in %s!\nno model was loaded... are ENDMDL statements there?' % filename)

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["index"] = self.data["index"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)
        self.data["altloc"] = alt
        self.data["icode"] = ins
        self.data["formal_charge"] = 0

    def import_gro(self, filename):
        '''
        read a gro possibly containing multiple structures.

        Any data already in the molecule is cleared first. Coordinates and box sizes are converted from nm to Angstrom,
        and the box of every frame is stored in properties['box']. All atoms are assigned to chain A, and their atomtype is guessed from atom and residue names.

        :param filename: name of .gro file to import
        :raises FileNotFoundError: if the file does not exist
        '''

        if not os.path.isfile(filename):
            raise FileNotFoundError("%s not found!" % filename)

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
                w = fin.readline()
                # Read array as defined by .gro style characters (res int, res, atomtype, int, x_coord, y_coord, z_coord)
                w = [w[0:5].strip(), w[5:10].strip(), w[10:15].strip(), w[15:20].strip(), w[20:28].strip(), w[28:36].strip(), w[36:44].strip()]
                resname = w[1]; resnumber=w[0]

                # read data useful for indexing (guess what is missing)
                d_data.append(["ATOM", w[3], w[2], resname, "A", resnumber, "1.0", "0.0", self._guess_gro_element(w[2], resname)])
                d.append([w[4], w[5], w[6]])
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
        cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
        idx = np.arange(len(data))
        self.data = pd.DataFrame(data, index=idx, columns=cols)
        self.data["index"] = idx # convert to internal numbering system

        #add additional information about van der waals radius and atoms charge
        vdw = self.know('atom_vdw')
        self.data['radius'] = [vdw.get(a, vdw['.']) for a in self.data['atomtype']]
        self.data['charge'] = np.zeros(len(d_data))

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)
        self.data["altloc"] = ""
        self.data["icode"] = ""
        self.data["formal_charge"] = 0

        fin.close()

    def assign_atomtype(self):
        '''
        guess atomtype from atom names, using knowledge['atomtype'], and overwrite the atomtype column of all atoms. Atoms with an unknown name get an empty atomtype.
        '''

        a_type = []
        for i in range(0, len(self.data), 1):
            atom = self.data["name"].values[i]
            try:
                a_type.append(self.knowledge["atomtype"][atom])
            except Exception:
                a_type.append("")

        self.data["atomtype"] = a_type

    def get_vdw_density(self, buff=3, step=0.5, kernel_half_width=10):
        '''
        generate density map of all atoms in the current conformation, convolving each atom with a gaussian kernel whose sigma depends on its atomtype (C, H, O, S or N, see _vdw_density_on_grid).

        :param buff: padding to add at points cloud boundaries, in Angstrom
        :param step: size of cubic voxels, in Angstrom
        :param kernel_half_width: kernel half width, in voxels
        :returns: :func:`Density <biobox.classes.density.Density>` object, containing a density map
        '''
        axes = self._grid_axes(self.points, step, buff)
        dens = self._vdw_density_on_grid(np.arange(len(self.points)), axes, step, kernel_half_width)

        from biobox.classes.density import Density
        D = Density()
        D.properties['density'] = dens
        D.properties['size'] = np.array(dens.shape)
        D.properties['origin'] = np.array([ax[0] for ax in axes])
        D.properties['delta'] = np.identity(3) * step
        D.properties['format'] = 'dx'
        D.properties['filename'] = ''
        D.properties["sigma"] = np.std(dens)

        return D

    def _vdw_density_on_grid(self, indices, axes, step, kernel_half_width):
        '''
        sum of the density maps of each atom type, each built from the atoms of that type on a common grid.

        Only atomtypes C, H, O, S and N contribute, each with its own gaussian sigma (in voxels), and each map is scaled to a maximum of 1 before summing.
        Selected atoms with an empty atomtype get the one given by knowledge['atomtype'] for their name or, for names not listed there, the element guessed from their name.
        Their atomtype is stored in self.data, while non-empty atomtypes and atoms outside the selection are left unchanged. A KeyError is raised if some atomtype is still unknown.

        :param indices: indices of atoms to include
        :param axes: grid axes, as returned by _grid_axes
        :param step: size of cubic voxels, in Angstrom
        :param kernel_half_width: kernel half width, in voxels
        :returns: 3D numpy array
        '''
        atomdata = [["C", 1.7, 1.455, 0.51], ["H", 1.2, 0.72, 0.25],
                    ["O", 1.52, 1.15, 0.42], ["S", 1.8, 1.62, 0.54],
                    ["N", 1.55, 1.2, 0.44]]

        # fill in empty atomtypes of selected atoms from their name, and test if successful
        indices = np.asarray(indices)
        blank = indices[self.data["atomtype"].values[indices] == '']
        if len(blank) > 0:
            known = self.know('atomtype')
            names = self.data["name"].values[blank]
            guessed = [known[n] if n in known else self._guess_element(n) for n in names]
            self.data.iloc[blank, self.data.columns.get_loc("atomtype")] = guessed

        atomtypes = self.data["atomtype"].values[indices]
        if np.any(atomtypes == ''):
            raise KeyError("Unknown atomtype for:\n%s"%self.data.iloc[indices[atomtypes == '']])

        pts = self.points[indices]
        dens = np.zeros([len(ax) for ax in axes])
        for d in atomdata:
            # use the standard density calculation with an atom-type specific sigma value
            sel = atomtypes == d[0]
            if np.any(sel):
                dens += self._density_on_grid(pts[sel], axes, step, d[2], kernel_half_width)

        return dens

    def get_electrostatics(self, step=1.0, buff=3, threshold=0.01, vdw_kernel_half_width=5, elect_kernel_half_width=12, chain='*', clear_mass=True):
        '''
        generate electrostatic potential maps of the current conformation, convolving the atomic charges (charge column) with a Coulomb kernel k/r, set to zero within 0.9 Angstrom of its centre.

        The grid encloses the atoms of the selected chains, while the mass density is built from all atoms falling on the grid.

        :param step: size of cubic voxels, in Angstrom
        :param buff: padding to add at points cloud boundaries, in Angstrom
        :param threshold: mass density value above which a voxel is considered occupied by atoms (see clear_mass)
        :param vdw_kernel_half_width: half width of the kernel used for the mass density, in voxels
        :param elect_kernel_half_width: half width of the Coulomb kernel, in Angstrom
        :param chain: select chain to use (accepts * as wildcard, or a list of chain names), default all chains
        :param clear_mass: if True, set the potential to zero where the mass density exceeds threshold
        :returns: :func:`Density <biobox.classes.density.Density>` object of the positive potential (negative values set to zero)
        :returns: :func:`Density <biobox.classes.density.Density>` object of the negative potential, sign inverted so that its values are positive (positive values set to zero)
        :returns: :func:`Density <biobox.classes.density.Density>` object of the mass density, on the same grid
        '''

        pts, idx = self.atomselect(chain, '*', '*', get_index=True)

        try:
            # numpy array of charges [c1, c2, c3, ...]
            charges = self.data['charge'].values[idx].astype(float)
        except Exception:
            raise KeyError('No charges associated with %s' % self)

        k = 8.9875517873681764  # Coulomb's constant in nN

        # rectangular box enclosing the selected atoms, shared by all maps
        axes = self._grid_axes(pts, step, buff)
        origin = np.array([ax[0] for ax in axes])

        # place Kronecker deltas in mesh grid, summing the charges falling in the same voxel
        d = np.zeros([len(ax) for ax in axes])
        np.add.at(d, self._grid_indices(pts, axes, step), charges)

        # 3d kernel centred on its middle voxel, reaching elect_kernel_half_width Angstrom
        half = int(round(elect_kernel_half_width / step))
        r = np.arange(-half, half + 1) * step
        x, y, z = np.meshgrid(r, r, r, indexing='ij')
        distance = np.sqrt(x * x + y * y + z * z)
        kernel = np.zeros(distance.shape)
        # a hyperbola is created only outside of H-vdW and inside of relevant kernel-half-width distance
        outside = distance > 0.9
        kernel[outside] = k / distance[outside]

        # convolve point mesh with 3d coulomb hyperbola
        e = scipy.signal.fftconvolve(d, kernel, mode='same')

        # define mass-occupied space of all atoms, on the same grid. The grid is padded by the
        # kernel width, so that atoms just outside it still contribute their density, then cropped
        from biobox.classes.density import Density
        pad = vdw_kernel_half_width
        padded = [np.concatenate((ax[0] - step * np.arange(pad, 0, -1), ax, ax[-1] + step * np.arange(1, pad + 1))) for ax in axes]
        lo = np.array([ax[0] for ax in padded]) - step / 2.0
        hi = np.array([ax[-1] for ax in padded]) + step / 2.0
        inside = np.where(np.all((self.points >= lo) & (self.points <= hi), axis=1))[0]
        dens = self._vdw_density_on_grid(inside, padded, step, vdw_kernel_half_width)[pad:-pad, pad:-pad, pad:-pad]
        mass_density = Density()
        mass_density.properties['density'] = dens
        mass_density.properties['size'] = np.array(dens.shape)
        mass_density.properties['origin'] = origin
        mass_density.properties['delta'] = np.identity(3) * step
        mass_density.properties['format'] = 'dx'
        mass_density.properties['filename'] = ''
        mass_density.properties["sigma"] = np.std(dens)

        if clear_mass:
            occupied = dens > threshold
            e[occupied] = 0

        # split the density into two maps
        e_pos = deepcopy(e)
        e_neg = deepcopy(e)
        e_pos[np.where(e_pos < 0)] = 0
        e_neg[np.where(e_neg >= 0)] = 0
        # changes sign of negative array for better visualization
        e_neg = np.negative(e_neg)

        # prepare density data structure for both positive and negative maps at
        # once
        D_pos = Density()
        D_neg = Density()
        D_pos.properties['density'] = e_pos
        D_neg.properties['density'] = e_neg
        D_pos.properties['size'] = D_neg.properties['size'] = np.array(e.shape)
        D_pos.properties['origin'] = origin
        D_neg.properties['origin'] = origin.copy()
        D_pos.properties['delta'] = D_neg.properties['delta'] = np.identity(3) * step
        D_pos.properties['format'] = D_neg.properties['format'] = 'dx'
        D_pos.properties['filename'] = D_neg.properties['filename'] = ''
        D_pos.properties["sigma"] = np.std(e_pos)
        D_neg.properties["sigma"] = np.std(e_neg)

        return D_pos, D_neg, mass_density

    def _apply_matrices(self, groups):
        '''
        build a molecule made of copies of chains of this molecule, each transformed as x' = Rx + t.

        The first copy of a chain keeps its name, and later copies take chain names not used in this molecule, with one character first and two characters once those run out.
        Only the current conformation is transformed, and the new molecule holds a single conformation.

        :param groups: list of (chains, matrices) pairs, where chains is a list of chain names (None for all chains) and matrices is an array of 3x4 [R|t] matrices
        :returns: new Molecule
        '''

        # chain names available for copies, in order, single characters first
        used = set(self.data["chain"].values)
        single = list(dict.fromkeys(self.chain_names))
        double = [a + b for a in single for b in single]
        available = [c for c in single + double if c not in used]

        copies = []
        for chains, mats in groups:
            present = used if chains is None else used.intersection(chains)
            copies.extend(list(present) * len(mats))
        needed = len(copies) - len(set(copies))
        if needed > len(available):
            raise ValueError("the transformed molecule needs %s new chain names, but only %s are available" % (needed, len(available)))

        named = set()
        data = []
        xyz = []
        for chains, mats in groups:
            if chains is None:
                idx = np.arange(len(self.data))
            else:
                idx = np.where(np.isin(self.data["chain"].values, chains))[0]

            if len(idx) == 0:
                continue

            for m in mats:
                xyz.append(np.dot(self.points[idx], m[:, 0:3].T) + m[:, 3])

                d = self.data.iloc[idx].copy()
                names = {}
                for c in pd.unique(d["chain"].values):
                    if c not in named:
                        names[c] = c
                        named.add(c)
                    else:
                        names[c] = available.pop(0)

                d["chain"] = d["chain"].map(names)
                data.append(d)

        if len(data) == 0:
            raise ValueError("none of the chains the matrices apply to is in the molecule")

        M = Molecule()
        M.knowledge = deepcopy(self.knowledge)
        M.data = pd.concat(data, ignore_index=True)
        M.data["index"] = np.arange(len(M.data))
        M.add_xyz(np.concatenate(xyz, axis=0))

        return M

    def apply_biomatrix(self, biomolecule=1):
        '''
        if biomatrix information is provided, generate a new molecule with the biological assembly described by REMARK 350.

        Each BIOMT operator is applied only to the chains listed for it, and chains not listed are left out, as in the assembly files of the PDB.
        Only the current conformation is transformed. The first copy of a chain keeps its name, and later copies take chain names not used in the molecule.

        :param biomolecule: id of the BIOMOLECULE to build (default 1)
        :returns: new Molecule containing the transformed copies of the chains, arranged according to the BIOMT statements of the requested biomolecule
        '''

        # if no biomatrix statement is found, return with error
        if "biomatrix" not in self.properties:
            raise KeyError("no biomatrix found in pdb %s" %self.properties["filename"])

        if biomolecule not in self.properties["biomatrix"]:
            raise KeyError("biomolecule %s not found, available: %s" %(biomolecule, sorted(self.properties["biomatrix"])))

        return self._apply_matrices(self.properties["biomatrix"][biomolecule])

    def apply_symmetry(self):
        '''
        if symmetry information is provided, generate a new molecule with all symmetry operators applied to all chains.

        Only the current conformation is transformed. The first copy of a chain keeps its name, and later copies take chain names not used in the molecule.

        :returns: new Molecule containing several copies of the current conformation, arranged according to SMTRY statements contained in pdb
        '''

        # if no symmetry statement is found, return with error
        if "symmetry" not in self.properties:
            raise KeyError("no symmetry matrix found in pdb %s" %self.properties["filename"])

        return self._apply_matrices([(None, self.properties["symmetry"])])

    def get_atoms_ccs(self):
        '''
        return the atomic radii used for CCS calculations of every atom in molecule, assigned by atomtype from knowledge['atom_ccs'] (atomtypes not listed there take the '.' value).

        The radii are stored in the atom_ccs column of self.data, and returned from there by later calls.

        :returns: numpy array with the radius of every atom, in Angstrom
        '''

        if "atom_ccs" in self.data.columns:
            return np.array(self.data["atom_ccs"].values)

        ccs = np.ones(len(self.points)) * self.know("atom_ccs")["."]
        for e in self.know("atom_ccs").keys():
            if e != ".":
                ccs[self.data["atomtype"].values == e] = self.knowledge["atom_ccs"][e]

        self.data["atom_ccs"] = ccs

        return ccs

    def get_data(self, indices=[], columns=[]):
        '''
        Return information about atoms of interest (i.e., slice the data DataFrame)

        :param indices: list of indices, if not provided all atom data is returned
        :param columns: list of columns (e.g. ["resname", "resid", "chain"]), if not provided all columns are returned
        :returns: numpy array containing a slice of molecule's data
        '''

        if len(indices) == 0 and len(columns) == 0:
            return self.data.values

        elif len(indices) == 0 and len(columns) != 0:
            return self.data[columns].values

        elif len(indices) != 0 and len(columns) == 0:
            return self.data.loc[indices].values

        else:
            return self.data.loc[indices, columns].values


    def set_data(self, value, indices=[], columns=[]):
        '''
        Set information about atoms of interest (i.e., assign values to a slice of the data DataFrame)

        :param value: value(s) to assign to the selected slice
        :param indices: list of indices, if not provided all atoms are modified
        :param columns: list of columns (e.g. ["resname", "resid", "chain"]), if not provided all columns are modified. Indices, columns or both must be provided.
        '''

        if len(indices) == 0 and len(columns) == 0:
            raise ValueError("indices, columns or both should be provided")

        elif len(indices) == 0 and len(columns) != 0:
            self.data[columns] = value

        elif len(indices) != 0 and len(columns) == 0:
            self.data.loc[indices] = value

        else:
            self.data.loc[indices, columns] = value


    def query(self, query_text, get_index=False):
        '''
        Select specific atoms in the molecule on the basis of a text query.

        :param query_text: string selecting atoms of interest. Uses the pandas query syntax, can access all columns in the dataframe self.data.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: coordinates of the selected points in the current conformation and, if get_index is set to true, a list [coordinates, indices] containing also their indices in self.points array.
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
        :param res: residue ID of desired atoms (accepts * as wildcard). Can also be a list or numpy array of of int. A residue number selects all its insertion codes, while a string such as "52A" selects only that insertion code.
        :param atom: name of desired atom (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the selected points in the current conformation and, if get_index is set to true, a list [coordinates, indices] containing also their indices in self.points array.
        '''

        # chain name boolean selector
        if isinstance(chain, str):
            if chain == '*':
                chain_query = np.array([True] * len(self.points))
            else:
                chain_query = self.data["chain"].values == chain

        elif isinstance(chain, list) or type(chain).__module__ == 'numpy':
            chain_query = self.data["chain"].values == chain[0]
            for c in range(1, len(chain), 1):
                chain_query = np.logical_or(chain_query, self.data["chain"].values == chain[c])
        else:
            raise TypeError("wrong type for chain selection. Should be str, list, or numpy")

        # residue boolean selector
        if isinstance(res, np.generic):
            res = res.item()

        if isinstance(res, str) and res == '*':
            res_query = np.array([True] * len(self.points))

        else:
            if isinstance(res, (str, int)):
                res = [res]
            elif not isinstance(res, (list, tuple, range, np.ndarray)):
                raise TypeError("wrong type for resid selection. Should be int, list, or numpy")

            if use_resname:
                res_query = np.isin(self.data["resname"].values, [str(r) for r in res])
            else:
                # a residue number alone selects all its insertion codes, "52A" selects only 52A
                parsed = [self._as_resid(r) for r in res]
                res_query = np.isin(self.data["resid"].values, [r for r, ic in parsed if ic is None])
                icode = self._column_or_blank("icode")
                for r, ic in parsed:
                    if ic is not None:
                        res_query = np.logical_or(res_query, np.logical_and(self.data["resid"].values == r, icode == ic))

        # atom name boolean selector
        if isinstance(atom, str):
            if atom == '*':
                atom_query = np.array([True] * len(self.points))
            else:
                atom_query = self.data["name"].values == atom
        elif isinstance(atom, list) or type(atom).__module__ == 'numpy':
            atom_query = self.data["name"].values == atom[0]
            for a in range(1, len(atom), 1):
                atom_query = np.logical_or(atom_query, self.data["name"].values == atom[a])
        else:
            raise TypeError("wrong type for atom selection. Should be str, list, or numpy")

        # slice data array and return result (colums 5 to 7 contain xyz coords)
        query = np.logical_and(np.logical_and(chain_query, res_query), atom_query)


        if get_index:
            return [self.points[query], np.where(query == True)[0]]
        else:
            return self.points[query]

    def _as_resid(self, res):
        '''
        convert a residue ID given as an int, a numpy scalar or a string such as "52" or "52A" into a residue number and an insertion code.

        :param res: residue ID
        :returns: residue number as int, and insertion code (None if not given)
        '''
        if isinstance(res, np.generic):
            res = res.item()

        if isinstance(res, str):
            match = re.match(r"^\s*(-?\d+)([A-Za-z]?)\s*$", res)
            if match is None:
                raise ValueError("resid %s is not an integer. To select by residue name, set use_resname=True" % res)
            return int(match.group(1)), (match.group(2) if match.group(2) != "" else None)

        return res, None

    @staticmethod
    def _parse_resid(text):
        '''
        read the residue number field of a PDB line (columns 23-26), in decimal or in hybrid-36.

        :param text: residue number field
        :returns: residue number as int
        '''
        text = text.strip()
        try:
            return int(text)
        except ValueError:
            pass

        try:
            value = int(text, 36) - 10 * 36**3 + 10**4
            if text[0].islower():
                value += 26 * 36**3
            return value
        except ValueError:
            raise ValueError("cannot read residue number %s" % text)

    def _column_or_blank(self, column):
        '''
        values of a text column of self.data, with missing values (or a missing column) as empty strings.

        :param column: column name
        :returns: numpy array of strings
        '''
        if column not in self.data.columns:
            return np.array([""] * len(self.data), dtype=object)
        return self.data[column].fillna("").astype(str).values

    def _formal_charges(self):
        '''
        formal charge of every atom, with missing values (or a missing formal_charge column) as 0.

        :returns: numpy array of integers
        '''
        if "formal_charge" not in self.data.columns:
            return np.zeros(len(self.data), dtype=int)
        return self.data["formal_charge"].fillna(0).values.astype(int)

    def _one_per_residue(self, indices):
        '''
        keep the first of the given atoms in every residue, identified by chain, residue number and insertion code.

        :param indices: atom indices
        :returns: list of atom indices
        '''
        chain = self.data["chain"].values
        resid = self.data["resid"].values
        icode = self._column_or_blank("icode")
        seen = set()
        keep = []
        for i in indices:
            key = (chain[i], resid[i], icode[i])
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return keep

    def _residue_starts(self):
        '''
        mark the first atom of every residue. A residue is a contiguous run of atoms sharing chain, residue number,
        insertion code and residue name, in which no atom name (with its alternate location) appears twice, so that two
        adjacent residues with the same number are told apart.

        :returns: boolean numpy array, one element per atom
        '''
        key = list(zip(self.data["chain"].values, self.data["resid"].values, self._column_or_blank("icode"), self.data["resname"].values))
        atom = list(zip(self.data["name"].values, self._column_or_blank("altloc")))
        starts = np.zeros(len(key), dtype=bool)
        seen = set()
        for i in range(len(key)):
            if i == 0 or key[i] != key[i - 1] or atom[i] in seen:
                starts[i] = True
                seen = set()
            seen.add(atom[i])
        return starts

    def atomignore(self, chain, res, atom, get_index=False, use_resname=False):
        '''
        Select specific atoms that do not match a specific query (chain, residue ID and atom name).
        Useful to remove from a molecule atoms unwanted for further analysis, alternative conformations, etc...

        :param chain: chain name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param res: residue ID (accepts * as wildcard). Can also be a list or numpy array of of int. Residue IDs are interpreted as in :func:`atomselect <biobox.classes.molecule.Molecule.atomselect>`.
        :param atom: atom name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of atoms in self.points array (and self.data)
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the points not matching the query in the current conformation and, if get_index is set to true, a list [coordinates, indices] containing also their indices in self.points array.
        '''

        #extract indices of atoms matching the query
        idxs = self.atomselect(chain, res, atom, get_index=True, use_resname=use_resname)[1]

        #invert the selection
        idxs2 = []
        for i in range(len(self.points)):
            if i not in idxs:
                idxs2.append(i)

        if get_index:
            return [self.points[idxs2], np.array(idxs2)]
        else:
            return self.points[idxs2]

    def same_residue(self, indices, get_index=False):
        '''
        Select all atoms belonging to the same residue (same chain, residue number and insertion code) as a given atom (or list of atoms)

        :param indices: indices of atoms of choice (integer or list of integers)
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: coordinates of the selected points in the current conformation (an empty list if none is found) and, if get_index is set to true, also their indices in self.points array.
        '''

        chain = self.data["chain"].values
        resid = self.data["resid"].values
        icode = self._column_or_blank("icode")
        indices = np.atleast_1d(indices)

        test = np.zeros(len(self.data), dtype=bool)
        for c, r, ic in set(zip(chain[indices], resid[indices], icode[indices])):
            test = np.logical_or(test, (chain == c) & (resid == r) & (icode == ic))

        idxs = np.where(test)[0]
        if len(idxs) > 0:
            pts = self.points[idxs]
        else:
            pts = []

        if get_index:
            return pts, idxs
        else:
            return pts

    def same_residue_unique(self, indices, get_index=False):
        '''
        Select atoms having the same residue (chain, residue number and insertion code) as a given atom (or list of atoms),
        considering only the contiguous run of atoms around the given atom in file order. Each atom is returned once.

        :param indices: indices of atoms of choice (integer or list of integers)
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: numpy array of coordinates of the selected points in the current conformation and, if get_index is set to true, also a numpy array of their indices in self.points array.
        '''

        try:
            test = len(indices)  # this should fail if indices is a number
            idlist = indices
        except Exception:
            idlist = [indices]

        # residues are identified by chain, residue number and insertion code
        key = list(zip(self.data["chain"].values, self.data["resid"].values, self._column_or_blank("icode")))
        pts = []
        idxs = []
        for i in idlist:
            done = False
            j = 0  # starting from same point
            while not done:

                if i - j < 0:
                    done = True

                elif key[i] == key[i - j]:

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

                elif key[i] == key[i + j]:

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

    def get_subset(self, indices, conformations=[], flip = False):
        '''
        Return a :func:`Molecule <biobox.classes.molecule.Molecule>` object containing only the selected atoms and frames

        :param indices: indices of atoms to extract, or boolean mask with one element per atom
        :param conformations: frames to extract (by default, all)
        :param flip: If true, extract atoms that DON'T match indices (default is False)
        :returns: :func:`Molecule <biobox.classes.molecule.Molecule>` object, with its current conformation set to the first extracted frame
        '''

        indices = np.asarray(indices)
        if indices.dtype == bool:
            indices = np.where(indices)[0]
        elif len(indices) == 0:
            indices = indices.astype(int)

        if flip:
            self_index = set(self.data["index"])
            idxs_flip = set(indices)
            indices = np.asarray(list(self_index - idxs_flip) + list(idxs_flip - self_index)) # replace indices with new keep list

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise IndexError("requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        idx = np.arange(len(indices))

        # create molecule, and push created data information
        M = Molecule()
        postmp = self.coordinates[:, indices]
        M.coordinates = postmp[frames]
        M.data = self.data.loc[indices]
        M.data = M.data.reset_index(drop=True)
        M.data["index"] = idx
        M.current = 0
        #M.points = M.coordinates[M.current]
        M.points = M.coordinates.view()[M.current]

        M.properties['center'] = M.get_center()

        return M

    def guess_chain_split(self, distance=3, use_backbone=True):
        '''
        reassign chain name, using distance cutoff (cannot be undone).
        If two consecutive atoms (or residues, see use_backbone) are beyond a cutoff, a new chain is assigned. Chains are named in the order of chain_names. Distances are measured in the current conformation.

        :param distance: distance cutoff, in Angstrom
        :param use_backbone: if True, a new chain starts at a residue whose N is farther than the cutoff from the C of the closest preceding residue having one (residues without N, e.g. ACE, ligands or water, never start a chain). If False, consecutive atoms in the sequence are compared
        :returns: number of chains
        :returns: list of indices of the first atom of each chain, followed by the number of atoms
        :returns: numpy array of the N-C distances, in Angstrom and rounded to 3 decimals, at which a new chain starts (empty if use_backbone is False)
        '''

        # identify different chains
        intervals = [0]

        gaps = []
        if not use_backbone:
            for i in range(len(self.coordinates[0]) - 1):
                dist = np.sqrt(np.dot(self.points[i] - self.points[i + 1], self.points[i] - self.points[i + 1]))
                if dist > distance:
                    intervals.append(i + 1)

        else:
            names = self.data["name"].values
            starts = np.flatnonzero(self._residue_starts())
            ends = np.r_[starts[1:], len(names)]
            last_C = None
            for a, b in zip(starts, ends):
                n = np.flatnonzero(names[a:b] == "N")
                if len(n) > 0 and last_C is not None:
                    dist = np.linalg.norm(self.points[last_C] - self.points[a + n[0]])
                    if dist > distance:
                        intervals.append(a)
                        gaps.append(dist)
                c = np.flatnonzero(names[a:b] == "C")
                if len(c) > 0:
                    last_C = a + c[0]

        intervals.append(len(self.coordinates[0]))

        # separate chains
        chain = np.empty(len(self.data), dtype=object)
        for i in range(len(intervals) - 1):
            thepos = i % len(self.chain_names)
            chain[intervals[i]:intervals[i + 1]] = self.chain_names[thepos]
        self.data["chain"] = chain

        return len(intervals) - 1, intervals, np.round(np.array(gaps), decimals=3)

    def get_pdb_data(self, indices=[]):
        '''
        aggregate data and point coordinates of the current conformation, and return in a unique data structure

        Returned data contains, for every atom, its data and coordinates
        in the same order as a pdb file, i.e.
        ATOM/HETATM, index, name, resname, chain name, residue ID, x, y, z, occupancy, beta factor, atomtype, alternate location, insertion code.

        :param indices: indices of atoms of interest. If empty (default), all atoms are returned.
        :returns: list containing, for every atom, a list of its 14 fields (values keep the type of the corresponding data column, coordinates are floats).
        '''

        if len(indices) == 0:
            indices = range(0, len(self.points), 1)

        altloc = self._column_or_blank("altloc")
        icode = self._column_or_blank("icode")

        # create a list containing all infos contained in pdb (point
        # coordinates and properties)
        d = []
        for i in indices:
            d.append([self.data["atom"].values[i],
                      self.data["index"].values[i],
                      self.data["name"].values[i],
                      self.data["resname"].values[i],
                      self.data["chain"].values[i],
                      self.data["resid"].values[i],
                      self.points[i, 0],
                      self.points[i, 1],
                      self.points[i, 2],
                      self.data["occupancy"].values[i],
                      self.data["beta"].values[i],
                      self.data["atomtype"].values[i],
                      altloc[i],
                      icode[i]])

        return d

    @staticmethod
    def _pdb_resid(resid):
        '''
        residue number as written in the 4 columns of a PDB line, keeping its last 4 digits if it does not fit.

        :param resid: residue number
        :returns: residue number that fits 4 columns
        '''
        resid = int(resid)
        if -999 <= resid <= 9999:
            return resid
        return int(str(resid)[-4:])

    @staticmethod
    def _parse_pdb_chain(line):
        '''
        chain name of an ATOM or HETATM line.

        A two-character segment identifier (columns 73-76) whose first character is the chain in column 22 holds the full chain name.

        :param line: ATOM or HETATM line
        :returns: chain name
        '''
        chain = line[21].strip()
        segid = line[72:76].strip()
        if len(segid) == 2 and segid[0] == chain:
            return segid
        return chain

    @staticmethod
    def _pdb_chain_segid(chain):
        '''
        split a chain name into the chain column (22) and the segment identifier (columns 73-76) of a PDB line.

        A one-character chain has an empty segment identifier. A two-character chain is written as its first character, and in full as segment identifier.

        :param chain: chain name, of one or two characters
        :returns: chain column
        :returns: segment identifier
        '''
        if len(chain) <= 1:
            return chain, ""
        if len(chain) == 2:
            return chain[0], chain
        raise ValueError("chain name %s is longer than two characters, which the PDB format cannot hold" % chain)

    @staticmethod
    def _pdb_atom_prefix(record, serial, name, resname, chain, resid, altloc="", icode=""):
        '''
        first 30 columns of an ATOM or HETATM line, up to the x coordinate, following the PDB format.

        Atom names of 4 characters, or starting with a digit, begin in column 13, shorter ones in column 14.
        A ValueError is raised if the chain name is longer than one character.

        :param record: record name (ATOM or HETATM)
        :param serial: atom serial number, as it should be written
        :param name: atom name
        :param resname: residue name
        :param chain: chain name, of one character
        :param resid: residue number, written with its last 4 digits if it does not fit
        :param altloc: alternate location indicator (default empty)
        :param icode: insertion code (default empty)
        :returns: string of 30 characters
        '''
        if len(chain) > 1:
            raise ValueError("chain name %s is longer than one character, which the PDB format cannot hold" % chain)

        if len(name) >= 4 or name[:1].isdigit():
            name = "%-4s" % name
        else:
            name = " %-3s" % name

        return "%-6s%5s %4s%1s%-4s%1s%4s%1s   " % (record, serial, name, altloc, resname, chain, Molecule._pdb_resid(resid), icode)

    @staticmethod
    def _pdb_ter(serial, resname, chain, resid, icode=""):
        '''
        TER record closing a chain, following the PDB format.

        :param serial: serial number of the TER record, as it should be written (the one following the last atom of the chain)
        :param resname: residue name of the last atom of the chain
        :param chain: chain name, of one character
        :param resid: residue number of the last atom of the chain, written with its last 4 digits if it does not fit
        :param icode: insertion code of the last atom of the chain (default empty)
        :returns: TER line, with its newline
        '''
        return "TER   %5s      %-4s%1s%4s%1s\n" % (serial, resname, chain, Molecule._pdb_resid(resid), icode)

    @staticmethod
    def _pdb_formal_charge(charge):
        '''
        formal charge as written in columns 79-80 of a PDB line.

        :param charge: integer formal charge, between -9 and 9
        :returns: digit followed by its sign (e.g. "2+" or "1-"), or an empty string for a zero charge
        '''
        charge = int(charge)
        if charge == 0:
            return ""
        return "%d%s" % (abs(charge), "+" if charge > 0 else "-")

    @staticmethod
    def _parse_formal_charge(text):
        '''
        formal charge from columns 79-80 of a PDB line.

        :param text: columns 79-80, e.g. "2+" or "1-" (a sign before the digit, e.g. "-1", is also accepted)
        :returns: integer formal charge, 0 if the field is blank or not a charge
        '''
        text = text.strip()
        if len(text) == 2 and text[0].isdigit() and text[1] in "+-":
            return int(text[0]) * (1 if text[1] == "+" else -1)
        if len(text) == 2 and text[0] in "+-" and text[1].isdigit():
            return int(text[1]) * (1 if text[0] == "+" else -1)
        return 0

    def _check_pdb_limits(self, conformations, indices):
        '''
        test whether the atoms to write fit the columns of the PDB format.

        Coordinates outside -999.999 to 9999.999 Angstrom cannot be written, and residue numbers outside -999 to 9999 are written with their last 4 digits.

        :param conformations: conformations to write
        :param indices: indices of atoms to write
        '''
        xyz = self.coordinates[np.asarray(conformations)][:, indices]
        if np.any(xyz < -999.9995) or np.any(xyz > 9999.9995):
            raise ValueError("PDB files must have coordinates between -999.999 and 9999.999 Angstrom")

        resid = self.data["resid"].values[indices].astype(int)
        if np.any(resid < -999) or np.any(resid > 9999):
            warnings.warn("residue numbers outside -999 to 9999 are written with their last 4 digits", stacklevel=3)

    def write_pdb(self, filename, conformations=[], indices=[], split_struc=False, dssp=False):
        '''
        overload superclass method for writing (multi)pdb. Every conformation is written as a MODEL/ENDMDL block, and the file ends with an END record.

        A TER record follows the last ATOM record of every chain (a chain being a run of atoms with the same chain name), as in the PDB format: HETATM records written after it (e.g. ligands or water) and chains made of HETATM records only (e.g. ions) get no TER. The TER record takes the next serial number, so that the atoms after it continue from the following one.
        Formal charges (column formal_charge of self.data, 0 if missing) are written in columns 79-80 as e.g. "2+" or "1-", and left blank when zero.

        :param filename: name of pdb file to be generated.
        :param indices: indices of atoms to write to file. If empty, all atoms are written. Indices obtainable with a call like: indices=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, a multipdb with all conformations will be produced.
        :param split_struc: Guess chain split on the atoms being written, and rename their chains accordingly (each guessed chain is then closed by TER after its last ATOM record). The molecule itself is not changed. Default: False. Set to False if protein is broken, but should retain chain lettering and doesn't have chain breaks.
        :param dssp: If using DSSP secondary structure check, requires that CRYST be the first line by default (hence write that line)
        :raises IndexError: if a requested conformation does not exist
        :raises ValueError: if coordinates, formal charges or chain names do not fit the PDB format

        Chain names of two characters are written as their first character in column 22, and in full as segment identifier (columns 73-76), which :func:`import_pdb <biobox.classes.molecule.Molecule.import_pdb>` reads back.
        Residue numbers outside -999 to 9999 are written with their last 4 digits, and a UserWarning is issued.
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
                raise IndexError("requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        if len(indices) == 0:
            indices = np.arange(len(self.points))

        # guess chains once, on a copy of the atoms being written, so that all models share them
        if split_struc:
            S = self.get_subset(indices, conformations=[frames[0]])
            S.guess_chain_split()
            chains = np.asarray(S.data["chain"].values, dtype=object)
        else:
            chains = np.asarray(self.data["chain"].values, dtype=object)[indices]

        self._check_pdb_limits(frames, indices)
        formal_charge = self._formal_charges()[indices]
        if np.any(np.abs(formal_charge) > 9):
            raise ValueError("PDB files must have formal charges between -9 and 9")

        # a TER record follows the last ATOM record of every chain, and takes the next serial number
        records = np.asarray(self.data["atom"].values, dtype=object)[indices]
        ter = np.zeros(len(indices), dtype=bool)
        bounds = np.r_[0, np.flatnonzero(chains[1:] != chains[:-1]) + 1, len(indices)] if len(indices) > 0 else np.array([0])
        for a, b in zip(bounds[:-1], bounds[1:]):
            polymer = np.flatnonzero(records[a:b] == "ATOM")
            if len(polymer) > 0:
                ter[a + polymer[-1]] = True
        serials = []
        ter_serials = {}
        serial = 1
        for i in range(len(indices)):
            serials.append(self._hybrid36(serial))
            serial += 1
            if ter[i]:
                ter_serials[i] = self._hybrid36(serial)
                serial += 1

        f_out = open(filename, "w")
        # the file is closed, and the current conformation restored, also when writing fails
        try:
            if dssp:
                f_out.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n") # only if doing secondary structure check

            for cnt, f in enumerate(frames):
                # get all informations from PDB (for current conformation) in a list
                f_out.write("MODEL     %4d\n" % (cnt + 1))
                self.set_current(f)
                d = self.get_pdb_data(indices)

                for i in range(0, len(d), 1):
                    chain, segid = self._pdb_chain_segid(chains[i])

                    # create and write PDB line
                    L = self._pdb_atom_prefix(d[i][0], serials[i], d[i][2], d[i][3], chain, d[i][5], d[i][12], d[i][13])
                    L += '%8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%2s\n' % (float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), segid, d[i][11], self._pdb_formal_charge(formal_charge[i]))
                    f_out.write(L)

                    # terminate chain
                    if ter[i]:
                        f_out.write(self._pdb_ter(ter_serials[i], d[i][3], chain, d[i][5], d[i][13]))

                f_out.write("ENDMDL\n")

            f_out.write("END\n")
        finally:
            f_out.close()
            self.set_current(currentbkp)

        return
    

    def write_gro(self, filename, conformations=[], indices=[], gmx_correction=False):
        '''
        write structure(s) in .gro format, converting coordinates from Angstrom to nm.

        The box of every frame is read from properties['box'] if available, otherwise it is the extent of the written atoms.

        :param filename: name of .gro file to be generated.
        :param indices: indices of atoms to write to file. If empty, all atoms are written. Indices obtainable with a call like: indices=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, all conformations will be written.
        :param gmx_correction: unused, kept for compatibility. Atom numbers always run from 1 and restart after 99999, as in GROMACS, and residue IDs above 99999 restart likewise.
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
                raise IndexError("requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        f_out = open(filename, "w")
        for f in frames:
            # get all informations from PDB (for current conformation) in a
            # list
            self.set_current(f)

            # ATOM/HETATM, index, atom name, resname, chain name, residue ID, x,
            # y, z, beta factor, occupancy, atomtype
            d = self.get_pdb_data(indices)
            f_out.write("%s\n" % filename.split(".")[0])
            f_out.write("%s\n" % len(d))
            for i in range(0, len(d), 1):
                # create and write .gro line
                resid = int(d[i][5])
                if resid > 99999:
                    resid = resid % 100000
                L = '%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n' % (resid, d[i][3], d[i][2], (i + 1) % 100000, float(d[i][6]) / 10.0, float(d[i][7]) / 10.0, float(d[i][8]) / 10.0)
                f_out.write(L)

            if "box" in self.properties:
                b = self.properties["box"][f] / 10.0
            else:
                # box enclosing the written atoms, in nm
                xyz = np.array([row[6:9] for row in d]).astype(float)
                b = (np.max(xyz, axis=0) - np.min(xyz, axis=0)) / 10.0

            formatting = ""
            for item in b:
                formatting+="%10.5f"
            formatting+="\n"

            f_out.write(formatting%tuple(b))

        f_out.close()
        self.set_current(currentbkp)
        return

    def beta_factor_from_rmsf(self, indices=-1):
        '''
        estimate atoms beta factor on the base of their RMSF over all conformations (B = 8 pi^2 RMSF^2 / 3), see :func:`rmsf <biobox.classes.structure.Structure.rmsf>`.
        The beta column of self.data is not modified.

        :param indices: indices of atoms of interest. If not set (default -1) all atoms will be considered.
        :returns: numpy array of beta factors, in Angstrom^2
        '''
        rmsf = self.rmsf(indices)
        return 8.0 * (np.pi**2) * (rmsf**2) / 3.0

    def rmsf_from_beta_factor(self, indices=[]):
        '''
        calculate RMSF from atoms beta factors (RMSF = sqrt(3 B / (8 pi^2))).

        :param indices: indices of atoms of interest. If not set all atoms will be considered.
        :returns: numpy array of RMSF values, in Angstrom
        '''

        try:
            if len(indices) == 0:
                b = self.data["beta"].values
            else:
                b = self.data["beta"].values[indices]

            return np.sqrt(b * 3 / (8 * np.pi * np.pi))

        except Exception:
            raise KeyError('beta factors missing?')

    def get_mass_by_residue(self, skip_resname=[]):
        '''
        Compute protein mass using residues (i.e. account also for atoms not present in the structure)

        Sum the average mass of every residue (using the knowledge base of residue masses in Dalton, knowledge['residue_mass']).
        Residues are identified by chain, residue number and insertion code. Masses are residue masses within a chain, so the water of the chain termini is not added.
        A KeyError is raised if a residue name is not in the knowledge base.
        The knowledge base can be expanded or edited by adding entries to the molecule's residue mass dictionary, e.g. to add the residue "TST" mass in molecule M type: M.knowledge['residue_mass']["TST"]=142.42

        :param skip_resname: list of resnames to skip. Useful to exclude ions water or other ligands from the calculation.
        :returns: mass of molecule in Dalton
        '''
        #@todo mass of N and C termini to add for every chain

        mass = 0
        chains = np.unique(self.data["chain"].values)
        for chainname in chains:
            # for every chain, get a list of all its (unique) resids
            indices = self.atomselect(chainname, "*", "*", True)[1]

            # for every residue in the chain, identified by residue number and insertion
            # code, take the resname of its first atom
            for i in self._one_per_residue(indices):
                resname = self.data['resname'].values[i]

                if resname not in skip_resname:
                    try:
                        # add mass of residue to total mass
                        mass += self.know('residue_mass')[resname]
                    except Exception:
                        #@todo: if residue is not known, why not summing constituent atoms masses, warning the user that it's an estimation?
                        raise KeyError("mass for resname %s is unknown!\nInsert a key in protein\'s masses dictionary knowledge['residue_mass'] and retry!\nex.: protein.knowledge['residue_mass'][\"TST\"]=142.42" %resname)

        return mass

    def get_mass_by_atom(self, skip_resname=[]):
        '''
        compute protein mass using atoms in pdb

        sum the mass of all atoms, according to their atomtype (using the knowledge base of atom masses in Dalton, knowledge['atom_mass']).
        A KeyError is raised if an atomtype is empty or not in the knowledge base.
        The knowledge base can be expanded or edited by adding or editing entries to the molecule's mass dictionary, e.g. to add the atom "PI" mass in molecule M type: M.knowledge['atom_mass']["PI"]=3.141592

        :param skip_resname: list of resnames to skip. Useful to exclude ions water or other ligands from the calculation.
        :returns: mass of molecule in Dalton
        '''

        mass = 0
        for i in range(0, len(self.data), 1):
            resname = self.data["resname"].values[i]
            atomtype = self.data["atomtype"].values[i]

            if resname not in skip_resname:
                try:
                    mass += self.know('atom_mass')[atomtype]
                except Exception:
                    if atomtype == "":
                        raise KeyError("no atomtype found for atom %s (name %s, resname %s, resid %s)!" % (i, self.data["name"].values[i], resname, self.data["resid"].values[i]))
                    else:
                        raise KeyError("mass for atom %s is unknown!\nInsert a key in protein\'s masses dictionary knowledge['atom_mass'] and retry!\nex.: protein.knowledge['atom_mass'][\"PI\"]=3.141592" %atomtype)

        return mass

    def s2(self, atomname1="N", atomname2="H"):
        '''
        compute the order parameter s2 over all conformations, given two atoms defining the vector of interest in every residue.

        A residue is measured if it contains exactly one atom named atomname2 (and an atom named atomname1). No superposition is performed.

        :param atomname1: name of the first atom (default N)
        :param atomname2: name of the second atom (default H)
        :returns: data numpy array (object dtype) containing information about residues for which measuring has been performed, one row [chain, resid, insertion code] per residue
        :returns: numpy array of s2 values of the residues for which both provided input atoms have been found, in the same order
        '''

        Nidx = self.atomselect("*", "*", atomname1, get_index=True)[1]
        Hidx = self.atomselect("*", "*", atomname2, get_index=True)[1]

        if len(Nidx) == 0:
            raise ValueError("no atom name %s found!"%atomname1)

        if len(Hidx) == 0:
            raise ValueError("no atom name %s found!"%atomname2)

        icode = self._column_or_blank("icode")
        Ndata = np.column_stack([self.data["chain"].values[Nidx], self.data["resid"].values[Nidx], icode[Nidx]])
        Hdata = np.column_stack([self.data["chain"].values[Hidx], self.data["resid"].values[Hidx], icode[Hidx]])

        a1 = []
        a2 = []
        d = []
        for i in range(0, len(Ndata), 1):
            j = np.where(
                (Hdata[:, 0] == Ndata[i, 0]) & (Hdata[:, 1] == Ndata[i, 1]) & (Hdata[:, 2] == Ndata[i, 2]))[0]

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


    def get_secondary_structure(self, dssp_path=''):
        '''
        compute the protein's secondary structure of the current conformation, calling DSSP.

        The temporary files tmp.pdb and result.dssp are written in, and removed from, the current working directory.

        :param dssp_path: DSSP executable (path and filename). If not provided, the default behaviour is to seek for this information in the environment variable DSSPPATH
        :returns: numpy array of characters, with one-letter-coded secondary structure according to DSSP, one per residue listed by DSSP (chain breaks skipped, "-" where DSSP assigns none).
        '''
        #dssp="~/bin/dssp-2.0.4-linux-amd64"
        if dssp_path == '':
            try:
                dssp_path = os.environ['DSSPPATH']
            except KeyError:
                raise RuntimeError("DSSPPATH environment variable undefined")

        # generate temporary PDB and calculate secondary structure using DSSP
        self.write_pdb("tmp.pdb", conformations=[self.current], split_struc=False, dssp=True)

        #TMP: assign all atoms to structure
        #subprocess.check_call('~/bin/amber16_tmp/bin/tleap -f build > /dev/null', shell=True)
        try:
            import subprocess
            subprocess.check_call("%s tmp.pdb -o result.dssp"%dssp_path, shell=True)
            fin=open("result.dssp","r")
        except Exception as e:
            raise RuntimeError("Could not calculate secondary structure! %s"%e) from e

        readit=False
        secstruct=[]
        for line in fin:

            if readit:
                try:
                    if line[13:15] == '!*' or line[13] == '!':
                        continue
                    else:
                        ss = line[16]
                        if line[16] == " ":
                            ss = "-"

                        secstruct.append(ss)
                except:
                    continue

            if "#" in line:
                readit=True

        fin.close()

        # clean temporary files
        os.remove("result.dssp")
        os.remove("tmp.pdb")

        return np.array(secstruct) #(secstruct[0:210])

    def renumber_resid_keep_chains(self, start_from=1, reset_resid_with_chain=True):
        '''
        Renumber residues consecutively in file order (starting from start_from variable), resetting the numbering per chain letter
        (i.e. not the structure.) Useful for insertion/grafting of motifs of arbitrary length, which disrupt the renumbering, or
        when the structure is broken and you want two or more discontinuous segements to have a single chain letter, and continuous resnums.

        Residues are delimited as in _residue_starts, so two residues with the same number are numbered separately, whether adjacent
        or in different places of the file, and residues without a CA (ligands, ions, water) are renumbered too.
        Insertion codes are cleared, since the new numbers are unique.

        :param start_from: Start counting resnums from this value (default 1)
        :param reset_resid_with_chain: Number each chain from start_from (default True), otherwise continue the numbering across chains
        '''

        self.data.reset_index(drop=True, inplace=True)
        self.data["index"] = np.arange(len(self.data))

        starts = self._residue_starts()
        chains = self.data["chain"].values[starts]

        # a chain whose residues appear in separate blocks of the file continues its own numbering
        next_resid = {}
        new_resid = np.zeros(len(chains), dtype=int)
        for cnt, c in enumerate(chains):
            key = c if reset_resid_with_chain else None
            new_resid[cnt] = next_resid.get(key, start_from)
            next_resid[key] = new_resid[cnt] + 1

        self.data["resid"] = new_resid[np.cumsum(starts) - 1]
        if "icode" in self.data.columns:
            self.data["icode"] = ""

    def reorder_resid(self, resids, chain="A", renumber=True):
        """
        Reorder the internal resid of a PDB structure (retaining the topology) based on the resids list.
        Number of elements in resids list must == number of resid in the chain. The chain keeps its place in the structure,
        and the reordering applies to every conformation.

        :param resids: List of residue numbers giving the new order of the residues of the chain. Doesn't have to be same values as native resid (the values are shifted so that the smallest one matches the smallest native resid), but must contain every residue once. There can be no numeric breaks (i.e., [1, 2, 3, 6, 7, 8, 4, 5] acceptable, [1, 2, 3, 8, 4, 5] is not
        :param chain: Chain to apply reordering to (default A)
        :param renumber: After restructuring metadata, renumber residues of the whole molecule with :func:`renumber_resid_keep_chains <biobox.classes.molecule.Molecule.renumber_resid_keep_chains>` (default True)
        """

        self.data.reset_index(drop=True, inplace=True)

        pos = np.flatnonzero(self.data["chain"].values == chain)
        if len(pos) == 0:
            raise ValueError("chain %s not found" % chain)
        resid = self.data["resid"].values[pos]

        # shift resids values so they match the values of the native resid
        resids = np.asarray(resids) + (np.min(resid) - np.min(resids))
        if len(resids) != len(np.unique(resid)) or set(resids.tolist()) != set(resid.tolist()):
            raise ValueError("resids must list every residue of chain %s once" % chain)

        # the chain's atoms are sorted by the position of their resid in resids, keeping their order within a residue,
        # and put back into the slots the chain occupied
        rank = {r: i for i, r in enumerate(resids.tolist())}
        order = np.arange(len(self.data))
        order[pos] = pos[np.argsort([rank[r] for r in resid.tolist()], kind="stable")]

        self.data = self.data.iloc[order].reset_index(drop=True)
        self.data["index"] = np.arange(len(self.data))
        self.coordinates = self.coordinates[:, order]
        self.points = self.coordinates.view()[self.current]

        if renumber:
            self.renumber_resid_keep_chains()

    def get_couples(self, indices, cutoff):
        '''
        given a list of indices, compute the all-vs-all distance in the current conformation and return only couples below a given cutoff distance

        useful for the detection of disulfide bridges or linkable sites via cross-linking (approximation, supposing euclidean distances)'

        :param indices: indices of atoms to check.
        :param cutoff: maximal distance, in Angstrom, to consider a couple as linkable. Only couples strictly closer than cutoff are returned.
        :returns: nx3 numpy array of floats containing, for every valid connection (each reported once), id of first atom, id of second atom and distance between the two. The first atom is the one appearing later in indices. If no couple is found, an empty array of shape (0,) is returned.
        '''

        import biobox.measures.interaction as I

        points1 = self.get_xyz()[indices]

        dist = I.distance_matrix(points1, points1)
        couples = I.get_neighbors(dist, cutoff)

        res = []
        for c in couples.transpose():
            if c[0] > c[1]:
                res.append([indices[c[0]], indices[c[1]], dist[c[0], c[1]]])

        return np.array(res)


    def match_residue(self, M2, sec = 3):
        '''
        Compares the sequences of two bb.Molecule() peptide strands (residue names of their CA atoms) and returns the resids within both peptides when the two are homogenous
        beyond a certain threshold. The default is 3 amino acids (given by sec) in a row must be identical.
        HIE, HIP and HID are compared as HIS, and four-letter residue names lose their first letter (e.g. Amber terminal names NALA, CHIE).

        Useful when aligning PDB structures that have been crystallised separately, so one may be missing the odd residue
        or have a few extra at the end.

        :param M2: The second bb.Molecule() to compare with
        :param sec: Number of consecutive amino acids in a row that must match before resid's are recorded
        :returns: list of matching resids in self
        :returns: list of the corresponding resids in M2
        '''
        # First run the match residue using the expected inputs
        M1_res, M2_res = self._match_residue_maths(M2, sec = sec)

        # Import a check to see if we've correctly counted the residues (e.g. in homodimer case)
        if np.shape(np.unique(M1_res)) != np.shape(np.unique(M2_res)):
            M2_res, M1_res = M2._match_residue_maths(self, sec = sec)

        return M1_res, M2_res


    def _match_residue_maths(self, M2, sec):
        '''
        Does the maths for match_residue. The reason for this additional step is that sometimes
        if the numbering is a bit off between the different proteins (i.e. a shift in the initial
        starting residues) and the protein is a homodimer, we can end up only adding the second
        monomer unit, and sometimes add it twice. Therefore we can compare two runs of this
        for an answer.

        :param M2: The second bb.Molecule() to compare with
        :param sec: Number of consecutive amino acids in a row that must match before resid's are recorded
        :returns: list of matching resids in self
        :returns: list of the corresponding resids in M2
        '''

        # Get residue names / unique IDs
        M1_reslist = self.data["resname"][self.data["name"] == 'CA'].values
        M2_reslist = M2.data["resname"][M2.data["name"] == 'CA'].values
        M1_resid = self.data["resid"][self.data["name"] == 'CA'].values
        M2_resid = M2.data["resid"][M2.data["name"] == 'CA'].values

        # Remove C or N prefixes
        for cnt, val in enumerate(M1_reslist):
            if len(val) == 4:
                M1_reslist[cnt] = val[1:]
            else:
                continue
        for cnt, val in enumerate(M2_reslist):
            if len(val) == 4:
                M2_reslist[cnt] = val[1:]
            else:
                continue

        # Rename residues temporararily so they match better
        M1_reslist[np.logical_or(np.logical_or(M1_reslist == 'HIE', M1_reslist == 'HIP'), M1_reslist == 'HID')] = 'HIS'
        M2_reslist[np.logical_or(np.logical_or(M2_reslist == 'HIE', M2_reslist == 'HIP'), M2_reslist == 'HID')] = 'HIS'

        M1_reskeep = []
        M2_reskeep = []
        M2_cnt = 0
        M1_cnt = 0

        while M1_cnt < len(M1_reslist):

            # Initial check to see if we have a run of good matches (more than coincidence).
            # Near the end of the first strand, the run is as long as the residues left
            run1 = M1_reslist[M1_cnt:(M1_cnt + sec)]
            run2 = M2_reslist[M2_cnt:(M2_cnt + sec)]
            if len(run1) == len(run2) and np.all(run1 == run2):

                while M1_reslist[M1_cnt] == M2_reslist[M2_cnt]:

                    M1_reskeep.append(M1_resid[M1_cnt])
                    M2_reskeep.append(M2_resid[M2_cnt])

                    M2_cnt += 1
                    M1_cnt += 1

                    # Break if we reach the maximum array length limit
                    if M1_cnt == len(M1_reslist) or M2_cnt == len(M2_reslist):
                        break

                    # Check if we conicidently had the correct corresponding resnames
                    if len(M1_reskeep) > 2:
                        if M1_reskeep[-1] - M1_reskeep[-2] != 1 and M2_reskeep[-1] - M2_reskeep[-2] == 1:
                            M1_reskeep = M1_reskeep[:-1]
                            M2_reskeep = M2_reskeep[:-1]
                            break
                        elif M1_reskeep[-1] - M1_reskeep[-2] == 1 and M2_reskeep[-1] - M2_reskeep[-2] != 1:
                            M1_reskeep = M1_reskeep[:-1]
                            M2_reskeep = M2_reskeep[:-1]
                            break
                        else:
                            continue

            # Elsewise move forward in count on second structure
            else:
                M2_cnt += 1

            # Break if M1 and M2 have reached their ends, restart if only M2 has
            if M1_cnt == len(M1_reslist): #and M2_cnt == len(M2_reslist):
                break
            # break if the length of residues in M2 we are counting to now are longer than
            # the possible max number of saved residues in M2
            elif len(M2_reskeep) >= len(M2_reslist):
                break
            # Need case so we don't recount a chain in the event of a homodimer
            elif M2_cnt >= len(M2_reslist):
                M1_cnt += 1
                M2_cnt = len(M1_reskeep)
            else:
                continue

        return M1_reskeep, M2_reskeep

    def pdb2pqr(self, ff="", amber_convert=True):
        '''
        Parses data from the pdb input into a pqr format. This uses the panda dataframe with the information
        regarding atom indexes, types etc. in the self.data files.
        It outputs a panda dataframe with the pqr equivalent information. It requires a datafile forcefield input.
        The default is the amber14sb.dat forcefield file held within the package data/ folder.

        The molecule itself is modified: chain IDs are reassigned by guess_chain_split, and with amber_convert residues are
        renamed in place to their forcefield names (e.g. NALA, CHID, HIE). The atomtype, radius and charge columns of self.data are not modified.
        A KeyError is raised if an atom (residue name and atom name) is not found in the forcefield file, and a UserWarning is issued if HIS residues are renamed.

        :param ff: name of forcefield text file input that needs to be read to read charges / vdw radii. If empty (default), amber14sb.dat is used.
        :param amber_convert: If True, will assume forcefield is amber and convert resnames as necessary
        :returns: pandas DataFrame, a copy of self.data whose atomtype, radius and charge columns hold the forcefield values
        '''

        intervals = self.guess_chain_split()[1]

        if amber_convert:
            # patch naming of C-termini
            for i in intervals[1:]:
                idxs = self.same_residue(i-1, get_index=True)[1]
                names = self.data.loc[idxs, ["name"]].values
                if np.any(names == "OC1") or np.any(names == "OXT"):
                    resname = self.data.loc[idxs[0], ["resname"]].values[0]
                    newresnames = np.array(["C"+resname]*len(idxs))
                    self.data.loc[idxs, ["resname"]] = newresnames

            # patch naming of N-termini
            for i in intervals[0:-1]:
                idxs = self.same_residue(i, get_index=True)[1]
                names = self.data.loc[idxs, ["name"]].values
                if np.any(names == "H1") and np.any(names == "H2"):
                    resname = self.data.loc[idxs[0], ["resname"]].values[0]
                    newresnames = np.array(["N"+resname]*len(idxs))
                    self.data.loc[idxs, ["resname"]] = newresnames

            # Need to check whether it matches HIE, HID or HIP depending on what protons are present
            resnames = self.data["resname"].values
            if np.any(np.isin(resnames, ["HIS", "NHIS", "CHIS"])):
                warnings.warn("found residue with name HIS, checking to see what protonation state it is in and reassigning to HIP, HIE or HID. You should check HIS in your pdb file is right to be sure!", stacklevel=2)
                names = self.data["name"].values
                bounds = np.r_[np.flatnonzero(self._residue_starts()), len(resnames)]
                for a, b in zip(bounds[:-1], bounds[1:]):
                    if resnames[a] not in ["HIS", "NHIS", "CHIS"]:
                        continue
                    has_hd1 = np.any(names[a:b] == "HD1")
                    has_he2 = np.any(names[a:b] == "HE2")
                    if has_hd1 and has_he2:
                        new = "HIP"
                    elif has_he2:
                        new = "HIE"
                    elif has_hd1:
                        new = "HID"
                    else:
                        continue
                    self.data.iloc[a:b, self.data.columns.get_loc("resname")] = resnames[a][:-3] + new

        if len(ff) == 0:
            folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            ff = os.path.join(folder, "data", "amber14sb.dat")

        if os.path.isfile(ff) != 1:
            raise FileNotFoundError("%s not found!" % ff)

        ff = np.loadtxt(ff, usecols=(0,1,2,3,4), dtype=str)

        cols = ['resname', 'name', 'charge', 'radius', 'atomtype'] # where radius is the VdW radius in the amber file
        idx = np.arange(len(ff))
        pqr_data = pd.DataFrame(ff, index=idx, columns=cols)

        charges = []
        radius = []
        atomtypes = []

        # Move through each line in the pdb.data file and find the corresponding charge / vdw radius as supplied by the forcefield
        for i, resnames in enumerate(self.data["resname"]):
            values_res = pqr_data["resname"] == resnames
            values_name = pqr_data["name"] == self.data["name"][i]
            values = np.logical_and(values_res, values_name)
            value_loc = pqr_data[values]

            if len(value_loc) == 0:
                raise KeyError("The atom names in your PDB file do not match the PQR file: atom %s (name %s, resname %s, resid %s) not found" % (self.data["index"].iloc[i], self.data["name"][i], resnames, self.data["resid"].iloc[i]))
            else:
                charges.append(float(value_loc.iloc[0]["charge"]))
                radius.append(float(value_loc.iloc[0]["radius"]))
                atomtypes.append(value_loc.iloc[0]["atomtype"])

        # Drop the beta factor / occupancy data to be replaced with charge / vdw radius numbers
        pqr = self.data.drop(['atomtype', 'radius', 'charge'], axis=1) #  remove obselete data
        pqr['atomtype'] = atomtypes  # Replace with Amber derived data for each atom
        pqr['radius'] = radius
        pqr['charge'] = charges

        return pqr

    def write_pqr(self, filename, conformations=[], indices=[]):
        '''
        write (multi)pqr, with charges and radii from :func:`pdb2pqr <biobox.classes.molecule.Molecule.pdb2pqr>` called with its default arguments (which modifies the molecule in place).

        Every conformation is followed by an END statement. Charges are written in columns 55-62 and radii in columns 63-69.

        :param filename: name of pqr file to be generated.
        :param indices: indices of atoms to write to file. If empty, all atoms are written. Indices obtainable with a call like: indices=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, a multi-model pqr with all conformations will be produced.
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
                raise IndexError("requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        # Get our PQR database style
        pqr = self.pdb2pqr()

        # rows of the atoms being written, in the molecule and in pqr
        rows = np.arange(len(self.points)) if len(indices) == 0 else np.asarray(indices)
        self._check_pdb_limits(frames, rows)
        serials = [self._hybrid36(i + 1) for i in range(len(rows))]

        f_out = open(filename, "w")

        for f in frames:
            # get all informations from PDB (for current conformation) in a list
            self.set_current(f)
            d = self.get_pdb_data(indices)

            for i in range(0, len(d), 1):
                # create and write PQR line, with charge and radius in columns 55-62 and 63-69 so
                # that they stay separated by whitespace
                q = pqr.iloc[rows[i]]
                L = self._pdb_atom_prefix(d[i][0], serials[i], d[i][2], d[i][3], d[i][4], d[i][5], d[i][12], d[i][13])
                L += '%8.3f%8.3f%8.3f%8.4f%7.4f       %2s\n' % (float(d[i][6]), float(d[i][7]), float(d[i][8]), float(q["charge"]), float(q["radius"]), d[i][11])
                f_out.write(L)

            f_out.write("END\n")

        f_out.close()

        self.set_current(currentbkp)

        return

    def clean(self, remove_non_amino=True):
        '''
        return a copy of the molecule without alternate locations and, optionally, without non amino acid residues (e.g. water, ions and ligands). The molecule itself is not changed.

        Residues are identified by chain, residue number and insertion code. Within a residue having alternate locations, only the alternate location
        with the highest mean occupancy over its atoms is kept (on a tie, the one appearing first). Atoms without an alternate location indicator are always kept,
        and the altloc of all atoms of the returned molecule is empty. All conformations of the kept atoms are returned.

        :param remove_non_amino: if True, keep only residues whose name is a standard amino acid, or one of its Amber N-terminal, C-terminal or protonation variants
        :returns: new :class:`biobox.classes.molecule.Molecule`
        '''

        # all amino acids (in case we want to remove non-standard residues). Also includes N and C prefixs
        amino = ['ILE','GLN', 'GLY', 'MSE', 'GLU', 'CYS', 'ASP', 'SER', 'HSD', 'HSE', 'PRO', 'CYX', 'HSP', 'HID', 'HIE', 'ASN',
                'HIP', 'VAL', 'THR', 'HIS', 'TRP', 'LYS', 'PHE', 'ALA', 'MET', 'LEU', 'ARG', 'TYR', 'NILE', 'NGLN', 'NGLY',
                'NMSE', 'NGLU', 'NCYS', 'NASP', 'NSER', 'NHSD', 'NHSE', 'NPRO', 'NCYX', 'NHSP', 'NHID', 'NHIE', 'NASN', 'NHIP',
                'NVAL', 'NTHR',  'NHIS','NTRP', 'NLYS', 'NPHE', 'NALA', 'NMET', 'NLEU', 'NARG', 'NTYR', 'CILE', 'CGLN', 'CGLY',
                'CMSE', 'CGLU', 'CCYS', 'CASP', 'CSER', 'CHSD', 'CHSE', 'CPRO', 'CCYX', 'CHSP', 'CHID', 'CHIE', 'CASN', 'CHIP',
                'CVAL', 'CTHR', 'CHIS', 'CTRP', 'CLYS', 'CPHE', 'CALA', 'CMET', 'CLEU', 'CARG', 'CTYR']

        chain = self.data["chain"].values
        resid = self.data["resid"].values
        icode = self._column_or_blank("icode")
        altloc = self._column_or_blank("altloc")
        occupancy = self.data["occupancy"].values.astype(float)

        # atoms of every alternate location, per residue, in order of appearance
        residues = {}
        for i in np.flatnonzero(altloc != ""):
            residues.setdefault((chain[i], resid[i], icode[i]), {}).setdefault(altloc[i], []).append(i)

        keep = np.ones(len(self.data), dtype=bool)
        for locations in residues.values():
            # max returns the first location having the highest mean occupancy
            best = max(locations, key=lambda loc: np.mean(occupancy[locations[loc]]))
            for loc, idx in locations.items():
                if loc != best:
                    keep[idx] = False

        if remove_non_amino:
            keep &= np.isin(self.data["resname"].values, amino)

        M = self.get_subset(np.flatnonzero(keep))
        M.data["altloc"] = ""

        return M


    def get_dipole_map(self, orig, pqr, time_start = 0, time_end = 2,resolution = 1., vox_in_window = 3., write_dipole_map = True, filename = "dipole_map.tcl"):
        '''
        Method for generating dipole maps to be used for electron density map generation. Also prints a dipole map as a result (and if desired). It calls a cython code in lib.

        :param orig: Origin points for voxel grid, as three arrays of voxel centre coordinates along x, y and z
        :param pqr: pandas DataFrame with a charge column, one row per atom. Can be generated by calling :func:`pdb2pqr <biobox.classes.molecule.Molecule.pdb2pqr>`
        :param time_start: First frame to parse in multipdb (default 0)
        :param time_end: frame at which parsing stops, excluded (default 2)
        :param resolution: Desired resolution of voxel, in Angstrom
        :param vox_in_window: Amount of surrounding space to contribute to local dipole. vox_in_window * resolution gives window size (in Ang.)
        :param write_dipole_map: Write a dipole map in TCL format to be read in via VMD (default True).
        :param filename: Name of desired dipole map to be written
        :returns: float32 numpy array of shape (time_end-time_start, nx, ny, nz, 3), the dipole vector of every voxel in every frame, where nx, ny and nz are the lengths of the three arrays of orig
        '''

        charges = pqr["charge"].values[:]

        crd = self.coordinates[time_start:time_end] # cut out coordinates we're interested in

        time_end -= time_start # shift to compensate for cutting the coordinates earlier
        time_start = 0

        dipole_map = e_density.c_get_dipole_map(crd = crd, orig = orig, charges = charges, time_start = time_start, time_end = time_end,resolution = resolution, vox_in_window = vox_in_window, write_dipole_map = write_dipole_map, filename = filename)

        return dipole_map

    def get_dipole_density(self, dipole_map, orig, min_val, V, filename, vox_in_window = 3., eqn = 'gauss', T = 310.15, P = 101. * 10**3, epsilonE = 54., resolution = 1.):
        '''
        Method to generate an electron density map based on a voxel grid of dipole vectors, and write it to a dx file. It calls a cython code in lib.

        :param dipole_map: The dipole map input. Can be generated with get_dipole_map above
        :param orig: Origin points for voxel grid
        :param min_val: Minimum coordinates of edge points for the voxel grid (i.e. a single x, y, z point defining the start point of the grid to match with the multipdb)
        :param V: Volume of a voxel (can be found by resolution**3, but left blank in case later version institute a sphere)
        :param filename: Name of electron density map file produced
        :param vox_in_window: Amount of surrounding space to contribute to local dipole. vox_in_window * resolution gives window size (in Ang.). The density function of each voxel is sampled within this window, centred on the voxel
        :param eqn: Equation mode to model the electron density, 'gauss' (default) or 'slater'
        :param T: Temperature of MD, in K
        :param P: Pressure of MD, in Pa
        :param epsilonE: Continuum dielectric surrounding the protein
        :param resolution: Desired resolution of voxel, in Angstrom
        :returns: 0 once the map is written to filename
        '''

        dummy = e_density.c_get_dipole_density(dipole_map = dipole_map, orig = orig, min_val = min_val, V = V, filename = filename, vox_in_window = vox_in_window, eqn = eqn, T = T, P = P, epsilonE = epsilonE, resolution = resolution)
        return dummy

    def _one_letter(self, resname):
        '''
        one-letter code of a residue name, reading Amber terminal names (e.g. NALA, CHIE) without their prefix.

        :param resname: residue name
        :returns: one-letter code, X if unknown
        '''
        mapping = self.knowledge["AA_mapping"]
        if resname in mapping:
            return mapping[resname]
        if len(resname) == 4 and resname[0] in "NC" and resname[1:] in mapping:
            return mapping[resname[1:]]
        return "X"

    def get_fasta(self, chains=True, chain_split=False):
        '''
        Generate the sequence associated with a moleule in a fasta approved format.
        The only thing that will need appending to text is the >SEQ information that you may want to edit (e.g. to include uniprot ID, strain if appropiate, etc.).
        If chains is True, then each new chain will append a / to designate the breaks.
        If chain_split is set to True, biobox will automatically split your protein according to where it sees gaps in the structure, and place / where these new chains begin.
        If you have gaps in your structure (e.g. from disordered regions), this can result in incorrect assignment of novel chains.

        Residue names are mapped through knowledge["AA_mapping"], which includes common variants (e.g. MSE, HIE, CYX).
        Amber terminal names (e.g. NALA, CHIE) are read without their prefix, and unknown residues are written as X.
        Only residues having a CA atom are included, and chains are written in sorted order of their names.

        :param chains: Assign / between chains. Default: True
        :param chain_split: Let biobox decide where the chain splits are (based on structure), with :func:`guess_chain_split <biobox.classes.molecule.Molecule.guess_chain_split>`, which renames the chains of the molecule in place. Default: False
        :returns: sequence string
        '''

        seq = ""
        if chain_split:
            self.guess_chain_split()
        if chains:
            for c in np.unique(self.data["chain"]):
                M = self.get_subset(self._one_per_residue(self.atomselect(c, "*", "CA", get_index=True)[1]))
                text = "".join([self._one_letter(S) for S in M.data["resname"]])
                if len(seq) == 0:
                    seq = text
                else:
                    seq += "/"
                    seq += text
        else:
            M = self.get_subset(self._one_per_residue(self.atomselect("*", "*", "CA", get_index=True)[1]))
            text = "".join([self._one_letter(S) for S in M.data["resname"]])
            if len(seq) == 0:
                seq = text
            else:
                seq += "/"
                seq += text
        return seq
