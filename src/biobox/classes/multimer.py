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

import numpy as np
import pandas as pd

from biobox.classes.polyhedron import Polyhedron
from biobox.classes.molecule import Molecule


class Multimer(Polyhedron):
    '''
    Construct and manipulate a protein assembly composed of several :func:`Molecule <biobox.classes.molecule.Molecule>` instances. Subclass of :func:`Polyhedron <biobox.classes.polyhedron.Polyhedron>`.
    '''

    def query(self, query_text, get_index=False):
        '''
        select specific atoms in a multimer on the basis of a text query.

        :param query_text: string selecting atoms of interest. Uses the pandas query syntax, can access all columns in the dataframe self.data (including "unit" and "unit_index").
        :param get_index: if set to True, the indices of selected atoms in the multimer's self.data are also returned
        :returns: coordinates of the selected points of the units' current conformations (in a unique kx3 numpy array), grouped by unit. If get_index is set to True, a list [coordinates, indices] is returned instead, where indices is a numpy array of row indices in the multimer's self.data.
        '''

        idx = self.data.query(query_text).index.values

        res = self.data.iloc[idx] #this is a new sliced dataframe
        targets = np.array(res.loc[:, ["unit", "unit_index"]].values)

        # append the coordinates of every unit within the query
        pts = np.empty([0, 3])
        for u in np.unique(targets[:, 0]):
            pos = targets[targets[:, 0] == u, 1].astype(int)
            this_unit = self.unit_labels[u]
            pts = np.concatenate((pts, self.unit[this_unit].points[pos]))

        if get_index:
            return [pts, idx]
        else:
            return pts


    def atomselect(self, u, chain, resid, atom, get_index=False, use_resname=False):
        '''
        select specific atoms in a multimer providing unit, chain, residue ID and atom name.

        :param u: label of desired unit to select in the multimer (str or int, accepts '*' as wildcard). Can also be a list or numpy array of labels.
        :param chain: selection of a specific chain name (accepts '*' as wildcard). Can also be a list or numpy array of strings.
        :param resid: residue ID of desired atoms (accepts '*' as wildcard). Can also be a list or numpy array of int.
        :param atom: name of desired atom (accepts '*' as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, the indices of selected atoms within each unit are also returned
        :param use_resname: if set to True, consider information in "resid" variable as resnames, and not resids
        :returns: coordinates of the selected points of the units' current conformations (in a unique kx3 numpy array). If get_index is set to True, a list [coordinates, indices] is returned instead, where indices has one entry per unit (in unit order): the indices of selected atoms in that unit's self.points array, or an empty list for units not selected.
        '''

        # extract id of units of interest
        if u == '*':
            unit_id = list(self.unit_labels.values())
        else:
            if isinstance(u, str) or isinstance(u, int):
                try:
                    unit_id = [self.unit_labels[str(u)]]
                except Exception as ex:
                    raise Exception("ERROR: unit %s not found!" % u)

            elif isinstance(u, list) or type(u).__module__ == 'numpy':
                unit_id = []
                for c in range(0, len(u), 1):
                    try:
                        unit_id.append(self.unit_labels[str(u[c])])
                    except Exception as ex:
                        raise Exception("ERROR: unit %s not found!" % u[c])
            else:
                raise Exception("ERROR: wrong type for unit selection. Should be str, int, list, or numpy")

        # initialize storage for indices and coordinates
        indices = []
        pts = np.empty([0, 3])
        for i in range(0, len(self.unit), 1):
            if i in unit_id:
                [pts_tmp, index_tmp] = self.unit[i].atomselect(chain, resid, atom, True, use_resname=use_resname)
                pts = np.concatenate((pts, pts_tmp))
                indices.append(index_tmp)
            else:
                # indices of all units must be stored. If unit is not
                # requested, return an empty array for it
                indices.append([])
        if get_index:
            return [pts, indices]
        else:
            return pts

    def make_molecule(self, rename_chains=False):
        '''
        Return a :func:`Molecule <biobox.classes.molecule.Molecule>` object containing all the points of the assembly, taken from the current conformation of every unit, with a single conformation.

        The data of all units is concatenated, keeping all their columns (including "unit" and "unit_index", which identify the unit every atom comes from) and renumbering the "index" column. The "charge" column is kept only if every unit has it. Knowledge about atom CCS is merged across units.

        :param rename_chains: if False (default), every atom keeps its original chain name, so that chain names are repeated across units. If True, all atoms of the i-th unit are given chain name chain_names[i] (A, B, C...), and the original chain names are discarded.
        :returns: :func:`Molecule <biobox.classes.molecule.Molecule>` object
        '''

        # create new data entry (renumber indices, reassign chain name), keeping every column of the units
        frames = []
        atom_ccs = {}
        for i in range(0, len(self.unit), 1):
            d = self.unit[i].data.copy()
            if rename_chains:
                d["chain"] = self.chain_names[i]
            frames.append(d)

            # merge knowledge about CCS acquired by different molecules
            for k in self.unit[i].knowledge['atom_ccs'].keys():
                atom_ccs[k] = self.unit[i].knowledge['atom_ccs'][k]

        data = pd.concat(frames, ignore_index=True)
        data["index"] = np.arange(len(data))

        # charges are kept only if every unit has them
        if not all("charge" in d.columns for d in frames):
            data = data.drop(columns="charge", errors="ignore")

        # create molecule, and push created data information
        M = Molecule()
        M.add_xyz(self.get_all_xyz())
        M.data = data
        M.properties['center'] = M.get_center()
        M.knowledge['atom_ccs'] = atom_ccs

        return M

    #def rmsd(self, ref_index, u="*", chain="*", resid="*", atom="*", align=False):
    #    '''
    #    Calculate the RMSD between atoms of interest in all structure with respect of a reference structure.

    #    supposes that all multimer subunits contain the same amount of alternative coordiantes.
    #    These are considered as representations of monomers conformations in a possible multimer.

    #    :param u: number of desired unit to select in the multimer
    #    :param chain: selection of a specific chain name (accepts '*' as wildcard). Can also be a list or numpy array of strings.
    #    :param resid: residue ID of desired atoms (accepts '*' as wildcard). Can also be a list or numpy array of of int.
    #    :param atom: name of desired atom (accepts '*' as wildcard). Can also be a list or numpy array of strings.
    #    :param ref_index: index of reference structure in conformations database
    #    :param align: if True, structures will all be aligned (cannot be undone)
    #    :returns: RMSD of all structures with respect of reference structure (in a numpy array)
    #    '''
    #
    #    if ref_index >= self.unit[0].coordinates.shape[0]:
    #        raise Exception("ERROR: requested frame %s as reference, but only %s frames are available!" %(ref_index, self.unit[0].coordinates.shape[0]))

    #    # select indices of atoms of interest and call overloaded method
    #    indices = self.atomselect(u, chain, resid, atom, get_index=True)[1]
    #    return super(Multimer, self).rmsd(ref_index, points_indices=indices, align=align)

    def get_data(self, indices, columns):
        '''
        Return information about atom of interest (i.e., slice the data DataFrame)

        :param indices: list of row indices in the multimer's self.data
        :param columns: list of columns (e.g. ["resname", "resid", "chain"])
        :returns: numpy array of the values in the selected rows and columns of the multimer's data DataFrame
        '''

        return self.data.loc[indices, columns].values

    def write_pdb(self, outname, rename_chains=False):
        '''
        Write a pdb of the multimeric assembly, one MODEL per frame. Every unit is written as a chain of its own (the i-th unit is chain i of chain_names, A, B, C...), replacing the original chain names, and is followed by a TER record. Atoms are renumbered sequentially across units.

        All units must have the same number of frames. Their current frames are restored after writing.

        :param outname: name of PDB file to generate
        :param rename_chains: unused. Every unit is always given its own chain name
        '''
        nframes = len(self.unit[0].coordinates)
        if any(len(u.coordinates) != nframes for u in self.unit):
            raise Exception("ERROR: all units must have the same number of frames")

        names = list(dict.fromkeys(self.chain_names))
        if len(self.unit) > len(names):
            raise Exception("ERROR: %s units, but only %s single-character chain names" % (len(self.unit), len(names)))

        currents = [u.current for u in self.unit]
        f_out = open(outname, "w")
        try:
            for f in range(nframes):
                f_out.write("MODEL        %i\n" % (f + 1))
                cnt = 1
                for j, u in enumerate(self.unit):
                    u.set_current(f)
                    # get data about points and their properties from the desired protein structure
                    d = u.get_pdb_data()
                    for i in range(0, len(d), 1):
                        L = Molecule._pdb_atom_prefix(d[i][0], Molecule._hybrid36(cnt), d[i][2], d[i][3], names[j], d[i][5], d[i][12], d[i][13])
                        L += '%8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                        f_out.write(L)
                        cnt += 1
                    f_out.write("TER\n")
                f_out.write("ENDMDL\n")
            f_out.write("END\n")
        finally:
            f_out.close()
            for u, c in zip(self.unit, currents):
                u.set_current(c)
