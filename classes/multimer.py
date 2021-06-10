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
import pandas as pd

from biobox.classes.polyhedron import Polyhedron
from biobox.classes.molecule import Molecule


class Multimer(Polyhedron):
    '''
    Construct and manipulate a protein assembly composed of several :func:`Molecule <molecule.Molecule>` instances. Subclass of :func:`Polyhedron <polyhedron.Polyhedron>`.
    '''

    def query(self, query_text, get_index=False):
        '''
        ## select specific atoms in a multimer un the basis of a text query.

        :param query_text: string selecting atoms of interest. Uses the pandas query syntax, can access all columns in the dataframe self.data.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: coordinates of the selected points (in a unique array) and, if get_index is set to true, a list of their indices in subunits' self.points array.
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
        ## select specific atoms in a multimer providing unit, chain, residue ID and atom name.

        :param u: number of desired unit to select in the multimer
        :param chain: selection of a specific chain name (accepts '*' as wildcard). Can also be a list or numpy array of strings.
        :param resid: residue ID of desired atoms (accepts '*' as wildcard). Can also be a list or numpy array of of int.
        :param atom: name of desired atom (accepts '*' as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the selected points (in a unique array) and, if get_index is set to true, a list of their indices in subunits' self.points array.
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

    def make_molecule(self):
        '''
        Return a :func:`Molecule <molecule.Molecule>` object containing all the points of the assembly. Chain will indicate different units, original chain value is pushed in segment entry.

        :returns: :func:`Molecule <molecule.Molecule>` object
        '''

        # create new data entry (renumber indices, reassign chain name)
        data = np.empty([0, 9])

        atom_ccs = {}
        r = []
        c = []
        skipcharge = False
        for i in range(0, len(self.unit), 1):
            data_tmp = self.unit[i].data[[
                "atom", "index", "name", "resname", "chain",
                "resid", "beta", "occupancy", "atomtype"]].values

            #data_tmp[:, 4] = self.chain_names[i] # avoid renaming the chains
            data = np.concatenate((data, data_tmp))

            # merge knowledge about CCS acquired by different molecules
            atom_ccs = {}
            for k in self.unit[i].knowledge['atom_ccs'].keys():
                atom_ccs[k] = self.unit[i].knowledge['atom_ccs'][k]

            if len(r) == 0:
                r = self.unit[i].data['radius']
            else:
                r = np.concatenate((r, self.unit[i].data['radius']))

            try:
                if len(c) == 0:
                    c = self.unit[i].data['charge']
                else:
                    c = np.concatenate((c, self.unit[i].data['charge']))
            except Exception as ex:
                skipcharge = True
                continue

        data[:, 1] = np.linspace(1, len(data), len(data)).astype(int)
        cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
        idx = np.arange(len(data))

        # create molecule, and push created data information
        M = Molecule()
        M.add_xyz(self.get_all_xyz())
        M.data = pd.DataFrame(data, index=idx, columns=cols)
        M.properties['center'] = M.get_center()
        M.knowledge['atom_ccs'] = atom_ccs
        M.data['radius'] = r

        if not skipcharge:
            M.data['charge'] = c



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

        :param indices: list of indices
        :param columns: list of columns (e.g. ["resname", "resid", "chain"])
        :returns: slice of molecule's data DataFrame
        '''

        return self.data.loc[indices, columns].values

    def write_pdb(self, outname):
        '''
        Write a pdb of the multimeric assembly.

        :param outname: name of PDB file to generate
        '''

        f_out = open(outname, "w")

        for f in range(len(self.unit[0].coordinates)):

            cnt = 1
            # set current state to new frame
            for j in range(0, len(self.unit), 1):
                self.unit[j].set_current(f)

        for j in range(0, len(self.unit), 1):
            # get data about points and their properties from the desired
            # protein structure
            d = self.unit[j].get_pdb_data()

            for i in range(0, len(self.unit[j].points), 1):
                # create and write PDB lin
                if d[i][2][0].isdigit():
                    L = '%-6s%5i %-5s%-4s%1s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], cnt, d[i][2], d[i][3], self.chain_names[j], int(d[i][5]), float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                else:
                    L = '%-6s%5i  %-4s%-4s%1s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], cnt, d[i][2], d[i][3], self.chain_names[j], int(d[i][5]), float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                #L='%-6s%5i  %-4s%-4s%1s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n'%(d[i][0], cnt, d[i][2], d[i][3], self.chain_names[j], d[i][5], d[i][6], d[i][7], d[i][8], d[i][9], d[i][10], d[i][11])
                f_out.write(L)
                cnt += 1

            f_out.write("TER\n")

        f_out.write("END\n")
        f_out.close()