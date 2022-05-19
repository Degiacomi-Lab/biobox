# Copyright (c) 2014-2022 Matteo Degiacomi
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

from copy import deepcopy
import numpy as np
from biobox.classes.structure import Structure
import pandas as pd


class Assembly(object):
    '''
    Construct and manipulate assemblies of multiple :func:`Structure <structure.Structure>` instances.
    '''

    # labels for chain names (will be assigned to individual members of the
    # assembly upon PDB creation).
    chain_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd',
                   'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                   'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    def __init__(self):
        '''
        An Assembly is composed of several building blocks (instances of Structure class) referred to as "unit", and stored in the self.unit list.
        User-friendly names for these unites are stored in the self.unit_labels list. If no name is provided, a number will be assigned (as a string, starting from 0).
        '''

        # list of Structure instances (or subclasses).
        self.unit = []
        # labels assigned to every structure. The length of this dictionary is
        # equal to that of unit.
        self.unit_labels = {}

        # current conformation selected from conformational database
        self.current = -1

        #metadata associated to every point
        self.data = pd.DataFrame(index=[], columns=[])


    def clear(self):
        '''
        remove all elements loaded in the assembly.
        '''
        # restart arrays
        self.unit = []
        self.unit_labels = {}

    def load(self, struct, n):
        '''
        load a list of identical structures (homo assembly).

        :param n: number of units
        :param struct: object of class Structure (or subclasses)
        '''
        dfs = [self.data]
        for i in range(len(self.unit), len(self.unit) + n, 1):
            e = deepcopy(struct)
            self.unit.append(e)
            self.unit_labels[str(i)] = i

            e.points = e.coordinates.view()[0]

            #add labeling to structures tables, prior concatenation
            e.data["unit"] = str(i)
            e.data["unit_index"] = e.data.index
            dfs.append(e.data)

        #create dataframe collecting information from all structures
        self.data = pd.concat(dfs)
        self.data.index = np.arange(len(self.data))

        self.current = 0

    def merge(self, assembly, n=1):
        '''
        add the structures contained in another assembly in the current one.

        :param assembly: object of class Assembly
        :param n: number if instances of assembly to merge (only one by default)
        '''
        atmp = deepcopy(assembly)
        for i in range(0, n, 1):
            for a in atmp.unit:
                self.load(a, 1)

    def append(self, structure, label=""):
        '''
        append a new :func:`Structure <structure.Structure>` instance into an existing assembly

        :param structure: :func:`Structure <structure.Structure>` object to be appended to assembly
        :param label: name to give to the assembly. If not provided a default value equal to the rank of the new Structure in the assembly will be assigned.
        :returns: label assigned to the new Structure in the assembly
        '''

        if label == "":
            self.unit_labels[str(len(self.unit) - 1)] = len(self.unit) - 1
            self.unit.append(structure)
        if label != "":
            if label not in self.unit_labels.keys():
                self.unit_labels[str(label)] = len(self.unit) - 1
                self.unit.append(structure)
            else:
                raise Exception("ERROR: label %s already existing in multimer!" %label)


        #append structure to dataframe
        structure.data["unit"] = label
        structure.data["unit_index"] = structure.data.index
        self.data = pd.concat([self.data, structure.data])
        self.data.index = np.arange(len(self.data))

        return label

    def add_conformation(self, new_assembly):
        '''
        append a new :func:`Assembly <assembly.Assembly>` instance into an existing assembly, as alternate conformation

        :param new_assembly: :func:`Assembly <assembly.Assembly>` object to be appended as alternative conformation
        '''
        if len(self.unit) != len(new_assembly.unit):
            raise Exception("ERROR: expecting %s subunits, found %s!" %(len(self.unit), len(new_assembly.unit)))

        self.current += 1

        for i in range(0, len(self.unit), 1):
            if self.unit[i].coordinates.shape[1] != new_assembly.unit[i].coordinates.shape[1]:
                raise Exception("ERROR: subunit %s conformation should have %s atoms, but %s found!" %(i, self.unit[i].coordinates.shape[0], new_assembly.unit[i].coordinates.shape[0]))

            self.unit[i].add_xyz(new_assembly.unit[i].get_xyz())
            self.unit[i].set_current(self.current)

    def load_list(self, struct_list, labels=[]):
        '''
        load a list of :func:`Structure <structure.Structure>` objects with their associated labels list (typically for hetero assemblies).

        :param struct_list: :func:`Structure <structure.Structure>` objects (or subclasses of it)
        :param labels: user-friendly names used to identify every structure. If empty, simple incremental integers are used.
        '''

        # check labels consistency
        if len(labels) != 0:
            if len(struct_list) != len(labels):
                raise Exception(
                    "ERROR: structures and labels lists have different length!")

            if len(np.unique(np.array(labels))) != len(labels):
                raise Exception(
                    "ERROR: duplicate label found in provided list!")

            # check that labels are all different, and that they don't already
            # exist in the list
            for l in labels:
                if l in self.unit_labels:
                    raise Exception("ERROR: label %s already exists!" % l)

        # append new structures to old ones
        dfs = [self.data]
        for i in range(len(self.unit), len(self.unit) + len(struct_list), 1):
            # create dictionary with neighbors
            e = deepcopy(struct_list[i])
            self.unit.append(e)

            if len(labels) != 0:
                lbl = labels[i]
            else:
                lbl = str(i)

            self.unit_labels[lbl] = i

            #add labeling to structures tables, prior concatenation
            e.data["unit"] = lbl
            e.data["unit_index"] = e.data.index
            dfs.append(e.data)

            e.points = e.coordinates.view()[0]


        #create dataframe collecting information from all structures
        self.data = pd.concat(dfs)
        self.data.index = np.arange(len(self.data))


    def make_structure(self):
        '''
        returns a :func:`Structure <structure.Structure>` object containing all the points of the assembly.

        :returns: :func:`Structure <structure.Structure>` object
        '''
        return Structure(p=self.get_all_xyz())

    def make_curved_chain(self, angle, dist, groups=[]):
        '''
        move loaded units so that they arrange in a bent chain.

        :param angle: chain curvature
        :param dist: distance between centers of mass
        :param groups: if set, a chain is formed by considering groups of loaded structures as unique objects.
                       If unset, every object is independently moved.
        '''

        # if no group has been selected, every subunit forms a group by itself
        if len(groups) == 0:
            for i in range(0, len(self.unit), 1):
                groups.append([i])

        # keep track of the position of previous member of chain
        last_center = np.array([0.0, 0.0, 0.0])

        # print groups, len(self.unit), len(groups)

        for i in range(0, len(groups), 1):
            # get group center, will be used to center it to the origin
            pts = self.unit[groups[i][0]].get_xyz()
            for j in range(1, len(groups[i]), 1):
                pts = np.concatenate((pts, self.unit[groups[i][j]].get_xyz()))

            current_center = np.mean(pts, axis=0)

            # center group, rotate it, and send it to designated area (element
            # by element)
            for j in range(0, len(groups[i]), 1):
                self.unit[groups[i][j]].translate(-current_center[0], -current_center[1], -current_center[2])
                self.unit[groups[i][j]].rotate(0, 0, angle * (i))

                x = last_center[0] + dist * np.cos(np.radians(angle * (i)))
                y = last_center[1] + dist * np.sin(np.radians(angle * (i)))
                self.unit[groups[i][j]].translate(x, y, 0.0)

            # compute group center after rototranslation, and store it for next
            # iteration
            pts = self.unit[groups[i][0]].get_xyz()
            for j in range(1, len(groups[i]), 1):
                pts = np.concatenate((pts, self.unit[groups[i][j]].get_xyz()))

            last_center = np.mean(pts, axis=0)

    def make_circular_symmetry(self, radius, displacement=0):
        '''
        assemble the loaded units in a circular symmetry.
        Supposes that all units are centered at the origin and oriented in the same way.

        :param radius: radial displacement with respect of the origin (along x axis)
        :param displacement: tangential displacement
        '''
        for i in range(0, len(self.unit), 1):

            # get the extreme point on the x axis and move the atom corresponding to it to the origin
            # add to the translation a displacement along the x axis
            # corresponding to the requested radius
            xyzMaxIndex = np.argmax(self.unit[i].points, axis=0)
            maxAtom = self.unit[i].points[xyzMaxIndex[0]]
            self.translate(-maxAtom[0] - radius, -maxAtom[1] + displacement, 0.0, i)

            # number of degrees to rotate
            angle = np.radians(i * (360.0 / float(len(self.unit))))
            Rz = np.array([[np.cos(angle), -(np.sin(angle)), 0],
                           [(np.sin(angle)), (np.cos(angle)), 0],
                           [0, 0, 1]])
            self.unit[i].apply_transformation(Rz.T)

    def make_stacked_rings(self, radius, z, t=0):
        '''
        construct a prism (two superimposed discs)

        :param radius: radial displacement with respect of the origin (along x axis)
        :param z: vertical displacement
        :param t: tangential displacement after radial displacement (along y axis)
        '''

        if np.mod(len(self.unit), 2) != 0:
            raise Exception("cannot build polyhedron, need an even number of units!")

        for i in range(0, int(len(self.unit) / 2.0), 1):

            # rotate the second half of subunits upside down
            self.rotate(180.0, 0.0, 0.0, i + int(len(self.unit) / 2.0))

            # move the subunits
            self.translate(radius, t, 0, i)
            self.translate(radius, t, z, i + int(len(self.unit) / 2.0))

            # number of degree to rotate
            angle = np.radians(i * (360.0 / (float(len(self.unit) / 2.0))))
            Rz = np.array([[np.cos(angle), -(np.sin(angle)), 0],
                           [(np.sin(angle)), (np.cos(angle)), 0],
                           [0, 0, 1]])
            self.unit[i].apply_transformation(Rz)
            self.unit[i + int(len(self.unit) / 2.0)].apply_transformation(Rz)
            self.translate(0, 0, z, i + int(len(self.unit) / 2.0))

    def make_prism(self, radius, z, a, b, c, t=0):
        '''
        construct a prism (bases only). For a perfect stacking, units should be first aligned along their principal axes.

        :param radius: radial displacement with respect of the origin (along x axis)
        :param z: vertical displacement
        :param a: rotation along x axis
        :param b: rotation along y axis
        :param c: rotation along z axis
        :param t: tangential displacement after radial displacement (along y axis)
        '''

        if np.mod(len(self.unit), 2) != 0:
            raise Exception("ERROR: cannot build polyhedron, need an even number of units!")

        for i in range(0, int(len(self.unit) / 2.0), 1):

            # rotate the second half of subunits upside down
            self.rotate(180.0, 0.0, 0.0, i + int(len(self.unit) / 2.0))

            # rotate everything by desired angles
            self.rotate(a, b, c, i)
            self.rotate(-a, -b, c, i + int(len(self.unit) / 2.0))

            # move the subunits
            self.translate(radius, t, 0, i)
            self.translate(radius, t, z, i + int(len(self.unit) / 2.0))

            # number of degree to rotate
            angle = np.radians(i * (360.0 / (float(len(self.unit) / 2.0))))
            Rz = np.array([[np.cos(angle), -(np.sin(angle)), 0],
                           [(np.sin(angle)), (np.cos(angle)), 0],
                           [0, 0, 1]])
            self.unit[i].apply_transformation(Rz)
            self.unit[i + int(len(self.unit) / 2.0)].apply_transformation(Rz)
            self.translate(0, 0, z, i + int(len(self.unit) / 2.0))

    def rotate(self, x, y, z, unit=[]):
        '''
        rotate desired units in the assembly.

        :param x: rotation around x
        :param y: rotation around y
        :param z: rotation around z
        :param unit: list of labels indicating which units to rotate (string or integer also accepted, for a single subunit). If undefind, all units will be rotated.
        '''
        if isinstance(unit, list):
            # rotate everything
            if len(unit) == 0:
                for i in range(0, len(self.unit), 1):
                    self.unit[i].rotate(x, y, z)
            # rotate desired units
            else:
                for u in unit:
                    label = self.unit_labels[str(u)]
                    self.unit[label].rotate(x, y, z)

        elif isinstance(unit, int) or isinstance(unit, str):
            label = self.unit_labels[str(unit)]
            self.unit[label].rotate(x, y, z)

        else:
            raise Exception("ERROR: unit keyword should be integer, float, list or numpy array!")

    def translate(self, x, y, z, unit=[]):
        '''
        translate desired units in the assembly.

        :param x: translation along x
        :param y: translation along y
        :param z: translation along z
        :param unit: list of labels indicating which units to translate (string or integer also accepted, for a single subunit). If undefind, all units will be translated.
        '''

        if isinstance(unit, list):
            # translate everything
            if len(unit) == 0:
                for i in range(0, len(self.unit), 1):
                    self.unit[i].translate(x, y, z)
            # translate desired units
            else:
                for u in unit:
                    label = self.unit_labels[str(u)]
                    self.unit[label].translate(x, y, z)

        elif isinstance(unit, int) or isinstance(unit, str):
            label = self.unit_labels[str(unit)]
            self.unit[label].translate(x, y, z)

        else:
            raise Exception("ERROR: unit keyword should be integer, float, list or numpy array!")

    def center_subunit(self, unit=-1):
        '''
        center individual subunit to origin.

        :param unit: label of unit to center. If undefined, all units will be individually centered.
        '''

        if unit == -1:
            for i in range(0, len(self.unit), 1):
                self.unit[i].center_to_origin()
        else:
            u = self.unit_labels[str(unit)]
            self.unit[u].center_to_origin()

    def center_assembly(self):
        '''
        center whole assembly to origin.
        '''
        pos = self.get_all_xyz()
        center = np.mean(pos, axis=0)
        self.translate(-center[0], -center[1], -center[2])

    def get_all_xyz(self):
        '''
        extract all structures coordinates in a unique array.

        :returns: collapsed version of assembly's atoms coordinates.
        '''
        pts = self.unit[0].get_xyz()
        for i in range(1, len(self.unit), 1):
            pts = np.concatenate((pts, self.unit[i].get_xyz()))

        return pts

    def get_uxyz(self):
        '''
        extract all structures coordinates in a a list, where every element contains an array of coordinates of a unit.

        :returns: list of units coordinates.
        '''
        pts = []
        for i in range(0, len(self.unit), 1):
            pts.append(self.unit[i].get_xyz())

        return np.array(pts)

    def get_size(self):
        '''
        compute dimensions of the structure along the x,y and z axes.

        .. note:: points VdW radii are not kept into account
        '''
        p = self.get_all_xyz()
        return np.max(p, axis=0) - np.min(p, axis=0)

    def contact_ratio(self, unit1, unit2):
        '''
        count the number of surface points in contact between two Structures part of the assembly.

        :returns: number of contacts
        '''
        u1 = self.unit_labels[str(unit1)]
        u2 = self.unit_labels[str(unit2)]
        contacts = self.unit[u1].check_inclusion(self.unit[u2].points)
        return float(contacts)

    def get_buried(self):
        '''
        compute buried surface (assembly sum of components asa minus assembly asa).

        :returns: buried_surface in A^2
        '''

        from biobox.measures.calculators import sasa

        # sum asa of individual components
        asa = 0
        for i in range(0, len(self.unit), 1):
            asa += sasa(self.unit[i])[0]

        # subtract assembly asa
        asa -= sasa(self)[0]
        return asa

    def write_pdb(self, filename):
        '''
        write a PDB file where every atom is a bid. VdW radius is written into beta factor.

        :param filename: name of pdb file to be produced
        '''

        fout = open(filename, "w")

        for i in range(0, len(self.unit), 1):
            for j in range(0, len(self.unit[i].points), 1):
                if len(self.data) > 99999:
                    idx_val = hex(1 + i + len(self.unit) * j).split('x')[1]  # remove 0x at start of hexadecimal number
                else:
                    idx_val = 1 + i + len(self.unit) * j

                l = (idx_val, "SPH", "SPH", self.chain_names[i],
                     i, self.unit[i].points[j, 0], self.unit[i].points[j, 1],
                     self.unit[i].points[j, 2], self.unit[i].data["radius"][j], 1.0, "C")
                L = 'ATOM  %5s  %-4s%-4s%1s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % l
                fout.write(L)

        fout.close()


    @ staticmethod
    def _components(fibertype):
        if fibertype == 'pmm':
            return ['p2', 'pm']

        # TODO
        elif fibertype == 'cmm':

            return ['p2', 'cm']
        # TODO
        elif fibertype == 'pmg':

            return

        # TODO
        elif fibertype == 'pgg':

            return

        # TODO
        elif fibertype == 'p31m':

            return

        # TODO
        elif fibertype == 'p3m1':

            return

        # TODO
        elif fibertype == 'p4g':

            return

        # TODO
        elif fibertype == 'p4m':

            return

        # TODO
        elif fibertype == 'p6m':

            return

        else:
            return [fibertype]

    @ staticmethod
    def num_units_fiber(Lpx, Lpy, min_height=10, fibertype=None):
        '''
        calculate number of repeting units to be used to form a fiber.

        :param Lpx: distance of the partner (point that will be superimposed to the origin) along x as number of steps in a 2D tiling.
        :param Lpy: distance of the partner (point that will be superimposed to the origin) along y as number of steps in a 2D tiling.
        :param min_height: optional, minimal height (number of repeting units along y) of the fiber (if min_height < Lpy, Lpy will be used as height of the fiber). Default is 10.
        '''

        Nx = Lpx
        Ny = max(min_height, Lpy)

        if fibertype:
            components = Assembly._components(fibertype)
            n = len(components)
        else:
            n = 1

        return n * Nx, Ny

    def make_fiber(self, vx, Lpx, Lpy, vy=None, gamma=np.pi/2, v=0, min_height=2, fibertype='p1oblique'):
        '''
        create a fiber, seen as the rolling of a plane with (vx, vy) tiling such that the repeating unit in position (Lpx, Lpy) will be overlapped to the origin.

        :param vx: distance between two first neighbors along x in a 2D tiling.
        :param Lpx: distance of the partner (point that will be superimposed to the origin) along x as number of steps in a 2D tiling.
        :param Lpy: distance of the partner (point that will be superimposed to the origin) along y as number of steps in a 2D tiling.
        :param vy: optional, distance between two first neighbors along y in a 2D tiling (needed for 'p1oblique', 'p1rectangular', 'pm', 'pg', 'p2', ... fiber types).
        :param gamma: optional, angle between vx and vy in rad, needed for 'p1oblique' fiber type and ignored for other fiber types (default is pi/2 - equivalent to p1rectangular).
        :param min_height: optional, minimal height (number of repeting units along y) of the fiber (if min_height < Lpy, Lpy will be used as height of the fiber). Default is 2.
        :param fibertype: optional, can be 'p1rectangular', 'p1hexagonal', 'p1oblique', pm', 'pg', 'cm', ... (default is 'p1oblique').
        :param v: optional, additional parameter needed for 'pm', 'pg', 'cm', 'p2', ... fiber types (default is None). List for composite fibertypes.
        '''

        if type(v) == int or type(v) == float:
            vlist = [v]
        else:
            vlist = list(v)

        if fibertype not in ['p1rectangular', 'p1oblique', 'p1hexagonal', 'pm', 'pg', 'cm', 'p2', 'p3', 'p4', 'p6', 'pmm', 'cmm']:
            raise Exception("fibertype %s not valid." %(fibertype))

        def lvalue(Lx, Ly, vx, vy):
            return np.sqrt((Lx * vx) ** 2 + (Ly * vy) ** 2)

        def thetavalue(Lx, Ly, vx, vy):
            try:
                return np.arctan2(Ly * vy , Lx * vx)
            except ZeroDivisionError:
                return np.sign(Ly) * np.pi / 2

        def phivalue(L, theta, thetap, Lp):
            return (2 * np.pi * L * np.cos(theta - thetap) / Lp) + np.pi
            # return (2 * np.pi * L * np.cos(theta - thetap) / Lp)


        def coords_in_fiber(Lx, Ly, vx, vy, Lp, x0, y0, z0, thetap):
            L = lvalue(Lx, Ly, vx, vy)
            theta = thetavalue(Lx, Ly, vx, vy)
            phi = phivalue(L, theta, thetap, Lp)
            Lp2pi = float(Lp) / (2 * np.pi)
            x2 = (Lp2pi + x0) * np.cos(phi) + z0 * np.sin(phi)
            z = - (Lp2pi + x0) * np.sin(phi) + z0 * np.cos(phi)
            x = x2 + Lp2pi
            y = L * np.sin(theta - thetap) + y0

            return x, y, z


        transformations = self._components(fibertype)

        if len(vlist) != len(transformations):
            raise Exception("%s parameters needed for %s fiber but only %s passed." %(len(transformations), fibertype, len(vlist)))

        psi = None

        nunitsdict ={'p1oblique': 1,
        'p1rectangular': 1,
        'p1hexagonal': 1,
        'p2': 2,
        'p3': 3,
        'p4': 2,
        'p6': 6,
        'pm': 2,
        'pg': 1,
        'cm': 2} # or cm 1?

        nunits = [nunitsdict[t] for t in transformations]
        nunits_tot = sum(nunits)
        t_order = [[(t, v)] * n for (t, v, n) in zip(transformations, vlist, nunits)]
        t_order = [v for sublist in t_order for v in sublist]


        if Lpx % nunits_tot != 0:
            raise Exception(
                    "ERROR: Lpx must be a multiple of %s when doing a %s tiling!" %(nunits_tot, fibertype))


        if 'p1hexagonal' in transformations or 'p2' in transformations or 'p3' in transformations or 'p4' in transformations or 'p6' in transformations or 'pg' in transformations or 'pm' in transformations:
            if Lpy % 2 == 1:
                raise Exception(
                    "ERROR: Lpy must be even when doing a %s tiling!" %(fibertype))
            vy = np.sqrt(3) * vx / 2
            if 'p6' in transformations:
                vy = 3 * vy

        if 'p1oblique' in transformations:
            vy = vy * np.sin(gamma)

        if 'p2' in transformations:
            psi = np.arcsin(float(vy) / np.sqrt(vx ** 2 + vy ** 2))

        if 'p4' in transformations or 'cm' in transformations:
            vy = vx

        Lp = lvalue(Lpx, Lpy, vx, vy)
        thetap = thetavalue(Lpx, Lpy, vx, vy)
        Nx, _ = Assembly.num_units_fiber(Lpx, Lpy, min_height=min_height)

        def basic_transform(u, Lx, Ly, fibertype, vx, vy, gamma, v, psi):
            if fibertype == 'p1hexagonal' and Ly % 2 == 1:
                Lx += .5

            elif fibertype == 'p1oblique' and Ly % 2 == 1:
                Lx += vy * np.cos(gamma)

            elif fibertype == 'p2':
                odd = (Ly % 2 == 1)
                if Lx % 2 == 1:
                    u.rotate(180, 0, 0) # why around x on not z??
                    Lx += v * np.cos(psi) / vx
                    Ly -= v * np.sin(psi) / vy
                    Lx -= 1
                else:
                    Lx -= v * np.cos(psi) / vx
                    Ly += v * np.sin(psi) / vy
                if odd:
                    Lx += 1

            elif fibertype == 'p3':
                odd = (Ly % 2 == 1)
                if Lx % 3 == 1:
                    u.rotate(-120, 0, 0) # why around x on not z??
                    Lx -= v * np.cos(np.pi / 6) / vx
                    Ly -= v * np.sin(np.pi / 6) / vy
                    Lx -= 1
                elif Lx % 3 == 2:
                    u.rotate(120, 0, 0) # why around x on not z??
                    Lx += v * np.cos(np.pi / 6) / vx
                    Ly -= v * np.sin(np.pi / 6) / vy
                    Lx -= 2
                else:
                    Ly += float(v) / vy
                if odd:
                    Lx += 1.5

            elif fibertype == 'p4':
                if Lx % 2 == 1:
                    if Ly % 2 == 1:
                        u.rotate(-90, 0, 0)
                        Lx -= v * np.cos(np.pi / 4) / vx
                    else:
                        u.rotate(180, 0, 0)
                        Lx += v * np.cos(np.pi / 4) / vx
                        Lx -= 1

                else:
                    if Ly % 2 == 1:
                        u.rotate(90, 0, 0)
                        Lx += v * np.cos(np.pi / 4) / vx
                        Lx += 1
                    else:
                        Lx -= v * np.cos(np.pi / 4) / vx
                    Ly -= v * np.sin(np.pi / 4) / vy

            elif fibertype == 'p6':
                odd = (Ly % 2 == 1)
                if Lx % 6 == 1:
                    u.rotate(-60, 0, 0) # why around x on not z??
                    Lx += v * np.cos(np.pi / 6) / vx
                    Ly += v * np.sin(np.pi / 6) / vy
                    Lx -= 1
                elif Lx % 6 == 2:
                    u.rotate(-120, 0, 0) # why around x on not z??
                    Lx += v * np.cos(np.pi / 6) / vx
                    Ly -= v * np.sin(np.pi / 6) / vy
                    Lx -= 2
                elif Lx % 6 == 3:
                    u.rotate(-180, 0, 0) # why around x on not z??
                    Ly -= float(v) / vy
                    Lx -= 3
                elif Lx % 6 == 4:
                    u.rotate(-240, 0, 0) # why around x on not z??
                    Lx -= v * np.cos(np.pi / 6) / vx
                    Ly -= v * np.sin(np.pi / 6) / vy
                    Lx -= 4
                elif Lx % 6 == 5:
                    u.rotate(-300, 0, 0) # why around x on not z??
                    Lx -= v * np.cos(np.pi / 6) / vx
                    Ly += v * np.sin(np.pi / 6) / vy
                    Lx -= 5
                else:
                    Ly += float(v) / vy
                if odd:
                    Lx += 3

            elif  fibertype == 'pm':
                if Lx % 2 == 0:
                    Lx += float(v) / vx
                else:
                    u.rotate(0, 180, 0)
                    Lx -= 1 + float(v) / vx

            elif fibertype == 'pg':
                if Ly % 2 == 0:
                    Lx += float(v) / vx
                else:
                    u.rotate(0, 180, 0)
                    Lx -= float(v) / vx

            elif fibertype == 'cm':
                if Lx % 2 == 0:
                    Lx += float(v) / vx
                else:
                    u.rotate(0, 180, 0)
                    Lx -= 1 + float(v) / vx
                if Ly % 2 == 1:
                    Lx += 1

            return Lx, Ly

        # print(zip(transformations, vlist))

        composite = (len(transformations) > 1)
        if not composite:
            t = transformations[0]
            v = vlist[0]
            for n, u in enumerate(self.unit):
                Ly = n / Nx
                Lx = n % Nx
                Lx, Ly = basic_transform(u, Lx, Ly, t, vx, vy, gamma, v, psi)
                coords_fiber = np.array([coords_in_fiber(Lx, Ly, vx, vy, Lp, x0, y0, z0, thetap) for [x0, y0, z0] in u.get_xyz()])
                self.unit[n].set_xyz(coords_fiber)

        else:
            for n, u in enumerate(self.unit):
                Ly = n / Nx
                Lx = n % Nx

                Lx1 = Lx
                Ly1 = Ly
                for i, (t, v) in enumerate(zip(transformations, vlist)):
                    nu = nunitsdict[t]
                    if n >= nu * i:
                        Lxt, Lyt = basic_transform(u, Lx, Ly, t, vx, vy, gamma, v, psi)
                        Lx1 += Lxt
                        Lyt += Lyt

                Lx = Lx1
                Ly = Ly1
                coords_fiber = np.array([coords_in_fiber(Lx, Ly, vx, vy, Lp, x0, y0, z0, thetap) for [x0, y0, z0] in u.get_xyz()])
                self.unit[n].set_xyz(coords_fiber)




