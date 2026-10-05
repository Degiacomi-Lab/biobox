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

from copy import deepcopy
import numpy as np
from biobox.classes.structure import Structure
import pandas as pd


class Assembly(object):
    '''
    Construct and manipulate assemblies of multiple :func:`Structure <biobox.classes.structure.Structure>` instances.
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
        User-friendly names for these units are stored in the self.unit_labels dictionary, mapping every label to the position of its unit in self.unit. If no name is provided, a number will be assigned (as a string, starting from 0).
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
        remove all elements loaded in the assembly (units and their labels). The self.data DataFrame is not emptied.
        '''
        # restart arrays
        self.unit = []
        self.unit_labels = {}

    def load(self, struct, n):
        '''
        load n identical structures (homo assembly), as deep copies of struct appended to the existing units. Every copy keeps the current frame of struct, and is labelled with its position in the assembly (as a string).

        :param struct: object of class Structure (or subclasses)
        :param n: number of units
        '''
        dfs = [self.data]
        for i in range(len(self.unit), len(self.unit) + n, 1):
            e = deepcopy(struct)
            self.unit.append(e)
            self.unit_labels[str(i)] = i

            # every unit keeps the current frame of the structure it was copied from
            # (deepcopy breaks the view of points on coordinates)
            e.set_current(struct.current)

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
        add the structures contained in another assembly in the current one, via :func:`load <biobox.classes.assembly.Assembly.load>`. The added units are labelled with their position in the assembly, the labels of the merged assembly are not kept.

        :param assembly: object of class Assembly
        :param n: number of instances of assembly to merge (only one by default)
        '''
        atmp = deepcopy(assembly)
        for i in range(0, n, 1):
            for a in atmp.unit:
                self.load(a, 1)

    def append(self, structure, label=""):
        '''
        append a new :func:`Structure <biobox.classes.structure.Structure>` instance into an existing assembly. The structure is not copied, and "unit" and "unit_index" columns are added to its data.

        :param structure: :func:`Structure <biobox.classes.structure.Structure>` object to be appended to assembly
        :param label: name to give to the new unit. If not provided a default value equal to the rank of the new Structure in the assembly (as a string) will be assigned.
        :returns: label assigned to the new Structure in the assembly
        '''

        index = len(self.unit)
        if label == "":
            label = str(index)

        if str(label) in self.unit_labels:
            raise Exception("ERROR: label %s already existing in multimer!" %label)

        self.unit_labels[str(label)] = index
        self.unit.append(structure)


        #append structure to dataframe
        structure.data["unit"] = label
        structure.data["unit_index"] = structure.data.index
        self.data = pd.concat([self.data, structure.data])
        self.data.index = np.arange(len(self.data))

        return label

    def add_conformation(self, new_assembly):
        '''
        append a new :func:`Assembly <biobox.classes.assembly.Assembly>` instance into an existing assembly, as alternate conformation.

        The current coordinates of every unit of new_assembly are added as a new conformation of the corresponding unit, and every unit, as well as the assembly, is set to that new conformation.

        :param new_assembly: :func:`Assembly <biobox.classes.assembly.Assembly>` object to be appended as alternative conformation, with as many units as this assembly, each with the same number of points
        '''
        if len(self.unit) != len(new_assembly.unit):
            raise Exception("ERROR: expecting %s subunits, found %s!" %(len(self.unit), len(new_assembly.unit)))

        # check every unit before modifying any
        for i in range(0, len(self.unit), 1):
            if self.unit[i].coordinates.shape[1] != new_assembly.unit[i].coordinates.shape[1]:
                raise Exception("ERROR: subunit %s conformation should have %s atoms, but %s found!" %(i, self.unit[i].coordinates.shape[1], new_assembly.unit[i].coordinates.shape[1]))

        # the new conformation is appended after the existing ones of every unit
        for i in range(0, len(self.unit), 1):
            self.unit[i].add_xyz(new_assembly.unit[i].get_xyz())
            self.unit[i].set_current(len(self.unit[i].coordinates) - 1)

        self.current = len(self.unit[0].coordinates) - 1

    def load_list(self, struct_list, labels=[]):
        '''
        load a list of :func:`Structure <biobox.classes.structure.Structure>` objects with their associated labels list (typically for hetero assemblies). Deep copies of the structures are appended to the existing units, each keeping the current frame of its original.

        :param struct_list: list of :func:`Structure <biobox.classes.structure.Structure>` objects (or subclasses of it)
        :param labels: user-friendly names used to identify every structure (stored as strings). If empty, the position of every unit in the assembly is used.
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
        first = len(self.unit)
        for k, struct in enumerate(struct_list):
            i = first + k
            e = deepcopy(struct)
            self.unit.append(e)

            if len(labels) != 0:
                lbl = str(labels[k])
            else:
                lbl = str(i)

            self.unit_labels[lbl] = i

            #add labeling to structures tables, prior concatenation
            e.data["unit"] = lbl
            e.data["unit_index"] = e.data.index
            dfs.append(e.data)

            # deepcopy breaks the view of points on coordinates
            e.set_current(struct.current)


        #create dataframe collecting information from all structures
        self.data = pd.concat(dfs)
        self.data.index = np.arange(len(self.data))


    def make_structure(self):
        '''
        returns a :func:`Structure <biobox.classes.structure.Structure>` object containing all the points of the current conformation of all units, as a single conformation, with their radii.

        :returns: :func:`Structure <biobox.classes.structure.Structure>` object
        '''
        radii = np.concatenate([u.data["radius"].values for u in self.unit])
        return Structure(p=self.get_all_xyz(), r=radii)

    def make_curved_chain(self, angle, dist, groups=None):
        '''
        move loaded units so that they arrange in a bent chain in the xy plane.

        The i-th group is centered at the origin, rotated around z by i*angle, and translated by dist along the direction at i*angle from x, starting from the center of the previous group (or from the origin, for the first group).

        :param angle: chain curvature, i.e. rotation angle between consecutive groups, in degrees
        :param dist: distance between centers of geometry of consecutive groups
        :param groups: if set, list of lists of unit positions (in self.unit). A chain is formed by considering every group of loaded structures as a unique object.
                       If unset, every object is independently moved.
        '''

        # if no group has been selected, every subunit forms a group by itself
        if groups is None or len(groups) == 0:
            groups = [[i] for i in range(len(self.unit))]

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
        assemble the loaded units in a circular symmetry around the z axis.
        Supposes that all units are centered at the origin and oriented in the same way.

        Every unit is translated so that its point with largest x coordinate is placed at x = -radius and y = displacement (z is unchanged), and the i-th unit is then rotated around z by i*360/n degrees (n units).

        :param radius: radial displacement with respect of the origin (along x axis)
        :param displacement: tangential displacement (along y axis)
        '''
        for i in range(0, len(self.unit), 1):

            # get the extreme point on the x axis and move the atom corresponding to it to the origin
            # add to the translation a displacement along the x axis
            # corresponding to the requested radius
            xyzMaxIndex = np.argmax(self.unit[i].points, axis=0)
            maxAtom = self.unit[i].points[xyzMaxIndex[0]]
            self.unit[i].translate(-maxAtom[0] - radius, -maxAtom[1] + displacement, 0.0)

            # number of degrees to rotate
            angle = np.radians(i * (360.0 / float(len(self.unit))))
            Rz = np.array([[np.cos(angle), -(np.sin(angle)), 0],
                           [(np.sin(angle)), (np.cos(angle)), 0],
                           [0, 0, 1]])
            self.unit[i].apply_transformation(Rz.T)

    def make_stacked_rings(self, radius, z, t=0):
        '''
        construct a prism (two superimposed discs). Requires an even number of units.

        The second half of the units is rotated by 180 degrees around x, all units are translated by radius along x and t along y (the second half also by z along z), and the i-th unit of each half is then rotated around z by i*360/(n/2) degrees (n units).

        :param radius: radial displacement with respect of the origin (along x axis)
        :param z: vertical displacement of the second disc
        :param t: tangential displacement after radial displacement (along y axis)
        '''

        if np.mod(len(self.unit), 2) != 0:
            raise Exception("cannot build polyhedron, need an even number of units!")

        for i in range(0, int(len(self.unit) / 2.0), 1):

            # rotate the second half of subunits upside down
            self.unit[i + int(len(self.unit) / 2.0)].rotate(180.0, 0.0, 0.0)

            # move the subunits
            self.unit[i].translate(radius, t, 0)
            self.unit[i + int(len(self.unit) / 2.0)].translate(radius, t, z)

            # number of degree to rotate
            angle = np.radians(i * (360.0 / (float(len(self.unit) / 2.0))))
            Rz = np.array([[np.cos(angle), -(np.sin(angle)), 0],
                           [(np.sin(angle)), (np.cos(angle)), 0],
                           [0, 0, 1]])
            self.unit[i].apply_transformation(Rz)
            self.unit[i + int(len(self.unit) / 2.0)].apply_transformation(Rz)

    def make_prism(self, radius, z, a, b, c, t=0):
        '''
        construct a prism (bases only). Requires an even number of units. For a perfect stacking, units should be first aligned along their principal axes.

        Like :func:`make_stacked_rings <biobox.classes.assembly.Assembly.make_stacked_rings>`, with the units of the first half additionally rotated by (a, b, c) and those of the second half by (-a, -b, c) before being translated.

        :param radius: radial displacement with respect of the origin (along x axis)
        :param z: vertical displacement of the second base
        :param a: rotation around x axis, in degrees
        :param b: rotation around y axis, in degrees
        :param c: rotation around z axis, in degrees
        :param t: tangential displacement after radial displacement (along y axis)
        '''

        if np.mod(len(self.unit), 2) != 0:
            raise Exception("ERROR: cannot build polyhedron, need an even number of units!")

        for i in range(0, int(len(self.unit) / 2.0), 1):

            # rotate the second half of subunits upside down
            self.unit[i + int(len(self.unit) / 2.0)].rotate(180.0, 0.0, 0.0)

            # rotate everything by desired angles
            self.unit[i].rotate(a, b, c)
            self.unit[i + int(len(self.unit) / 2.0)].rotate(-a, -b, c)

            # move the subunits
            self.unit[i].translate(radius, t, 0)
            self.unit[i + int(len(self.unit) / 2.0)].translate(radius, t, z)

            # number of degree to rotate
            angle = np.radians(i * (360.0 / (float(len(self.unit) / 2.0))))
            Rz = np.array([[np.cos(angle), -(np.sin(angle)), 0],
                           [(np.sin(angle)), (np.cos(angle)), 0],
                           [0, 0, 1]])
            self.unit[i].apply_transformation(Rz)
            self.unit[i + int(len(self.unit) / 2.0)].apply_transformation(Rz)

    def rotate(self, x, y, z, unit=[]):
        '''
        rotate desired units in the assembly around the origin (see :func:`Structure.rotate <biobox.classes.structure.Structure.rotate>`).

        :param x: rotation around x, in degrees
        :param y: rotation around y, in degrees
        :param z: rotation around z, in degrees
        :param unit: list of labels indicating which units to rotate (string or integer also accepted, for a single subunit). If undefined, all units will be rotated.
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
        :param unit: list of labels indicating which units to translate (string or integer also accepted, for a single subunit). If undefined, all units will be translated.
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
        center whole assembly to origin, i.e. translate all units so that the center of geometry of all their points is at the origin.
        '''
        pos = self.get_all_xyz()
        center = np.mean(pos, axis=0)
        self.translate(-center[0], -center[1], -center[2])

    def get_all_xyz(self):
        '''
        extract all structures coordinates (current conformation of every unit) in a unique array.

        :returns: nx3 numpy array of the coordinates of all units, concatenated in unit order.
        '''
        pts = self.unit[0].get_xyz()
        for i in range(1, len(self.unit), 1):
            pts = np.concatenate((pts, self.unit[i].get_xyz()))

        return pts

    def get_uxyz(self):
        '''
        extract all structures coordinates in a list, where every element contains an array of coordinates of a unit (current conformation).

        :returns: list of numpy arrays, one per unit.
        '''
        return [self.unit[i].get_xyz() for i in range(len(self.unit))]

    def get_size(self):
        '''
        compute dimensions of the assembly along the x,y and z axes.

        .. note:: points VdW radii are not kept into account

        :returns: numpy array with the extent along x, y and z
        '''
        p = self.get_all_xyz()
        return np.max(p, axis=0) - np.min(p, axis=0)

    def contact_ratio(self, unit1, unit2):
        '''
        compute the fraction of the points of a unit falling within another unit, as tested by the check_inclusion method of the first unit (e.g. :func:`Ellipsoid.check_inclusion <biobox.classes.convex.Ellipsoid.check_inclusion>`).

        :param unit1: label of the unit whose volume is tested. Its class must provide a check_inclusion method (:func:`Sphere <biobox.classes.convex.Sphere>` and :func:`Ellipsoid <biobox.classes.convex.Ellipsoid>` do)
        :param unit2: label of the unit whose points are tested
        :returns: fraction of the points of unit2 inside unit1, between 0 and 1 (float), 0 if unit2 has no points
        '''
        u1 = self.unit_labels[str(unit1)]
        u2 = self.unit_labels[str(unit2)]
        if not hasattr(self.unit[u1], "check_inclusion"):
            raise Exception("ERROR: unit %s is a %s, which has no check_inclusion method" % (unit1, type(self.unit[u1]).__name__))
        inside = np.asarray(self.unit[u1].check_inclusion(self.unit[u2].points), dtype=bool)
        if len(inside) == 0:
            return 0.0
        return float(np.mean(inside))

    def get_buried(self):
        '''
        compute buried surface (assembly sum of components asa minus assembly asa), with :func:`sasa <biobox.measures.calculators.sasa>` and its default parameters.

        :returns: buried surface in A^2
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
        write a PDB file where every point is a bead, using the current conformation of every unit.

        As in :func:`Structure.write_pdb <biobox.classes.structure.Structure.write_pdb>`, every point is named SPH, occupancy is 1 and the radius is written in the beta factor column.
        The i-th unit (starting from 0) is given chain name chain_names[i] and residue number i. Atom serials run from 1 in file order, in hybrid-36 above 99999.

        :param filename: name of pdb file to be produced
        '''

        fout = open(filename, "w")

        serial = 0
        for i in range(0, len(self.unit), 1):
            radii = self.unit[i].data["radius"].values
            for j in range(0, len(self.unit[i].points), 1):
                serial += 1

                # occupancy 1, radius in the beta factor column
                l = (Structure._hybrid36(serial), "SPH", "SPH", self.chain_names[i],
                     i, self.unit[i].points[j, 0], self.unit[i].points[j, 1],
                     self.unit[i].points[j, 2], 1.0, radii[j], "C")
                L = 'ATOM  %5s  %-4s%-4s%1s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % l
                fout.write(L)

        fout.close()


    @ staticmethod
    def _components(fibertype):
        '''
        decompose a fiber type into the basic fiber types it is composed of.

        :param fibertype: name of the fiber type
        :returns: ['p2', 'pm'] for 'pmm', ['p2', 'cm'] for 'cmm', None for the types not implemented ('pmg', 'pgg', 'p31m', 'p3m1', 'p4g', 'p4m', 'p6m'), and [fibertype] for any other type
        '''
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
        calculate number of repeating units to be used to form a fiber.

        :param Lpx: distance of the partner (point that will be superimposed to the origin) along x as number of steps in a 2D tiling.
        :param Lpy: distance of the partner (point that will be superimposed to the origin) along y as number of steps in a 2D tiling.
        :param min_height: optional, minimal height (number of repeating units along y) of the fiber (if min_height < Lpy, Lpy will be used as height of the fiber). Default is 10.
        :param fibertype: optional, fiber type. If set, the number of units along x is multiplied by the number of basic fiber types it is composed of (see make_fiber).
        :returns: number of units along x (Lpx, times the number of components of fibertype if set)
        :returns: number of units along y, max(min_height, Lpy)
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

        .. warning:: experimental method, not validated against reference geometries. Known issues: the row of the n-th unit is computed as n / Lpx without rounding, so units sit at fractional rows; the composite fiber types ('pmm', 'cmm') do not combine their component transformations correctly; min_height has no effect.

        The n-th unit is placed in the tiling at column n % Lpx and row n / Lpx, and the current coordinates of every unit (taken as coordinates relative to its tile) are replaced by their position in the rolled fiber.
        Lpx must be a multiple of the number of units per tile of the fiber type, and Lpy must be even for fiber types involving 'p1hexagonal', 'p2', 'p3', 'p4', 'p6', 'pg' or 'pm'.

        :param vx: distance between two first neighbors along x in a 2D tiling.
        :param Lpx: distance of the partner (point that will be superimposed to the origin) along x as number of steps in a 2D tiling.
        :param Lpy: distance of the partner (point that will be superimposed to the origin) along y as number of steps in a 2D tiling.
        :param vy: optional, distance between two first neighbors along y in a 2D tiling. Used only by 'p1oblique' and 'p1rectangular' fiber types, for the other types it is computed from vx.
        :param gamma: optional, angle between vx and vy in rad, used by 'p1oblique' fiber type and ignored for other fiber types (default is pi/2, equivalent to p1rectangular).
        :param v: optional, additional parameter needed for 'pm', 'pg', 'cm', 'p2', ... fiber types (default is 0). List with one value per component for composite fibertypes ('pmm', 'cmm').
        :param min_height: optional, passed to :func:`num_units_fiber <biobox.classes.assembly.Assembly.num_units_fiber>`, of which only the number of units along x is used, so it does not affect the result. Default is 2.
        :param fibertype: optional, one of 'p1rectangular', 'p1oblique', 'p1hexagonal', 'pm', 'pg', 'cm', 'p2', 'p3', 'p4', 'p6', 'pmm', 'cmm' (default is 'p1oblique').
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




