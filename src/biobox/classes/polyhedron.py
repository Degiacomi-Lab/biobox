# Copyright (c) 2014-2022 Matteo Degiacomi
#
# Biobox is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# Biobox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Biobox ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author : Matteo Degiacomi, matteo.degiacomi@gmail.com

import os
from copy import deepcopy
import numpy as np
from biobox.classes.structure import Structure
from biobox.classes.assembly import Assembly


class Polyhedron(Assembly):
    '''
    Subclass of :func:`Assembly <assembly.Assembly>`, allowing the assembly of polyhedral symmetries.
    After instantiation, the first method to be called is :func:`setup_polyhedron <polyhedron.Polyhedron.setup_polyhedron>`
    '''

    def setup_polyhedron(self, polyname, M, dbfilename=""):
        '''
        load information for the generation of a polyhedral assembly.

        It loads the desired :func:`Structure <structure.Structure>` to be used as building block (it automatically centers and alignes it).
        it also loads and geometric information about the desired polyhedron from database.
        database contains this information: [polymer_name number_of_vertices numbe_of_edges vertices_coordinates vertices_connectivity connectivity_type]

        * Vertices_coordinates contains a list of 3D positions, formatted as x1 y1 z1 x2 y2 z2,...
        * Connectivity lists couples of vertices ID being connected, formatted as conn1_1 conn1_2 conn2_1 conn2_2...
        * conn_type allows to group edges, so that rotation angles can be applied to subgroups of those. This is a list of numbers, that should include zero as lowest number.

        A default conn_type is provided by database, the user can however regroup the edges at will.

        .. note:: This method has to be called first.

        :param polyname: name of the polyhedron to be assembled. The name should be located in the polyhedra database.
        :param M: monomer to be used as building block. Should be an instance of Structure class.
        :param dbfilename: polyhedra database file.
        '''

        if len(dbfilename) == 0:
            folder = os.path.dirname(os.path.realpath(__file__))
            folder = os.sep.join(folder.split(os.sep)[:-1])
            dbfilename = "%s%sdata%spolyhedron_database.dat" %(folder, os.sep, os.sep)

        if os.path.isfile(dbfilename) != 1:
            raise Exception("ERROR: %s not found!" % dbfilename)

        # name of file containing polyhedra database
        self.dbfilename = dbfilename
        # name of polyhedron
        self.polyname = polyname

        # search information in new format database about desired polimer
        try:
            self.edges, self.v, self.conn, self.conn_type = self._search_database(polyname, dbfilename=self.dbfilename)
        except Exception as e:
            raise Exception("%s" % e)

        # monomeric subunit for polyhedron construction (instance of class
        # Molecule)
        self.building_block = deepcopy(M)
        self.building_block.align_axes()

        # box size
        dimensions = self.building_block.get_size()
        # box size along major axis (length)
        self.L = dimensions[0]
        # box size along second axis (width)
        self.W = dimensions[1]
        # box size along third axis (height)
        self.H = dimensions[2]

        # deformation database: [edge_id class x y z]. Edges having the same class will be scaled by the same coefficient.
        #@todo insert information about deformations directly into database. At the moment, it has to be explicitely provided via add_deformation method.
        self.deform = np.array([])

    def add_deformation(self, edges, vector=[]):
        '''
        Add entry in deformation database. Can be called only after the setup_polyhedron method.

        :param edges: integer or list of integers, defining edges indices that should be subjected to deformation.
        :param vector: axis along which deformation should take place. If not given, radial deformation will be assumed.
        '''

        # determine index for new entry in database
        if len(self.deform) == 0:
            idx = 0
        else:
            idx = np.max(self.deform[:, 1]) + 1

        # transform integer into list, in case an interger only is provided
        if isinstance(edges, int):
            e = [edges]
            edges = e

        # iterate over all edges and create new entries for database
        tmp = []
        for i in edges:

            # add deformation axis, if given
            if len(vector) == 3:
                vector /= np.linalg.norm(vector)
                tmp.append([i, idx, vector[0], vector[1], vector[2]])

            # axis not given, assume radial deformation
            else:
                ax = self.v[i] / np.linalg.norm(self.v[i])
                tmp.append([i, idx, ax[0], ax[1], ax[2]])

        # update deformation database
        if len(self.deform) > 0:
            self.deform = np.concatenate((self.deform, np.array(tmp)))
        else:
            self.deform = np.array(tmp)

    def generate_polyhedron(self, S, alpha, beta, gamma, deformation=[], add_conformation=False):
        '''
        Given polyhedron information previously loaded (using setup_polyhedron), generate an appropriate symmetry figure.

        :param S: polyhedron size (radius of mean sphere).
        :param alpha: rotation on molecule's first principal axis.
        :param beta: rotation on molecule's second principal axis.
        :param gamma: rotation on molecule's third principal axis.
        :param deformation: if list is provided, deformations will be applied according to self.deformation database.\n
               deformation list length should be equal to the amount of deformation classes)
        :param add_conformation: if True, the coordinates of the new poyhedron will be added to the conformational database as a new alternative conformation.\n
                If False, old polyhedron coordianates will be substituted.
        '''

        # if deformation coefficients are given, check first that they match
        # the number of classes in deformation database
        if len(deformation) > 0 and len(deformation) != len(np.unique(self.deform[:, 1])):
            raise Exception("ERROR: %s deformation coefficients expected, but %s found" % (len(deformation), np.unique(self.deform[:, 1])))

        self.psi, self.phi, self.nu, self.circumradius, self.midradius = self.get_polyhedron_properties(S)

        if not add_conformation:
            # clear any previous polyhedron and load anew
            self.clear()
            self.load(self.building_block, self.edges)
            self.current = 0
        else:
            # add structure to existing units as alternative conformation
            # (initially all identical)
            self.current += 1
            for i in range(0, len(self.unit), 1):
                self.unit[i].add_xyz(self.building_block.get_xyz())
                self.unit[i].set_current(self.current)

        # test type consistency: if arrays are provided, length should be equal
        # to the amount of different types in conn_type
        if 'ndarray' in str(type(alpha)):
            try:
                if len(alpha) != len(beta) or len(alpha) != len(gamma) or len(alpha) != len(np.unique(self.conn_type)):
                    print("ERROR: inconsistent length in provided angle arrays")
                    print("> received the following angles: %s, %s, %s" % (alpha, beta, gamma))
                    return -1
            except Exception as e:
                raise Exception("ERROR: all angle arrays should have length %s" %(len(np.unique(self.conn_type))))

        # generate desired polyhedral coordinates (note: internally we work in
        # radians, not degrees)
        poly_xyz = self._polycalc_core(self.W, self.L, self.H, S, self.nu, self.phi, np.radians(alpha), np.radians(beta), np.radians(gamma), deformation)

        for x in range(0, len(self.unit), 1):
            coords = poly_xyz[x, :, :].squeeze()

            # if add_conformation:
            #    self.unit[x].add_xyz(coords)
            # else:
            self.unit[x].set_xyz(coords)

    def set_current(self, index):
        '''
        Select current polyhedral conformation from ensemble of previously generated alternatives.
        This places the frame pointer at the same desired position in all polyhedron subunits.

        :param index: number of alternative conformation (starting from 0)
        '''
        for x in range(0, len(self.unit), 1):
            self.unit[x].set_current(index)

    def delete_xyz(self, index):
        '''
        Delete one conformation in the conformational database.
        the new current conformation will be the previous one

        :param index: alternative coordinates set to remove (starting from zero)
        '''

        for u in range(0, len(self.unit), 1):
            self.unit[u].delete_xyz(index)

        if index > 0:
            self.current = index - 1
        else:
            self.current = 0

    def rmsd_distance_matrix(self, points_indices=[]):
        '''
        Calculate the RMSD between all structures with respect of a reference structure.
        uses Kabsch alignement algorithm.

        :param points_indices: indices of points of interest. This must be a list of indices of atoms in unites, i.e. [[unit1_indices],[unit2_indices],...]
        :returns: RMSD of all structures with respect of reference structure (in a numpy array)
        '''

        # this method exploits the RMSD method implemented in Structure class.
        S = Structure()


        # all alternative coordinates are accumulated in a Structure instance
        for i in range(0, self.unit[0].coordinates.shape[0], 1):
            self.set_current(i)

            # if specific coordinates are requested, load only those
            if len(points_indices) > 0:
                pts = self.unit[0].get_xyz()
                for i in range(1, len(self.unit), 1):
                    pts = np.concatenate((pts, self.unit[i].get_xyz()[points_indices[i], :]))

            # otherwise, load everything
            else:
                pts = self.get_all_xyz()

            S.add_xyz(pts)

        return S.rmsd_distance_matrix()

    def write_poly_architecture(self, output="", scale=10, deformation=[], colors=[]):
        '''
        dump PDB file and tcl script loadable in VMD showing the loaded polyhedral scaffold.

        pseudoatoms are placed in vertices, cylinders connect them. Cylinder color code matches connection type.

        :param scale: vertices scaling factor (i.e. how much you want to blow up your architecture)
        :param colors: list of colors to be used when coloring the cylinders in VMD session. By default, the following 25 VMD colors are available (in this order): blue, red, gray, orange, yellow, tan ,silver, green, white, pink, cyan, purple, lime, mauve, ochre, iceblue, black, yellow2, green2, cyan2, blue2, violet, magenta, red2, orange2.
        :param deformation: if provided, deformations will be applied as described in deformation database
        :param output: name of output files (without extension. .pdb and .tcl will be automatically added). By default, the name will be the polyhedron name.
        '''

        if output == "":
            output = self.polyname

        if len(colors) == 0:
            colors = ['blue', 'red', 'gray', 'orange', 'yellow', 'tan', 'silver', 'green', 'white', 'pink', 'cyan', 'purple', 'lime',
                      'mauve', 'ochre', 'iceblue', 'black', 'yellow2', 'green2', 'cyan2', 'blue2', 'violet', 'magenta', 'red2', 'orange2']

        pos = self.v * scale

        if len(deformation) > 0:
            if len(self.deform) == 0:
                raise Exception("ERROR: %s deformation coefficients provided, but no deformation axis found!" % len(deformation))

            elif len(deformation) == len(self.deform[:, 1]):
                for d in self.deform:
                    pos[int(d[0])] += deformation[int(d[1])] * d[2:5]

            else:
                raise Exception("ERROR: %s deformation coefficients expected, but %s found" % (len(deformation), np.unique(self.deform[:, 1])))

        # output vertices coordinates
        S = Structure(pos)
        S.write_pdb("%s.pdb" % output)

        # output VMD script loading the structure and drawing colored cylinders
        # between them if a connection exists
        fout = open("%s.tcl" % output, "w")
        fout.write("mol new %s.pdb\n" % output)
        fout.write("mol modstyle 0 top VDW 1.000000 12.000000\n")

        for k in range(0, len(self.conn), 1):

            i = pos[self.conn[k][0]]
            j = pos[self.conn[k][1]]

            clr = colors[self.conn_type[k % len(colors)]]
            fout.write("draw color %s\n" % clr)
            fout.write("draw cylinder {%s %s %s} {%s %s %s} radius 0.5\n" %(i[0], i[1], i[2], j[0], j[1], j[2]))

        fout.close()

    def get_neighbors(self, return_chain_names=False):
        '''
        create dictionary defining neighborhood relationships, i.e. edges having a vertex in common (key=edge chain name, value: numpy array of neighboring chains)

        :param return_chain_names: if False, neighbors indices are returned, if True indices will be converted in chain names (as assigned when a pdb is written)
        :returns: dictionary containing neighborhood information
        '''

        # create dictionary with neighbors (key: chain name, values: numpy
        # array of neighboring chains)
        neigh_dict = {}
        for i in range(0, len(self.conn), 1):
            neighs = []
            for j in range(0, len(self.conn), 1):
                if i != j:
                    if self.conn[i, 0] in self.conn[j] or self.conn[i, 1] in self.conn[j]:
                        if return_chain_names:
                            neighs.append(self.chain_names[j])
                        else:
                            neighs.append(j)

            if return_chain_names:
                neigh_dict[self.chain_names[i]] = np.array(neighs)
            else:
                neigh_dict[i] = np.array(neighs)

            return neigh_dict

    def _search_database(self, polyname, dbfilename="polyhedron_database_complete.dat"):
        '''
        search new style database (with plain formatting and containing connectivity information)
        '''

        # check polyhedra database existence
        if os.path.exists(dbfilename) == 0:
            raise Exception('ERROR: file %s not found!' % dbfilename)

        # look for desired polyhedron in database
        fin = open(dbfilename, 'r')
        v = []
        conn = []
        conn_type = []
        for line in fin.readlines():

            linetosave = line.split()
            if linetosave[0] == polyname:
                vert = int(linetosave[1])
                edges = int(linetosave[2])

                # prepare vertices list
                for i in range(vert):
                    v.append((float(linetosave[3 + i * 3]), float(linetosave[3 + i * 3 + 1]), float(linetosave[3 + i * 3 + 2])))

                # prepare connectivity list (could add feature that
                # connectivity is computed, if does not exist)
                for i in range(edges):
                    conn.append((int(linetosave[3 + 3 * vert + i * 2]), int(linetosave[3 + 3 * vert + i * 2 + 1])))

                # prepare connection type information
                if len(linetosave) == 3 + 3 * vert + 2 * edges:
                    print("WARNING: database contains no connection type for %s. Supposing all edges have same connection type." % polyname)
                    conn_type = np.zeros(edges)

                elif len(linetosave) == 3 + 3 * vert + 3 * edges:
                    for i in range(edges):
                        conn_type.append(int(linetosave[3 + 3 * vert + 2 * edges + i]))

                else:
                    raise Exception("ERROR: database inconsistency for connectivity type information in polyhedron %s" % polyname)

                break

        fin.close()

        if len(v) == 0:
            raise Exception("ERROR: polyhedron %s not found in database %s!" %(polyname, dbfilename))

        return edges, np.array(v), np.array(conn), np.array(conn_type)

    def get_polyhedron_properties(self, S):
        '''
        retrieve polyhedron properties
        '''

        x = self.edges * 1.0  # x=number of edges
        y = 2 * self.edges / len(self.v) * 1.0  # y=average connectvity
        psi = 2 * np.pi * (1 / y - 1 / x)
        phi = 2 * np.pi / y  # average facial angle

        top = 1 - np.cos(psi)  # numerator
        bot = 1 - np.cos(phi)  # denominator
        nu = np.arccos(np.sqrt(top / bot))  # tangent curvature angle

        circumradius = S / (2 * np.sin(nu))  # circumradius
        midradius = circumradius * np.cos(nu)  # mid radius

        return psi, phi, nu, circumradius, midradius

    def _polycalc_core(self, W, L, H, S, nu, phi, alpha, beta, gamma, deformation=[], get_box_edges=False):
        '''
        launch polyhedron assembly
        '''

        kay = (2 * H * np.tan(nu) + W / (np.tan(phi / 2) * np.cos(nu))) / L + 1  # scaling factor for truncation
        scale = kay * S  # factor to adjust polyhedron
        vscale = self.v * scale  # rescale polyhedron

        # if needed, apply edges deformation
        if len(deformation) > 0:
            for d in self.deform:
                vscale[int(d[0])] += deformation[int(d[1])] * d[2:5]

        # build bounding box vertices
        vrecty = self._rectanglify(vscale, L, W, H, kay)

        # iterate over all edges
        v_cubody_data = []

        if get_box_edges:
            boxes_edges = []

        for i in range(len(self.conn)):

            # cube located where the protein will have to be displaced
            vcuby = []

            # dig out all 8 vertices for each new rectanglified edge (4 on each
            # end). Distinguish the 'outer' 4 vertices from the 'inner' 4
            # vertices
            for j in range(len(vrecty)):
                if(self.conn[i, 0] == vrecty[j][0] and self.conn[i, 1] == vrecty[j][1]):
                    for k in range(4):
                        vcuby.append((vrecty[j][2 + k][0],
                                      vrecty[j][2 + k][1],
                                      vrecty[j][2 + k][2]))
                elif(self.conn[i, 1] == vrecty[j][0] and self.conn[i, 0] == vrecty[j][1]):
                    for k in range(4):
                        vcuby.append((vrecty[j][2 + k][0],
                                      vrecty[j][2 + k][1],
                                      vrecty[j][2 + k][2]))

            if get_box_edges:
                boxes_edges.extend(vcuby)

            # find the angles required to rotate the central pdb file, to align
            # it with the box (previously was also returning vtbox)
            if 'ndarray' not in str(type(alpha)):
                vtdata = self._cuboid_adjust(np.array(vcuby), self.unit[i].get_xyz(), alpha, beta, gamma, (i, 0))
            else:
                conntype = self.conn_type[i]
                vtdata = self._cuboid_adjust(np.array(vcuby), self.unit[i].get_xyz(), alpha[conntype], beta[conntype], gamma[conntype], (i, 0))

            # append vertex array with pdb file
            v_cubody_data.append(vtdata)

        # protein enclosing boxes
        if get_box_edges:
            Str = Structure(np.array(boxes_edges))
            Str.write_pdb("edges.pdb")

        # return atoms coordinates
        return np.array(v_cubody_data).astype(float)

    def _rectanglify(self, vscale, L, W, H, kay):
        '''
        take a vertex list and return a rectanglified vertex list
        '''

        vrecty = []
        for i in range(len(vscale)):  # for each individual vertex position
            v1 = vscale[i]

            # scroll through the other vertices...
            for j in range(len(vscale)):

                #tt = self._conn_find(i,j)
                tst = False
                for k in range(len(self.conn)):
                    if(self.conn[k, 0] == i and self.conn[k, 1] == j) or (self.conn[k, 1] == i and self.conn[k, 0] == j):
                        tst = True
                        break

                if i != j and tst:  # if the two are connected then get to work!

                    v2 = vscale[j]  # get the vector to the second vertex
                    # get normalised edge vector
                    v2n = (v2 - v1) / np.linalg.norm(v2 - v1)
                    v3c = np.cross(v1, v2)  # get perpendicular to edge
                    v3 = v3c / np.linalg.norm(v3c)

                    # walk short distance from corner to rectangle face
                    i2 = v2n * (kay - 1) * L / 2.0
                    i3 = v3 * W / 2.0  # half width step perpendicular to edge

                    temp1 = v1 + i2  # vector to face of rectangle from corner down edge
                    temp2 = temp1 + i3  # vector to side of rectangle face
                    temp3 = temp1 - i3  # vector to other side of rectangle face

                    # vector perpendicular to edge and width
                    v4c = np.cross(v2n, v3)
                    v4 = v4c / np.linalg.norm(v4c)
                    temp4 = temp2 - v4 * H  # step off outside edge by the height #1
                    temp5 = temp3 - v4 * H  # step off outside edge by heigh #2

                    # take the rectangle face, indexed by vertex into recty array
                    vrecty.append([i, j, temp2, temp3, temp4, temp5])

        return vrecty

    def _cuboid_adjust(self, vcuby, v_data, alpha, beta, gamma, ii):
        '''
        take a cuboid and a source cuboid and data file, and output vertex array with data and box transposed into target position
        ii parameter: if second element not 0, indicates the number of elements per ring (second half rotated by 180 degrees)
        '''

        lim = 1E-7

        # move vertices to centre of mass
        com_cuby = np.mean(vcuby, axis=0)
        vt = vcuby - com_cuby

        if ii[1] != 0:
            # for the double rings, work out what the beta=0 position is based
            # on the centre of face vector
            e = np.mean(vt[[2, 3, 6, 7]], axis=0)

            # work out what centre of face vector makes with Z axis
            betaZero = np.arccos(np.dot(e, np.array([0.0, 0.0, 1.0])) / np.linalg.norm(e))
            # correct to make sure beta is offset by this much
            beta = (np.pi / 2.0 - betaZero) + beta

        tusk = 5.0  # These random rotations make sure that we don't get stuck in an initially bad position
        # perform a random rotation 1
        vt = self._poly_rotate(vt, tusk, 0.0, 0.0)
        # perform a random rotation 2
        vt = self._poly_rotate(vt, 0.0, tusk, 0.0)
        # perform a random rotation 3
        vt = self._poly_rotate(vt, 0.0, 0.0, tusk)

        # find angle of (x,y) component of vector from origin to centre of face
        # (0,1,2,3) makes with positive x axis
        e = np.mean(vt[0:4], axis=0)

        if np.abs(e[0]) != 0 and np.abs(
                e[1]) != 0:  # exception in the case where the 'face' vector is NOT along the +/- z axis
            # theta1=vec_ang((e[0],e[1],0.0),(1.0,0.0,0.0))
            e[2] = 0.0
            theta1 = np.arccos(np.dot(e, np.array([1.0, 0.0, 0.0])) / np.linalg.norm(e))

            if e[1] < 0:
                theta1 = theta1 * -1.0
        else:  # in the case where the face vector IS along the +/- z axis
            theta1 = 0.0

        # perform the z rotation
        vt2 = self._poly_rotate(vt, 0.0, 0.0, theta1)

        # find angle of (z, x) component of vector from origin to centre of
        # face (0, 1, 2, 3) makes with positive x axis
        e = np.mean(vt2[0:4], axis=0)

        if e[2] != 0 and e[
                0] != 0:  # exception for when the face vector is along the +/- y axis
            #theta2=vec_ang((e[0], 0.0, e[2]), (1.0, 0.0, 0.0))
            e[1] = 0.0
            theta2 = np.arccos(np.dot(e, np.array([1.0, 0.0, 0.0])) / np.linalg.norm(e))

            if e[2] > 0:
                theta2 = theta2 * -1.0
        # do this if the vector is along the +/- y axis (TETRAHEDRON WANTS
        # ZERO. SO DOES OCTAHEDRON)
        else:
            theta2 = 0.0

        # perform the 'alpha' lateral rotation
        vt2 = self._poly_rotate(vt2, 0.0, theta2, 0.0)

        # find angle of (y, z) compoent of vector from origin to centre of face
        # (0, 1, 2, 3) makes with positive x axis
        e = vt2[0] - vt2[1]

        # perform the z rotation
        #theta3 = vec_ang((0.0, e[1], e[2]), (0.0, 1.0, 0.0))
        e[0] = 0.0
        theta3 = np.arccos(np.dot(e, np.array([0.0, 1.0, 0.0])) / np.linalg.norm(e))

        if e[2] < 0:
            theta3 = theta3 * -1.0
        vt2 = self._poly_rotate(vt2, theta3, 0.0, 0.0)  # perform an x rotation

        # test for convergence problems
        e = vt2[0] - vt2[1]
        if np.abs(e[0]) > lim:
            return -1

        # PERFORM THE REVERSE ROTATIONS AND TRANSLATION
        vrd = v_data
        if ii[0] >= ii[1] and ii[1] != 0:  # DO THIS FOR DOUBLE RINGS
            vrd = self._poly_rotate(vrd, 0.0, np.pi, 0.0)  # 180 about y
            beta = beta * -1.0

        # perform the 'beta' edge rotation
        vrd = self._poly_rotate(vrd, beta, 0.0, 0.0)
        # perform the 'alpha' lateral rotation
        vrd = self._poly_rotate(vrd, 0.0, 0.0, alpha)
        # perform the 'gamma' lateral rotation
        vrd = self._poly_rotate(vrd, 0.0, gamma, 0.0)
        vrd = self._poly_rotate(vrd, -theta3, 0.0, 0.0)  # x rotation by theta3
        vrd = self._poly_rotate(vrd, 0.0, -theta2, 0.0)  # y rotation by theta2
        vrd = self._poly_rotate(vrd, 0.0, 0.0, -theta1)  # z rotation by theta1
        # perform the reverse random rotation 1
        vrd = self._poly_rotate(vrd, 0.0, 0.0, -tusk)
        # perform the reverse random rotation 2
        vrd = self._poly_rotate(vrd, 0.0, -tusk, 0.0)
        # perform the reverse random rotation 3
        vrd = self._poly_rotate(vrd, -tusk, 0.0, 0.0)

        # perform the reverse of the translation on the structure return
        # coordinates
        return vrd + com_cuby

    def _poly_rotate(self, v_in, a, b, c):
        '''
        function of rotating a point about three euler angles
        requires three rotation matricies (for x y and z axis)
        will return vector from com to new position
        1 0 0         #cosb 0 -sinb  #cosc sinc 0
        0 cosa sina   #0 1 0         #-sinc cosc 0
        0 -sina cosa  #sinb 0 cosb   #0 0 1
        convention is X,Y,Z and right handed
        expects angles to be provided in in radians
        '''

        v = np.array(v_in)

        if a != 0:
            m1 = np.array([[1.0, 0.0, 0.0],
                           [0, np.cos(a), np.sin(a)],
                           [0, -np.sin(a), np.cos(a)]])
            v = np.dot(v, m1.T)

        if b != 0:
            m2 = np.array([[np.cos(b), 0, -np.sin(b)],
                           [0.0, 1.0, 0.0],
                           [np.sin(b), 0, np.cos(b)]])
            v = np.dot(v, m2.T)

        if c != 0:
            m3 = np.array([[np.cos(c), np.sin(c), 0],
                           [-np.sin(c), np.cos(c), 0],
                           [0.0, 0.0, 1.0]])
            return np.dot(v, m3.T)
        else:
            return v
