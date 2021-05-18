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

import heapq
import scipy.spatial.distance as SD
import numpy as np
from sklearn.cluster import DBSCAN

import biobox.lib.fastmath as FM # cythonized
from biobox.lib.graph import Graph # cythonized

from biobox.classes.structure import Structure
from biobox.classes.convex import Sphere


class PriorityQueue(object):
    '''
    Queue for shortest path algorithms in Graph class.
    '''

    def __init__(self):
        self.elements = []

    def empty(self):
        '''
        clear priority queue
        '''
        return len(self.elements) == 0

    def put(self, item, priority):
        '''
        add element in priority queue"

        :param item: item to add in queue
        :param priority: item's priority
        '''
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        '''
        pop top priority element from queue
        '''
        return heapq.heappop(self.elements)[1]


class Path(object):
    '''
    Methods for finding shortest paths between points in a point cloud (typically a :func:`Structure <structure.Structure>` object).

    Two algorithms are implemented: A* and Theta*.
    In order to deal with lists of links within the same dataset, two methodologies are offered, global or local.

    * global builds a grid around the whole ensemble of points once, and every subsequent distance measure is performed on the same grid.
    * local a local grid is displaced to surround the points to be linked.

    Global take longer to initialize, and is more memory intensive, but subsequent measures are quick.
    Local takes less memory, but individual measures take more time to be performed.
    '''

    def __init__(self, points):
        '''
        An accessibility graph must be initially created around the provided points cloud.
        This is a mesh where none of its nodes clashes with any of the provided point in the cloud.
        After instantiation, :func:`setup_local_search <path.Path.setup_local_search>` or :func:`setup_global_search <path.Path.setup_global_search>` must first be called (depending on whether one wants to generate a single grid encompassing all the protein, or a smaller, moving grid).

        :param points: points for representing obstacles.
        '''
        self.graph = Graph(points)
        self.kind = "none"

    def setup_local_search(self, step=1.0, maxdist=28, params=np.array([])):
        '''
        setup Path to perform path search using the local grid method.
        This method (or :func:`setup_global_search <path.Path.setup_global_search>`) must be called before and path detection can be launched with :func:`search_path <path.Path.search_path>`.

        :param step: grid step size
        :param maxdist: clash detection threshold
        '''
        self.graph.make_grid(step=step, maxdist=maxdist, params=params)
        self.maxdist = maxdist
        self.kind = "local"

    def setup_global_search(self, step=1.0, maxdist=34, use_hull=True, boundaries=[], cloud=np.array([]), params=np.array([])):
        '''
        setup Path to perform path search using the a global grid wrapping all the obtacles region.
        This method (or :func:`setup_local_search <path.Path.setup_local_search>`) must be called before and path detection can be lauched with :func:`search_path <path.Path.search_path>`.

        :param step: grid step size.
        :param maxdist: clash detection threshold
        :param use_hull: if True, points not laying within the convex hull wrapping around obtacles will be excluded
        :param boundaries: build a grid within the desired box boundaries (if defined, maxdist parameter is ignored)
        :param cloud: build a grid using a points cloud as extrema for the construction of the box. If defined, maxdist and boundaries parameters are ignored.
        '''

        # if len(boundaries) == 0:
        #    smax=np.max(self.graph.prot_points, axis=0)+self.graph.step
        #    smin=np.min(self.graph.prot_points, axis=0)-self.graph.step
        #    boundaries=np.array([smin, smax]).T
        self.graph.make_global_grid(step=step, use_hull=use_hull, boundaries=np.array(boundaries), cloud=np.array(cloud), params=params)
        self.maxdist = maxdist
        self.kind = "global"


    def search_path(self, start, end, method="theta", get_path=True, update_grid=True, test_los=True):
        '''
        Find the shortest accessible path between two points.

        :param start: coordinates of starting point
        :param end: coordinates of target point
        :param method: can be theta or astar
        :param get_path: return full path (not only waypoints)
        :param update_grid: if True, grid will be recalculated (for local search only)
        :param test_los: if true, a line of sight postprocessing will be performed to make paths straighter
        '''

        ###INITIALIZE PATH SEARCH###
        euclidean = np.sqrt(np.dot(start - end, start - end))

        # if euclidean path is requested, don't go any futher
        # not the fastest way (when dealing with spheres, would be better to calculate pairwise distances in distance_matrix)
        # this said, though, we keep distance method selection packed in the
        # same function
        if method == "euclidean":
            waypoints = np.array([start, end])
            if get_path:
                waypoints = self._get_trails(waypoints)

            return euclidean, waypoints

        # if points are too far, skip it
        if euclidean > self.maxdist:
            return -1, np.array([])

        # specify grid search type (local or global)
        if self.kind == "local" and update_grid:
            self.graph.place_local_grid(start, end)

        elif self.kind != "global" and self.kind != "local":
            raise Exception(
                "setup_local_search or setup_global_search must first be called")

        connect_thresh = self.maxdist + self.graph.step

        # get indices of closest graph neighbors in graph, corresponding to points to connect
        # in this case start and end will be picked within the same ensemble of coordinates
        # first, check if atoms are accessible, exit if likely to be buried
        dists, idx_3d_start = self.graph.get_closest_nodes(np.array([start]))
        if dists[0] > connect_thresh:
            return -2, np.array([])

        dists, idx_3d_end = self.graph.get_closest_nodes(np.array([end]))
        if dists[0] > connect_thresh:
            return -2, np.array([])

        idx_start = self.graph.get_flat_index(np.array(idx_3d_start.T))[0]
        idx_end = self.graph.get_flat_index(np.array(idx_3d_end.T))[0]

        # if start and end see each other, do not run shortest path algorithms
        if self._line_of_sight(idx_3d_start[0], idx_3d_end[0]):
            waypoints = np.array([start, self.graph.get_points_from_idx_flat(idx_start), self.graph.get_points_from_idx_flat(idx_end), end])

        else:
            ###COMPUTE CURVED DISTANCE###

            if method == "old_theta":  # original theta* algorithm
                came_from, cost_so_far = self.theta_star(idx_start, idx_end)
            elif method == "astar":  # A* algorithm
                came_from, cost_so_far = self.a_star(idx_start, idx_end)
            # lazy theta* algorithm, more lightweight than the original
            elif method == "lazytheta" or method == "theta":
                came_from, cost_so_far = self.lazy_theta_star(
                    idx_start, idx_end)
            else:
                print("ERROR: search method %s unknown." % method)
                return -1, np.array([])

            # get waypoints using path dictionary, end node index and target
            # endpoint (prepended to resulting path)
            waypoints = self._get_waypoints(came_from, idx_start, idx_end, end, start, clean_lineofsight=test_los)

        if len(waypoints) == 0:  # points are disconnected
            return -1, waypoints


        # measure path length
        dist = self._measure_path(waypoints)

        # on request, get full path
        if get_path:
            waypoints = self._get_trails(waypoints)

        return dist, waypoints

    # interpret shortest path algorithm output
    def _get_waypoints(self, came_from, idx_start, best_end_idx, endpoint, start, clean_lineofsight=True):

        # build path form search algorithm output and compute cost
        # flattened indices (start with graph endpoint)
        pts_idx = [best_end_idx]
        # points coordinates (start with best position within endpoints)
        pts_crd = [endpoint]
        pts_crd.append(self.graph.get_points_from_idx_flat(best_end_idx))  # points coordinates

        cnt = 0
        # safety mechanism, regions of start and end point are disconnected
        while cnt < np.sum(self.graph.access_grid):

            # extract previous point, and continue if not null
            try:
                pred = came_from[pts_idx[-1]]

            except Exception as ex:
                return -1, np.array([])

            if pred == idx_start:
                break

            # coordinates of previous point
            predpt = self.graph.get_points_from_idx_flat(pred)
            # print(predpt)

            # extend path, and compute distance (store waypoints)
            pts_crd.append(predpt)
            pts_idx.append(pred)

            cnt += 1

        if cnt == np.sum(self.graph.access_grid):
            return np.array([])

        pts_idx.append(idx_start)  # start points indices

        pts_crd.append(self.graph.get_points_from_idx_flat(idx_start))
        pts_crd.append(start)  # start points coordinates

        # if no waypoint cleaning is required, return obtained path
        if not clean_lineofsight:
            return np.array(pts_crd)

        # clear waypoints located between two points "seeing eachother"
        test = np.ones(len(pts_crd)).astype(bool)

        w = []
        for p in pts_idx:
            p3d = self.graph.get_3d_index(p)
            w.append(p3d)

        waypoints = np.array(w)

        # forward
        i = 1
        while i < len(waypoints):

            if test[i] == 1:  # if point is visible

                for j in range(i+1, len(waypoints)):
                    #for j in range(len(waypoints)-1, i + 1, -1):
                    # test from end to second neighbor
                    if test[j] == 1:
                        v = waypoints[i].copy()
                        w = waypoints[j].copy()
                        if self._line_of_sight(v, w):
                            #print(i, j)
                            test[i + 1:j+1] = False
                            #break
            i += 1

        return np.array(pts_crd)[test]

    # fill intermediate regions between waypoints with points
    # points are separated with steps of 1A (or less)
    def _get_trails(self, waypoints):

        wpts = [waypoints[0]]

        for i in range(1, len(waypoints), 1):

            # add points in interval between old and new point
            vec = waypoints[i - 1] - waypoints[i]
            distance = np.sqrt(np.dot(vec, vec))

            if distance < 1: #was 0.01A
                continue

            vec /= np.linalg.norm(vec)
            start_here = waypoints[i]
            for a in np.linspace(distance, 1, int(distance)):
                pt = start_here + a * vec
                wpts.append(pt)

            wpts.append(waypoints[i])

        return np.array(wpts)

    # measure the length of a path, provided as waypoints
    def _measure_path(self, waypoints):

        # distance initialized with distance between best starting point and
        # closes graph node
        dist = 0
        for i in range(1, len(waypoints), 1):
            dist += np.sqrt(np.dot(waypoints[i - 1] - waypoints[i], waypoints[i - 1] - waypoints[i]))

        return dist

    # line-of-sight test. Draw line between two points using Bresenham algorithm.
    # line-of-sight established if all voxels are true in
    # self.graph.access_grid
    def _line_of_sight(self, a, b):
        return FM.c_line_of_sight(self.graph.access_grid, a, b)

    # def _heuristic(self, a, b):
    # return self.graph.heuristic(a, b)
    #a1 = np.array(self.graph.get_3d_index(a.T)).astype(float)
    #b1 = np.array(self.graph.get_3d_index(b.T)).astype(float)
    # return np.dot(b1-a1, b1-a1) #manhattan!

    def a_star(self, start, goal):
        '''
        A* algorithm, find path connecting two points in the graph.

        :param start: starting point (flattened coordinate of a graph grid point).
        :param goal: end point (flattened coordinate of a graph grid point).
        '''

        self.frontier = PriorityQueue()
        self.frontier.put(start, 0)
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[start] = start
        self.cost_so_far[start] = 0

        while not self.frontier.empty():
            self.current = self.frontier.get()

            if self.current == goal:
                break

            for thenext in self.graph.neighbors(self.current, True):

                new_cost = self.cost_so_far[self.current] + self.graph.cost(self.current, thenext)

                if thenext not in self.cost_so_far or new_cost < self.cost_so_far[thenext]:
                    self.cost_so_far[thenext] = new_cost
                    priority = new_cost + self.graph.heuristic(goal, thenext)
                    self.frontier.put(thenext, priority)
                    self.came_from[thenext] = self.current

        return self.came_from, self.cost_so_far


    def theta_star(self, start, goal):
        '''
        Theta* algorithm, find path connecting two points in the accessibility graph.

        :param start: starting point (flattened coordinate of a graph grid point).
        :param goal: end point (flattened coordinate of a graph grid point).
        '''

        self.frontier = PriorityQueue()
        self.frontier.put(start, 0)
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[start] = start
        self.cost_so_far[start] = 0

        while not self.frontier.empty():
            self.current = self.frontier.get()

            if self.current == goal:
                break

            for thenext in self.graph.neighbors(self.current, True):

                new_cost = self.cost_so_far[self.current] + self.graph.cost(self.current, thenext)

                # test line-of-sight between predecessor of current, and thenext
                # node.
                if self.came_from[self.current] != start:
                    a1 = np.array(self.graph.get_3d_index(self.came_from[self.current]))
                    b1 = np.array(self.graph.get_3d_index(thenext))
                    visible = self._line_of_sight(a1, b1)

                else:
                    visible = False

                # if visible, current node is useless and can be ignored.
                if visible:

                    newcost = self.cost_so_far[self.came_from[self.current]] + self.graph.cost(self.came_from[self.current], thenext)

                    if thenext not in self.cost_so_far or newcost < self.cost_so_far[thenext]:
                        self.came_from[thenext] = self.came_from[self.current]
                        self.cost_so_far[thenext] = newcost
                        priority = self.cost_so_far[thenext] + self.graph.heuristic(goal, thenext)
                        self.frontier.put(thenext, priority)

                elif thenext not in self.cost_so_far or new_cost < self.cost_so_far[thenext]:
                    self.cost_so_far[thenext] = new_cost
                    priority = new_cost + self.graph.heuristic(goal, thenext)
                    self.frontier.put(thenext, priority)
                    self.came_from[thenext] = self.current

        return self.came_from, self.cost_so_far


    def lazy_theta_star(self, start, goal):
        '''
        Lazy Theta* algorithm (better than Theta* in terms of amount of line-of-sight tests), find path connecting two points in the graph.

        :param start: starting point (flattened coordinate of a graph grid point).
        :param goal: end point (flattened coordinate of a graph grid point).
        '''

        self.frontier = PriorityQueue()
        self.frontier.put(start, 0)
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[start] = start
        self.cost_so_far[start] = 0

        while not self.frontier.empty():
            self.current = self.frontier.get()

            # test line-of-sight between predecessor of current, and current
            # (test if line_of_sight guess was right)
            if self.current != start:
                a1 = np.array(self.graph.get_3d_index(self.came_from[self.current]))
                b1 = np.array(self.graph.get_3d_index(self.current))
                visible = self._line_of_sight(a1, b1)
            else:
                visible = True

            # if not visible, select closest visible neighbor
            if not visible:
                min_pos = -1
                min_cost = 10000000
                for thenext in self.graph.neighbors(self.current, True):
                    if thenext in self.cost_so_far and thenext != self.came_from[self.current]:
                        new_cost = self.cost_so_far[thenext] + self.graph.cost(self.current, thenext)
                        if new_cost < min_cost:
                            min_cost = new_cost
                            min_pos = thenext

                self.came_from[self.current] = min_pos
                self.cost_so_far[self.current] = min_cost

            if self.current == goal:
                break

            for thenext in self.graph.neighbors(self.current, True):

                parentpos = self.came_from[self.current]
                test_cost = self.cost_so_far[parentpos] + self.graph.cost(parentpos, thenext)

                if thenext not in self.cost_so_far or test_cost < self.cost_so_far[thenext]:
                    self.came_from[thenext] = self.came_from[self.current]
                    self.cost_so_far[thenext] = test_cost
                    priority = self.cost_so_far[thenext] + self.graph.heuristic(goal, thenext)
                    self.frontier.put(thenext, priority)

        return self.came_from, self.cost_so_far


    def smooth(self, chain, move_angle_thresh=0.0):
        '''
        Utility method aimed at smoothing a chain produced by A* or Theta*, to make it less angular.

        :param chain: numpy array containing the list of points composing the path
        :param move_angle_thresh: if angle between three consecutive points is greater than this threshold, smoothing is performed
        :returns: smoothed chain (3xN numpy array)
        '''
        # if chain is too short, return
        if len(chain) <= 1:
            return 0.0, chain

        elif len(chain) == 2:
            return np.sqrt(np.dot(chain[1] - chain[0], chain[1] - chain[0])), chain

        angles_test = np.zeros(len(chain) - 2)

        # first test: scan all angles, and pinpoint the ones to check
        for i in range(1, len(chain) - 1, 1):

            mod1 = np.linalg.norm(chain[i] - chain[i - 1])
            mod2 = np.linalg.norm(chain[i + 1] - chain[i])

            if mod1 == 0 or mod2 == 0:
                angles_test[i - 1] = 1
                continue

            a1 = (chain[i] - chain[i - 1]) / mod1
            a2 = (chain[i + 1] - chain[i]) / mod2

            if not np.any(a1 != a2):
                continue

            dd = np.dot(a1, a2)
            if dd > 1.0:
                dd = 1.0

            # if an angle is not close to straight, flag it for straightening
            if np.degrees(np.arccos(dd)) > move_angle_thresh:
                angles_test[i - 1] = 1

        # for every flagged angle, try to straighten
        while np.any(angles_test == 1):

            for i in range(0, len(angles_test), 1):
                # if an angle is not (almost) straight, try to straighten it
                if angles_test[i] == 1:

                    # angle i involves atoms i, i+1 (center to be displaced)
                    # and i+2
                    point = (chain[i + 2] + chain[i]) / 2.0
                    chain[i + 1] = point[:]

                    angles_test[i] = -1
                    # if angle has been moved, tag for angle check its
                    # neighbors
                    if i >= 1 and angles_test[i - 1] != -1:
                        angles_test[i - 1] = 1
                    if i < len(angles_test) - 1 and angles_test[i + 1] != -1:
                        angles_test[i + 1] = 1

        dist = 0
        for i in range(0, len(chain) - 1, 1):
            dist += np.sqrt(np.dot(chain[i] - chain[i + 1], chain[i] - chain[i + 1]))

        return dist, chain


    def write_grid(self, filename="grid.pdb"):
        '''
        Write the accessibility graph to a PBB file

        :param filename: output file name
        '''

        # it is not in graph, to keep cython as clean as possible
        w = np.array(np.where(self.graph.access_grid)).T.astype(float)
        pts = self.graph.get_points_from_idx(w)
        S = Structure(p=pts)
        S.write_pdb(filename)



class Xlink(Path):
    '''
    subclass of :func:`Path <path.Path>`, measure cross-linking distance between atom pairs in a molecule.

    * after instantiation, call first :func:`set_clashing_atoms <path.Xlink.set_clashing_atoms>` to define molecule's atoms of interest for clash detection.
    * Subsequently, call either :func:`setup_local_search <path.Xlink.setup_local_search>` or :func:`setup_local_search <path.Xlink.setup_global_search>` to prepare the points grid used for path detection.
    * Physical distances between a list of atom indices can be finally computed with :func:`distance_matrix <path.Xlink.distance_matrix>` or, between two atoms only, with :func:`search_path <path.Path.search_path>`.

    If molecule contains multiple conformations, conformation i can be chosen by calling Xlink.molecule.set_current(i) before performing the procedure described in superclass.
    '''

    def __init__(self, molecule):
        '''
        :param molecule: :func:`Molecule <molecule.Molecule>` instance
        '''
        self.molecule = molecule


    def set_clashing_atoms(self, atoms=[], densify=True, atoms_vdw=False, points=[]):
        '''
        define atoms to consider for clash detection.

        :param atoms: atomnames to consider for clash detection. If undefined, protein backbone and C beta will be considered.
        :param densify: if True, all atoms not solvent exposed will be considered for clash detection.
        '''

        if len(points) > 0:
            self.graph = Graph(points)
            self.params = np.array([])
            self.kind = "none"
            return

        if len(atoms) == 0:
            atoms = ["CA", "C", "N", "O", "CB"]

        # if true, consider all atoms in molecule's core as obstacles (make
        # protein core denser)
        if densify:

            # get indices of surface atoms
            from biobox.measures.calculators import sasa
            surf_id = sasa(self.molecule, n_sphere_point=300)[2]

            remove_id = []
            for i in surf_id:
                # if point is surface atom, and not backbone or CB, flag for
                # removal
                if self.molecule.data["name"][i] not in atoms:
                    remove_id.append(i)

            p = self.molecule.get_xyz()
            mask = np.ones(len(p))
            mask[remove_id] = 0
            idxs = mask.astype(bool)
            points = p[idxs]

        else:
            points, idxs = self.molecule.atomselect("*", "*", atoms, get_index=True)

        if atoms_vdw:
            # if no atomype is present, guess it
            if np.any(self.molecule.data["atomtype"] == ''):
                self.molecule.assign_atomtype()

            # knowledge base: define points standard deviations
            atomdata = [["C", 1.7, 1.455, 0.51],
                        ["H", 1.2, 0.72, 0.25],
                        ["O", 1.52, 1.15, 0.42],
                        ["S", 1.8, 1.62, 0.54],
                        ["N", 1.55, 1.2, 0.44]]
            self.params = np.zeros((len(points), 3)).astype(float)
            self.params[:, 1] = 1.0  # set defaults
            a_cnt = 1
            for a in atomdata:
                pos = self.molecule.data["atomtype"][idxs] == a[0]
                self.params[pos, 0] = a_cnt
                self.params[pos, 1] = a[2]  # *2.0
                self.params[pos, 2] = a[3]
                a_cnt += 1

        else:
            self.params = np.array([])

        # graph containing a grid of clash-free points
        # here, clashing points are provided, grid definition will come in a
        # second step, since two methods to do so are available
        self.graph = Graph(points)
        # grid building method, can be "local" or "global".
        self.kind = "none"

        return idxs


    def setup_local_search(self, step=1.0, maxdist=28):
        '''
        setup Path to perform path search using the local grid method.

        This method (or :func:`setup_global_search <path.Path.setup_global_search>`) must be called before and path detection can be launched with :func:`search_path <path.Path.search_path>`.

        :param step: grid step size
        :param maxdist: clash detection threshold
        '''
        super(Xlink, self).setup_local_search(step=step, maxdist=maxdist, params=self.params)


    def setup_global_search(self, step=1.0, maxdist=28, use_hull=False, boundaries=[], cloud=np.array([])):
        '''
        setup Path to perform path search using the a global grid wrapping all the obtacles region.
        This method (or :func:`setup_local_search <path.Path.setup_local_search>`) must be called before and path detection can be lauched with :func:`search_path <path.Path.search_path>`.

        :param step: grid step size.
        :param maxdist: clash detection threshold
        :param use_hull: if True, points not laying within the convex hull wrapping around obtacles will be excluded
        :param boundaries: build a grid within the desired box boundaries (if defined, maxdist parameter is ignored)
        :param cloud: build a grid using a points cloud as extrema for the construction of the box. If defined, maxdist and boundaries parameters are ignored.
        '''

        super(Xlink, self).setup_global_search(step=step, maxdist=maxdist, use_hull=use_hull, boundaries=boundaries, cloud=cloud, params=self.params)


    def write_protein_points(self, filename="protein_points.pdb"):
        '''
        Write the points selected for clash detection into a pdb file

        :param filename: output file name
        '''
        S = Structure(p=self.graph.prot_points)
        S.write_pdb(filename)


    def distance_matrix(self, indices, method="theta", get_path=False, smooth=True, verbose=False, test_los=True, flexible_sidechain=False, sphere_pts_surf=4.0):
        '''
        compute distance matrix between provided indices.

        :param indices: atoms indices (within the data structure, not the original pdb file). Get the indices via molecule.atomselect(...) command.
        :param method: can be "theta" or "astar".
        :param get_path: if true, a list containing all the paths is also returned
        :param smooth: if True, path will be refined to make turns less angular.
        :param verbose: if True, the algorithm will dump text in console
        :param sphere_pts_surf: surface occupied per sphere point, in A2. The smaller, the higher the points density
        :param test_los: if true, a line of sight postprocessing will be performed to make paths straighter
        :param flexible_sidechain: if True, the selected atoms will be rotated around their associated CA, in order to scan for alternative sidechain arrangements. A sphere of clash-free alternative conformations is generated, and the shortest distance accounting for all these different possibilities is returned. Note that this method is computationally expensive.
        :returns: distance matrix (numpy 2d array). matrix will contain -1 if atoms are too far, and -2 if one of the two atoms is buried. If get_path is True, a list of paths is also returned (format: [[id1, id2], [path]]).
        '''

        # if flexible sidechain is needed
        if flexible_sidechain:
            spheres = []
            for i in indices:
                try:
                #s = self._get_sphere(i)
                    s = self._get_half_sphere(i, pts_surf=sphere_pts_surf)

                except Exception as ex:
                    raise Exception(str(ex))

                if len(s) > 0:
                    spheres.append(s)

                    if verbose:
                        print("> made sphere with %s points" % len(s))
                        Sph = Structure(p=s)
                        Sph.write_pdb("sphere%s.pdb" % i)

            if len(spheres) < 2:
                raise Exception("less than 2 atoms available for linkage, cannot compute distance matrix!")

        # find indices corresponding coordinates
        pts = []
        for i in indices:
            try:
                pts.append(self.molecule.points[i])
            except Exception as ex:
                raise Exception("could not find index %s in molecule!" % i)

        # allocate distance matrix
        distance = np.zeros((len(indices), len(indices)))

        if get_path:
            paths = []

        # iterate over every pair
        for i in range(0, len(indices) - 1, 1):
            for j in range(i, len(indices), 1):

                if i == j:
                    continue

                # extract atom's residue information, in case
                # verbosity is requested             
                if verbose:
                    l1 = self.molecule.data.loc[indices[i], ["resname", "chain", "resid"]].values
                    l2 = self.molecule.data.loc[indices[j], ["resname", "chain", "resid"]].values

                # if sidechain flexibility is needed, launch ensemble of
                # measures on spheres
                if flexible_sidechain:

                    bestdist = 1000000
                    bestpath = []
                    update_grid = True

                    # get euclidean distance matrix
                    dist_sph = SD.cdist(spheres[i], spheres[j])

                    # halting condition identifying spheres contact (useless to
                    # continue with point by point comparison)
                    if np.min(dist_sph) < self.graph.step:
                        bestdist = self.graph.step

                    else:
                        # sort measures order from shortest to longest,
                        # according to euclidean distance
                        idxs = np.array(np.unravel_index(np.argsort(dist_sph, axis=None), dist_sph.shape)).T
                        # in case both spheres contain just one point
                        #if dist_sph.shape[0] == 1 and dist_sph.shape[1] == 1:
                        #    idxs = idxs[0]
                        for k in range(0, len(idxs), 1):

                            # stop if euclidean distance is greater than max
                            # distance threshold
                            if dist_sph[idxs[k, 0], idxs[k, 1]] > self.maxdist:
                                pts_crd = []
                                break
 
                            # stop if euclidean distance is greater than best
                            # curved path found up to now
                            if dist_sph[idxs[k, 0], idxs[k, 1]] > bestdist:
                                pts_crd = []
                                break

                            dist_tmp, pts_crd = self.search_path(spheres[i][idxs[k, 0]], spheres[j][idxs[k, 1]], method=method, get_path=get_path, test_los=test_los, update_grid=update_grid)
                            # update_grid = False #for testing, this is
                            # commented out

                            # dist = -1: sites are too far, dist == -2: one of
                            # the two targets is buried
                            if dist_tmp <= 0:
                                continue

                            if smooth:
                                dist_tmp, pts_crd = self.smooth(
                                    pts_crd, move_angle_thresh=0)

                            if dist_tmp < bestdist:
                                bestdist = dist_tmp
                                bestpath = pts_crd

                    # best resolution equal to grid size
                    if bestdist < self.graph.step:
                        bestdist = self.graph.step
                        if verbose:
                            print("> %s%s_%s and %s%s_%s can get in contact!" % (l1[0], l1[2], l1[1], l2[0], l2[2], l2[1]))

                    # if no connection found, indicate failure
                    if bestdist == 1000000:
                        dist = -1
                        if verbose:
                            print("> %s%s_%s and %s%s_%s cannot be linked!" % (l1[0], l1[2], l1[1], l2[0], l2[2], l2[1]))

                    else:
                        dist = bestdist
                        pts_crd = bestpath

                # if rigid sidechain, compute distance directly on atoms of
                # interest
                else:
                    # launch search on atoms of interest
                    dist, pts_crd = self.search_path(
                        pts[i], pts[j], method=method, get_path=get_path, test_los=test_los)

                    # if verbosity requested, report on failed measures
                    if verbose:
                        if dist == -1:
                            print("> %s%s_%s and %s%s_%s are too far!" % (l1[0], l1[2], l1[1], l2[0], l2[2], l2[1]))

                        if dist == -2:
                            print("> %s%s_%s vs %s%s_%s : likely buried targets" % (l1[0], l1[2], l1[1], l2[0], l2[2], l2[1]))

                    # dist = -1: sites are too far, dist == -2: one of the two
                    # targets is buried
                    if dist <= 0:
                        distance[i, j] = dist
                        distance[j, i] = dist
                        continue

                    if smooth:
                        dist, pts_crd = self.smooth(pts_crd, move_angle_thresh=0)

                distance[i, j] = dist
                distance[j, i] = dist

                # if verbosity requested, report obtaiend distance
                if verbose and dist > 0:
                    print("> %s%s_%s vs %s%s_%s: %5.2fA" % (l1[0], l1[2], l1[1], l2[0], l2[2], l2[1], dist))

                if get_path:
                    path_data = [[i, j]]
                    path_data.extend(pts_crd)
                    paths.append(path_data)

        if get_path:
            return distance, paths
        else:
            return distance

    # build sphere around a sidechain atom
    def _get_sphere(self, i, thresh=2.0):

        D = self.molecule.data
        l = D[i]
        if l[2] == "CA":
            raise Exception(
                "For flexible mode, a side chain atom must be provided!")

        # extract position of alpha carbon associated to provided side chain
        # atom
        test1 = np.logical_and(D[:, 4] == l[4], D[:, 5] == l[5])
        test2 = D[:, 2] == "CA"
        test = np.logical_and(test1, test2)

        pos = np.where(test)[0]
        if len(pos) == 1:
            alpha = self.molecule.points[test][0]
        else:
            return []

        # build sphere
        side = self.molecule.points[i]
        radius = np.sqrt(np.dot(side - alpha, side - alpha))
        # allow a surface of 4 A^2 to every point
        n_sphere_point = int(4.0 * np.pi * (radius**2) / 4.0)
        Sph = Sphere(radius, n_sphere_point=n_sphere_point, radius=0.0)
        Sph.translate(alpha[0], alpha[1], alpha[2])

        # return only clash free points in sphere
        dist = SD.cdist(Sph.points, self.molecule.points)

        res = [side]
        for k in range(0, dist.shape[0], 1):
            # keep sphere points at more than 1A from all neighbors
            if not np.any(dist[k] < thresh):
                res.append(Sph.points[k])

        return np.array(res)

    # build sphere around a sidechain atom. List of radii is valid for lysine
    def _get_half_sphere(self, i, pts_surf=4.0, thresh=2.0, radii=[6.3, 5.9, 5.4, 4.8]):

        D = self.molecule.data.values
        l = D[i]
        if l[2] == "CA":
            raise Exception("For flexible mode, a side chain atom must be provided!")

        pts, idxs = self.molecule.same_residue_unique(i, get_index=True)

        resdata = D[idxs]

        posCA = pts[resdata[:, 2] == "CA"][0]

        # if a list of radii for concentric spheres is not provided, guess a
        # single radius on the basis of distance of linkage atom from CA
        side = self.molecule.points[i]
        if len(radii) == 0:
            radii = [np.sqrt(np.dot(side - posCA, side - posCA))]

        # build concentric spheres
        # allow a surface of pts_den A^2 to every point
        n_sphere_point = int(4.0*np.pi*(radii[0]**2)/float(pts_surf))
        to_test = np.arange(n_sphere_point)
        res = []

        pts_dist = -1
        pts_dist_test = False
        for radius in radii:
            Sph = Sphere(radius, n_sphere_point=n_sphere_point, radius=0.0)
            Sph.translate(posCA[0], posCA[1], posCA[2])

            if not pts_dist_test:
                dsts = SD.cdist(Sph.points, Sph.points)
                pts_dist = np.min(dsts[dsts != 0])
                pts_dist_test = True

            # accept only clash free points in sphere.
            # unacceptable positions will be retested in the next smaller sphere
            dist = SD.cdist(Sph.points,self.molecule.points)
            new_to_test = []
            for k in range(0,dist.shape[0],1):
                #keep sphere points at more than threshold from all neighbors
                if not np.any(dist[k]<thresh) and k in to_test:
                    res.append(Sph.points[k])

                elif k in to_test:
                    new_to_test.append(k)

            to_test = new_to_test

        # accept only points in same half sphere of side chain
        posCB = pts[resdata[:,2]=="CB"][0]
        posO = pts[resdata[:,2]=="O"][0]
        posC = pts[resdata[:,2]=="C"][0]
        posN = pts[resdata[:,2]=="N"][0]

        # compute residue plane
        plane_vec1 = posC-posN
        plane_vec2 = posC-posO
        xprod = np.cross(plane_vec1, plane_vec2)
        xprod /= np.linalg.norm(xprod)

        # compute angle of CB with respect of plane normal
        side_vec = posCB-posCA
        side_vec /= np.linalg.norm(side_vec)
        angle1 = np.rad2deg(np.arccos(np.dot(xprod, side_vec)))

        res2 = [side]
        for p in res:
            side_vec2 = p-posCA
            side_vec2 /= np.linalg.norm(side_vec2)
            angle2 = np.rad2deg(np.arccos(np.dot(xprod, side_vec2)))

            if (angle1>90 and angle2>90) or (angle1<90 and angle2<90):
                res2.append(p)

        res3 = np.array(res2)

        #select only points reachable from the actual available coordinate
        rds = np.sort(np.array(radii)) #sorted radii list
        dists = np.array([rds[i+1]-rds[i] for i in range(len(rds)-1)]) #distances between adjacent spherical shells
      
        step = np.max([pts_dist, np.max(dists), 3.0])
        db = DBSCAN(eps=step, min_samples=2).fit(res3)
        if db.labels_[0] != -1:
            R = res3[db.labels_ == db.labels_[0]]
        else:
            R = np.array([res3[0]])
        
        return R
