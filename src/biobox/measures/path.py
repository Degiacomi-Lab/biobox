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

import heapq
import warnings
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
        '''
        Create an empty queue.
        '''
        self.elements = []

    def empty(self):
        '''
        test whether the priority queue is empty.

        :returns: True if the queue contains no element
        '''
        return len(self.elements) == 0

    def put(self, item, priority):
        '''
        add element in priority queue.

        :param item: item to add in queue
        :param priority: item's priority (the lower, the earlier the item is popped)
        '''
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        '''
        pop top priority element from queue.

        :returns: the item having the lowest priority value
        '''
        return heapq.heappop(self.elements)[1]


class Path(object):
    '''
    Methods for finding shortest paths between points in a point cloud (typically a :func:`Structure <biobox.classes.structure.Structure>` object).

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
        After instantiation, :func:`setup_local_search <biobox.measures.path.Path.setup_local_search>` or :func:`setup_global_search <biobox.measures.path.Path.setup_global_search>` must first be called (depending on whether one wants to generate a single grid encompassing all the protein, or a smaller, moving grid).

        :param points: points for representing obstacles, as an (n, 3) numpy array.
        '''
        self.graph = Graph(points)
        self.kind = "none"

    def setup_local_search(self, step=1.0, maxdist=28, params=np.array([])):
        '''
        setup Path to perform path search using the local grid method.
        This method (or :func:`setup_global_search <biobox.measures.path.Path.setup_global_search>`) must be called before and path detection can be launched with :func:`search_path <biobox.measures.path.Path.search_path>`.

        :param step: grid step size, in A
        :param maxdist: edge length of the cubic local grid, in A. Pairs of points further apart than this Euclidean distance are not searched (see :func:`search_path <biobox.measures.path.Path.search_path>`)
        :param params: exclusion radius in A per obstacle point (1D array), as built by :func:`set_clashing_atoms <biobox.measures.path.Xlink.set_clashing_atoms>` with atoms_vdw. A grid point closer to an obstacle point than its exclusion radius is inaccessible, whatever the grid step. If empty, the default density model is used (obstacle points convolved with a single Gaussian kernel)
        '''
        self.graph.make_grid(step=step, maxdist=maxdist, params=params)
        self.maxdist = maxdist
        self.kind = "local"

    def setup_global_search(self, step=1.0, maxdist=34, use_hull=True, boundaries=[], cloud=np.array([]), params=np.array([])):
        '''
        setup Path to perform path search using the a global grid wrapping all the obtacles region.
        This method (or :func:`setup_local_search <biobox.measures.path.Path.setup_local_search>`) must be called before and path detection can be lauched with :func:`search_path <biobox.measures.path.Path.search_path>`.

        :param step: grid step size, in A.
        :param maxdist: pairs of points further apart than this Euclidean distance (in A) are not searched (see :func:`search_path <biobox.measures.path.Path.search_path>`). It does not affect the grid size
        :param use_hull: if True, points not laying within the convex hull wrapping around obtacles will be excluded
        :param boundaries: build a grid within the desired box boundaries, given as [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If neither boundaries nor cloud is defined, the grid wraps all obstacle points
        :param cloud: build a grid using a points cloud as extrema for the construction of the box (extended by one step on every side). If defined, the boundaries parameter is ignored.
        :param params: exclusion radius in A per obstacle point (1D array), as built by :func:`set_clashing_atoms <biobox.measures.path.Xlink.set_clashing_atoms>` with atoms_vdw. A grid point closer to an obstacle point than its exclusion radius is inaccessible, whatever the grid step. If empty, the default density model is used (obstacle points convolved with a single Gaussian kernel)
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

        :param start: coordinates of starting point (numpy array of 3 floats)
        :param end: coordinates of target point (numpy array of 3 floats)
        :param method: "theta" or "lazytheta" (Lazy Theta*), "old_theta" (Theta*), "astar" (A*) or "euclidean" (straight line, ignoring obstacles)
        :param get_path: if True, the returned path is filled with intermediate points spaced by at most 1 A (not only waypoints), see ``_get_trails``. The returned length is measured on the waypoints either way
        :param update_grid: if True, grid will be recalculated (for local search only)
        :param test_los: if true, a line of sight postprocessing will be performed to make paths straighter
        :returns: path length in A. It is -1 if the points are further apart than maxdist, are disconnected or method is unknown (a UserWarning is issued for an unknown method), and -2 (likely buried target) if no accessible grid point is found next to start or end, or if the SQUARED distance (in A2) to the closest one, as returned by ``Graph.get_closest_nodes``, exceeds maxdist + step (a value in A, compared as is)
        :returns: path coordinates as an (n, 3) numpy array ordered from end to start, or an empty array on failure
        :raises RuntimeError: if neither setup_local_search nor setup_global_search was called (except for the "euclidean" method)
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

        if self.kind != "global" and self.kind != "local":
            raise RuntimeError(
                "setup_local_search or setup_global_search must first be called")

        # if points are too far, skip it
        if euclidean > self.maxdist:
            return -1, np.array([])

        # place the local grid between the two points
        if self.kind == "local" and update_grid:
            self.graph.place_local_grid(start, end)

        connect_thresh = self.maxdist + self.graph.step

        # get indices of closest graph neighbors in graph, corresponding to points to connect
        # in this case start and end will be picked within the same ensemble of coordinates
        # first, check if atoms are accessible, exit if likely to be buried. The squared distance
        # to the closest grid point is compared with connect_thresh, and a target without
        # accessible grid point (placeholder index -1) is buried
        dists_start, idx_3d_start = self.graph.get_closest_nodes(np.array([start]))
        if dists_start[0] > connect_thresh or idx_3d_start[0][0] < 0:
            return -2, np.array([])

        dists_end, idx_3d_end = self.graph.get_closest_nodes(np.array([end]))
        if dists_end[0] > connect_thresh or idx_3d_end[0][0] < 0:
            return -2, np.array([])

        idx_start = self.graph.get_flat_index(np.array(idx_3d_start.T))[0]
        idx_end = self.graph.get_flat_index(np.array(idx_3d_end.T))[0]

        # if start and end see each other, do not run shortest path algorithms. The straight
        # line between them is used if it is clear, otherwise the path goes through their graph nodes
        if self._line_of_sight(idx_3d_start[0], idx_3d_end[0]):
            if self._segment_clear(start, end, self._target_exemptions(start, end)):
                waypoints = np.array([start, end])
            else:
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
                warnings.warn("search method %s unknown." % method, stacklevel=2)
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
        '''
        build the list of waypoints from the output of a shortest path algorithm.

        :param came_from: dictionary associating each flat grid index to its predecessor, as returned by the path search methods
        :param idx_start: flat index of the grid point closest to the start
        :param best_end_idx: flat index of the grid point closest to the end
        :param endpoint: coordinates of the end point
        :param start: coordinates of the start point
        :param clean_lineofsight: if True, grid points located between two mutually visible grid points of the path are removed
        :returns: waypoints coordinates as an (n, 3) numpy array, from endpoint to start, or an empty array if start and end are disconnected
        '''

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

            except Exception:
                # the goal was never reached: start and end are disconnected
                return np.array([])

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

        # clear graph nodes located between two nodes "seeing eachother": from the last kept
        # node, skip nodes as long as the following one is visible, so that every segment between
        # kept nodes is tested
        nodes = np.array([self.graph.get_3d_index(p) for p in pts_idx])
        keep = [0]
        for k in range(1, len(nodes) - 1):
            if not self._line_of_sight(nodes[keep[-1]].copy(), nodes[k + 1].copy()):
                keep.append(k)
        keep.append(len(nodes) - 1)

        # pts_crd holds the endpoint, then one coordinate per node, then the start point
        return np.array([pts_crd[0]] + [pts_crd[k + 1] for k in keep] + [pts_crd[-1]])

    # test whether the segment between two positions crosses only accessible grid points, with
    # the line of sight test used by the path search. Grid points closer to a center c than a
    # radius r, for every (c, r) in exempt, count as accessible
    def _segment_clear(self, p, q, exempt=()):
        '''
        test whether the segment between two positions crosses only accessible grid points.

        :param p: coordinates of the first position
        :param q: coordinates of the second position
        :param exempt: list of (center, radius) tuples. Grid points within radius of a center count as accessible during the test
        :returns: True if both positions are within the grid, the grid point of p is accessible and the line of sight between the grid points of p and q is clear
        '''

        g = self.graph
        shape = np.array(g.access_grid_shape)
        ends = g.get_idx_from_points(np.array([p, q], dtype=float))
        if np.any(ends < 0) or np.any(ends >= shape):
            return False

        changed = self._open_exemptions(exempt)
        try:
            clear = bool(g.access_grid[tuple(ends[0])]) and bool(self._line_of_sight(ends[0].copy(), ends[1].copy()))
        finally:
            self._close_exemptions(changed)

        return clear

    # temporarily mark as accessible the grid points closer to a center c than a radius r, for
    # every (c, r) in exempt. Returns what _close_exemptions needs to restore them
    def _open_exemptions(self, exempt):
        '''
        temporarily mark as accessible the grid points within a radius of given centers (and the grid point of each center).

        :param exempt: list of (center, radius) tuples
        :returns: list of (grid indices, previous accessibility values) tuples, to be passed to :func:`_close_exemptions <biobox.measures.path.Path._close_exemptions>`
        '''

        g = self.graph
        shape = np.array(g.access_grid_shape)
        changed = []
        for c, r in exempt:
            c = np.asarray(c, dtype=float)
            lo = np.clip(g.get_idx_from_points(np.array([c - r]))[0], 0, shape - 1)
            hi = np.clip(g.get_idx_from_points(np.array([c + r]))[0], 0, shape - 1)
            box = np.indices(hi - lo + 1).reshape(3, -1).T + lo
            near = box[np.linalg.norm(g.get_points_from_idx(box.astype(float)) - c, axis=1) <= r]
            own = g.get_idx_from_points(np.array([c]))
            own = own[np.all((own >= 0) & (own < shape), axis=1)]
            near = np.vstack([near, own])
            changed.append((near, g.access_grid[tuple(near.T)].copy()))
            g.access_grid[tuple(near.T)] = True
        return changed

    def _close_exemptions(self, changed):
        '''
        restore the accessibility of grid points modified by :func:`_open_exemptions <biobox.measures.path.Path._open_exemptions>`.

        :param changed: list of (grid indices, previous accessibility values) tuples, as returned by :func:`_open_exemptions <biobox.measures.path.Path._open_exemptions>`
        '''

        for near, values in reversed(changed):
            self.graph.access_grid[tuple(near.T)] = values

    # regions around the targets of a path not tested by _segment_clear. Targets lie in
    # inaccessible space, so the region extends from each target to its closest accessible grid point
    def _target_exemptions(self, *targets):
        '''
        build the regions around path targets to be considered accessible.

        :param targets: coordinates of the targets
        :returns: list of (target coordinates, radius) tuples, the radius being the distance between the target and its closest accessible grid point (0 if none is found)
        '''

        exempt = []
        for t in targets:
            dist, idx = self.graph.get_closest_nodes(np.array([t], dtype=float))
            if idx[0][0] < 0:
                exempt.append((np.asarray(t, dtype=float), 0.0))
            else:
                exempt.append((np.asarray(t, dtype=float), np.sqrt(dist[0]) + 1e-9))
        return exempt

    # fill intermediate regions between waypoints with points
    # points are separated with steps of 1A (or less)
    def _get_trails(self, waypoints):
        '''
        fill the segments between consecutive waypoints with intermediate points, keeping the waypoints order.

        A segment of length d is split into ceil(d) intervals of equal length (at most 1 A). Every waypoint appears once, the first and last waypoints included. A waypoint coinciding with the previous one is not repeated.

        :param waypoints: waypoints coordinates, as an (n, 3) numpy array
        :returns: path coordinates, as an (m, 3) numpy array
        '''

        waypoints = np.asarray(waypoints, dtype=float)
        if len(waypoints) == 0:
            return waypoints

        wpts = [waypoints[0]]

        for i in range(1, len(waypoints), 1):

            # add points in interval between old and new point, ending on the new point
            vec = waypoints[i] - waypoints[i - 1]
            distance = np.sqrt(np.dot(vec, vec))

            if distance == 0:
                continue

            n = int(np.ceil(distance))
            for k in range(1, n, 1):
                wpts.append(waypoints[i - 1] + vec * (k / float(n)))

            wpts.append(waypoints[i])

        return np.array(wpts)

    # measure the length of a path, provided as waypoints
    def _measure_path(self, waypoints):
        '''
        measure the length of a path.

        :param waypoints: path coordinates, as an (n, 3) numpy array
        :returns: sum of the Euclidean distances between consecutive points
        '''

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
        '''
        test the line of sight between two grid points, drawing a line between them with Bresenham algorithm.

        :param a: 3D index of the first grid point
        :param b: 3D index of the second grid point
        :returns: True if all grid points along the line are accessible
        '''
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
        :returns: dictionary associating each visited flat grid index to its predecessor (the start is its own predecessor)
        :returns: dictionary associating each visited flat grid index to its path cost from start, in grid steps
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
        :returns: dictionary associating each visited flat grid index to its predecessor (the start is its own predecessor)
        :returns: dictionary associating each visited flat grid index to its path cost from start, in grid steps
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
        :returns: dictionary associating each visited flat grid index to its predecessor (the start is its own predecessor)
        :returns: dictionary associating each visited flat grid index to its path cost from start, in grid steps
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
        A point is moved only if the path stays in accessible space, except next to the chain ends (the targets, which lie in inaccessible space).

        :param chain: numpy array containing the list of points composing the path
        :param move_angle_thresh: if angle between three consecutive points is greater than this threshold (in degrees), smoothing is performed
        :returns: length of the smoothed chain, in A (0.0 if the chain contains less than two points)
        :returns: smoothed chain (Nx3 numpy array, the input is not modified)
        '''
        # if chain is too short, return
        if len(chain) <= 1:
            return 0.0, chain

        elif len(chain) == 2:
            return np.sqrt(np.dot(chain[1] - chain[0], chain[1] - chain[0])), chain

        chain = np.array(chain, dtype=float)
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

        # for every flagged angle, try to straighten. Grid points next to the chain ends (the
        # targets) count as accessible while testing moves
        changed = self._open_exemptions(self._target_exemptions(chain[0], chain[-1]))
        try:
            self._straighten(chain, angles_test)
        finally:
            self._close_exemptions(changed)

        dist = 0
        for i in range(0, len(chain) - 1, 1):
            dist += np.sqrt(np.dot(chain[i] - chain[i + 1], chain[i] - chain[i + 1]))

        return dist, chain

    # move the middle point of every flagged angle of a chain halfway between its neighbours,
    # if the path stays clear
    def _straighten(self, chain, angles_test):
        '''
        move the middle point of every flagged angle halfway between its neighbours, if both new segments are clear. The untested neighbouring angles of a moved point are flagged in turn. Both arrays are modified in place.

        :param chain: path coordinates, as an (n, 3) numpy array of floats
        :param angles_test: numpy array of n-2 flags, 1 for angles to straighten. Tested angles are set to -1
        '''

        while np.any(angles_test == 1):

            for i in range(0, len(angles_test), 1):
                # if an angle is not (almost) straight, try to straighten it
                if angles_test[i] == 1:

                    # angle i involves atoms i, i+1 (center to be displaced)
                    # and i+2
                    point = (chain[i + 2] + chain[i]) / 2.0
                    angles_test[i] = -1

                    # move the point only if the path stays clear
                    if not self._segment_clear(chain[i], point) or not self._segment_clear(point, chain[i + 2]):
                        continue

                    chain[i + 1] = point

                    # if angle has been moved, tag for angle check its
                    # neighbors
                    if i >= 1 and angles_test[i - 1] != -1:
                        angles_test[i - 1] = 1
                    if i < len(angles_test) - 1 and angles_test[i + 1] != -1:
                        angles_test[i + 1] = 1


    def write_grid(self, filename="grid.pdb"):
        '''
        Write the accessible grid points to a PDB file

        :param filename: output file name
        '''

        # it is not in graph, to keep cython as clean as possible
        w = np.array(np.where(self.graph.access_grid)).T.astype(float)
        pts = self.graph.get_points_from_idx(w)
        S = Structure(p=pts)
        S.write_pdb(filename)



class Xlink(Path):
    '''
    subclass of :func:`Path <biobox.measures.path.Path>`, measure cross-linking distance between atom pairs in a molecule.

    * after instantiation, call first :func:`set_clashing_atoms <biobox.measures.path.Xlink.set_clashing_atoms>` to define molecule's atoms of interest for clash detection.
    * Subsequently, call either :func:`setup_local_search <biobox.measures.path.Xlink.setup_local_search>` or :func:`setup_global_search <biobox.measures.path.Xlink.setup_global_search>` to prepare the points grid used for path detection.
    * Physical distances between a list of atom indices can be finally computed with :func:`distance_matrix <biobox.measures.path.Xlink.distance_matrix>` or, between two atoms only, with :func:`search_path <biobox.measures.path.Path.search_path>`.

    If molecule contains multiple conformations, conformation i can be chosen by calling Xlink.molecule.set_current(i) before performing the procedure described in superclass.
    '''

    def __init__(self, molecule):
        '''
        :param molecule: :func:`Molecule <biobox.classes.molecule.Molecule>` instance
        '''
        self.molecule = molecule


    def set_clashing_atoms(self, atoms=[], densify=True, atoms_vdw=False, probe=1.7, points=[]):
        '''
        define atoms to consider for clash detection.

        :param atoms: atomnames to consider for clash detection. If undefined, protein backbone and C beta will be considered.
        :param densify: if True, all atoms not solvent exposed will be considered for clash detection, in addition to the solvent exposed atoms named in atoms.
        :param atoms_vdw: if True, every obstacle point is given an exclusion radius equal to its van der Waals radius plus probe: grid points closer than that to the atom are inaccessible, whatever the grid step (stored in self.params). Van der Waals radii are taken from knowledge['atom_vdw'] by atomtype (the '.' entry for unknown atomtypes, atomtypes being guessed from atom names if any is missing). The molecule's 'radius' column is not used, since import_pqr and pdb2pqr fill it with force field radii. If False, the default density model is used (obstacle points convolved with a single Gaussian kernel)
        :param probe: probe radius in A (the linker thickness) added to the van der Waals radii (atoms_vdw only). The default is the van der Waals radius of carbon
        :param points: if not empty, these coordinates (an (n, 3) array) are used as obstacles instead of the molecule's atoms, and all other parameters are ignored
        :returns: if densify is True, boolean mask over the molecule's atoms flagging those used as obstacles. If densify is False, indices of the atoms used as obstacles. None if points is provided
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

            # exclusion radius of every obstacle point: van der Waals radius plus probe
            vdw = self.molecule.know('atom_vdw')
            atomtypes = self.molecule.data["atomtype"].values[idxs]
            self.params = np.array([vdw.get(str(a).strip().upper(), vdw['.']) for a in atomtypes], dtype=float) + probe

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

        This method (or :func:`setup_global_search <biobox.measures.path.Path.setup_global_search>`) must be called before and path detection can be launched with :func:`search_path <biobox.measures.path.Path.search_path>`.

        :param step: grid step size, in A
        :param maxdist: edge length of the cubic local grid, in A. Pairs of points further apart than this Euclidean distance are not searched

        Grid accessibility follows the model chosen in :func:`set_clashing_atoms <biobox.measures.path.Xlink.set_clashing_atoms>` (atoms_vdw).
        '''
        super(Xlink, self).setup_local_search(step=step, maxdist=maxdist, params=self.params)


    def setup_global_search(self, step=1.0, maxdist=28, use_hull=False, boundaries=[], cloud=np.array([])):
        '''
        setup Path to perform path search using the a global grid wrapping all the obtacles region.
        This method (or :func:`setup_local_search <biobox.measures.path.Path.setup_local_search>`) must be called before and path detection can be lauched with :func:`search_path <biobox.measures.path.Path.search_path>`.

        :param step: grid step size, in A.
        :param maxdist: pairs of points further apart than this Euclidean distance (in A) are not searched. It does not affect the grid size
        :param use_hull: if True, points not laying within the convex hull wrapping around obtacles will be excluded
        :param boundaries: build a grid within the desired box boundaries, given as [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If neither boundaries nor cloud is defined, the grid wraps all obstacle points
        :param cloud: build a grid using a points cloud as extrema for the construction of the box (extended by one step on every side). If defined, the boundaries parameter is ignored.

        Grid accessibility follows the model chosen in :func:`set_clashing_atoms <biobox.measures.path.Xlink.set_clashing_atoms>` (atoms_vdw).
        '''

        super(Xlink, self).setup_global_search(step=step, maxdist=maxdist, use_hull=use_hull, boundaries=boundaries, cloud=cloud, params=self.params)


    def write_protein_points(self, filename="protein_points.pdb"):
        '''
        Write the points selected for clash detection into a pdb file

        :param filename: output file name
        '''
        S = Structure(p=self.graph.prot_points)
        S.write_pdb(filename)


    def distance_matrix(self, indices, method="theta", get_path=False, smooth=True, verbose=False, test_los=True, flexible_sidechain=False, sphere_pts_surf=4.0, sphere_thresh=2.0, sphere_radii=[6.3, 5.9, 5.4, 4.8]):
        '''
        compute distance matrix between provided indices.

        :param indices: atoms indices (within the data structure, not the original pdb file). Get the indices via molecule.atomselect(...) command.
        :param method: path search method, see :func:`search_path <biobox.measures.path.Path.search_path>` ("theta", "lazytheta", "old_theta", "astar" or "euclidean").
        :param get_path: if true, a list containing all the paths is also returned, each filled with intermediate points spaced by at most 1 A. With smooth, the reported distance is the length of the smoothed filled path
        :param smooth: if True, path will be refined to make turns less angular.
        :param verbose: if True, the algorithm will dump text in console
        :param sphere_pts_surf: surface occupied per sphere point, in A2. The smaller, the higher the points density (flexible_sidechain only, see :func:`get_half_sphere <biobox.measures.path.Xlink.get_half_sphere>`)
        :param sphere_thresh: minimal distance in A between a sphere point and any atom of the molecule (flexible_sidechain only)
        :param sphere_radii: radii in A of the concentric spheres built around each CA (flexible_sidechain only)
        :param test_los: if true, a line of sight postprocessing will be performed to make paths straighter
        :param flexible_sidechain: if True, the selected atoms will be rotated around their associated CA, in order to scan for alternative sidechain arrangements. A sphere of clash-free alternative conformations is generated, and the shortest distance accounting for all these different possibilities is returned. Note that this method is computationally expensive.
        :returns: distance matrix in A (numpy 2d array, len(indices) x len(indices)). matrix will contain -1 if atoms are too far or cannot be linked, and -2 if one of the two atoms is buried (rigid side chains only). With flexible_sidechain, distances shorter than the grid step are set to the grid step.
        :returns: only if get_path is True, list of paths, each formatted as [[i, j], path], where i and j are positions in indices and path is an (n, 3) numpy array. Pairs that cannot be linked have no entry. With flexible_sidechain, two spheres closer than the grid step are joined by the straight segment between their two closest points.
        '''

        # if flexible sidechain is needed
        if flexible_sidechain:
            spheres = []
            for i in indices:
                s = self.get_half_sphere(i, pts_surf=sphere_pts_surf, thresh=sphere_thresh, radii=sphere_radii)

                if len(s) > 0:
                    spheres.append(s)

                    if verbose:
                        print("> made sphere with %s points" % len(s))
                        Sph = Structure(p=s)
                        Sph.write_pdb("sphere%s.pdb" % i)

            if len(spheres) < 2:
                raise ValueError("less than 2 atoms available for linkage, cannot compute distance matrix!")

        # find indices corresponding coordinates
        pts = []
        for i in indices:
            try:
                pts.append(self.molecule.points[i])
            except Exception as e:
                raise IndexError("could not find index %s in molecule!" % i) from e

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
                    pts_crd = []
                    update_grid = True

                    # get euclidean distance matrix
                    dist_sph = SD.cdist(spheres[i], spheres[j])

                    # halting condition identifying spheres contact (useless to
                    # continue with point by point comparison). The path is the
                    # straight segment between the two closest points, from end to start
                    if np.min(dist_sph) < self.graph.step:
                        bestdist = self.graph.step
                        a, b = np.unravel_index(np.argmin(dist_sph), dist_sph.shape)
                        bestpath = np.array([spheres[j][b], spheres[i][a]])
                        if get_path:
                            bestpath = self._get_trails(bestpath)

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

                # pairs that cannot be linked have no path entry
                if get_path and dist > 0:
                    #path_data = [[i, j]]
                    #path_data.extend(pts_crd)
                    path_data = [[i, j], np.array(pts_crd)]
                    paths.append(path_data)

        if get_path:
            return distance, paths
        else:
            return distance

    def get_half_sphere(self, i, pts_surf=4.0, thresh=2.0, radii=[6.3, 5.9, 5.4, 4.8]):
        '''
        positions a side chain atom can reach by rotating around the CA of its residue, used by
        :func:`distance_matrix <biobox.measures.path.Xlink.distance_matrix>` with flexible_sidechain.

        Points are placed on concentric spheres centred on the CA. A point closer than thresh to any atom of the molecule
        is discarded and retested on the next sphere of the list. Only points on the same side of the backbone plane
        (through N, C and O) as the CB are kept, and of these only the cluster connected to the atom's current position.

        :param i: index of a side chain atom
        :param pts_surf: surface occupied per sphere point on the first sphere, in A2. The smaller, the higher the points density
        :param thresh: minimal distance in A between a sphere point and any atom of the molecule
        :param radii: radii in A of the concentric spheres, the first setting the number of points. The default suits lysine. If empty, a single sphere is built at the distance between the atom and its CA
        :returns: numpy array of points, the atom's current position first
        '''

        D = self.molecule.data.values
        l = D[i]
        if l[2] == "CA":
            raise ValueError("For flexible mode, a side chain atom must be provided!")

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

        step = np.max([pts_dist, 3.0] + list(dists))
        db = DBSCAN(eps=step, min_samples=2).fit(res3)
        if db.labels_[0] != -1:
            R = res3[db.labels_ == db.labels_[0]]
        else:
            R = np.array([res3[0]])

        return R
