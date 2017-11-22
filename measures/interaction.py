# Copyright (c) 2014 Matteo Degiacomi
#
# SBT is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# SBT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with SBT ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author : Matteo Degiacomi, matteothomas.degiacomi@gmail.com

'''
Helper functions for the calculation of pairwise interactions
'''

import numpy as np
import scipy.spatial.distance as S


def distance_matrix(points1, points2):
    '''
    compute full distance matrix between two points clouds

    :param points1: nx3 numpy array containing points coordinates
    :param points2: mx3 numpy array containing points coordinates
    :returns: nxm distance matrix (numpy array)
    '''
    return S.cdist(points1, points2)

def get_neighbors(dist, cutoff):
    '''
    detect interfacing points (couples at less than a certain cutoff distance)

    :param dist: distance matrix
    :param cutoff: maximal distance at which lennard-jones interaction is considered
    :returns: list of interacting couples
    '''
    return np.array(np.where(dist < cutoff))

def lennard_jones(points1, points2, epsilon=1.0, sigma=2.7, cutoff=12.0, coeff1=9, coeff2=6):
    '''
    Compute m-n lennard-jones potential between two ensembles of points.

    :param points1: nx3 numpy array containing points coordinates
    :param points2: mx3 numpy array containing points coordinates
    :param epsilon: epsilon parameter
    :param sigma: sigma parameter
    :param cutoff: maximal distance at which lennard-jones interaction is considered
    :param coeff1: coefficient of repulsive term (m coefficient)
    :param coeff2: coefficient of attractive term (n coefficient)
    :returns: m-n lennard-jones potential (kJ/mol)
    '''

    # get contacting residues
    dist = distance_matrix(points1, points2)
    couples = get_neighbors(dist, cutoff)

    enrg = 0.0
    for i in range(0, len(couples[0]), 1):
        the_dist = dist[couples[0, i], couples[1, i]]
        enrg += 4 * epsilon * ((sigma / the_dist)**coeff1 - (sigma / the_dist)**coeff2)

    return enrg
