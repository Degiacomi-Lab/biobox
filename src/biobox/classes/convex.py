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
# Author : Matteo Degiacomi, matteothomas.degiacomi@gmail.com

'''
Convex shapes made of spherical points.

Every shape is defined by a nominal convex body K, whose dimensions are the ones passed to the constructor (points included).
The points centres lie on the boundary of C, the body K shrunk by the points radius pr (the points at distance >= pr from the boundary of K), so that every point sphere touches the surface of K from inside.
The point spheres trace the body T = C + B(pr) (K with its edges rounded by pr, T = K for a smooth K).

Measures follow from the Steiner formula for the convex body C, having surface S, volume V and integral of mean curvature M (the integral of its support function over the unit sphere):

- surface of T: S + 2 M pr + 4 pi pr^2
- volume of T: V + S pr + M pr^2 + 4/3 pi pr^3
- CCS (projection approximation, i.e. mean projected area, equal to a quarter of the surface for a convex body): (S + 2 M rho + 4 pi rho^2) / 4, with rho = pr + gas
'''

from biobox.classes.structure import Structure
import numpy as np
from scipy.special import ellipe, ellipeinc, ellipkinc
from scipy.spatial import ConvexHull, HalfspaceIntersection


def _steiner(S, M, V, rho):
    '''
    surface and volume of the parallel body at distance rho of a convex body (Steiner formula).

    :param S: surface of the convex body
    :param M: integral of mean curvature of the convex body
    :param V: volume of the convex body
    :param rho: distance
    :returns: surface and volume of the parallel body
    '''
    return S + 2 * M * rho + 4 * np.pi * rho**2, V + S * rho + M * rho**2 + 4 * np.pi * rho**3 / 3.0


def _steps(length, step):
    '''
    equally spaced values from 0 to length (both included), spaced by at most step.

    :param length: interval length
    :param step: maximal spacing
    :returns: numpy array of values
    '''
    n = max(1, int(np.ceil(length / step - 1e-9)))
    return np.linspace(0, length, n + 1)


def _ellipse_perimeter(a, b):
    '''
    perimeter of an ellipse (complete elliptic integral of the second kind).

    :param a: first semi-axis
    :param b: second semi-axis
    :returns: perimeter
    '''
    a, b = max(a, b), min(a, b)
    return 4 * a * ellipe(1 - (b / a)**2)


def _ellipsoid_surface(a, b, c):
    '''
    surface of an ellipsoid (Legendre form, incomplete elliptic integrals).

    :param a: first semi-axis
    :param b: second semi-axis
    :param c: third semi-axis
    :returns: surface
    '''
    a, b, c = sorted([a, b, c], reverse=True)
    if a - c <= 1e-12 * a:
        return 4 * np.pi * a**2
    phi = np.arccos(c / a)
    m = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
    s = np.sin(phi)
    return 2 * np.pi * c**2 + 2 * np.pi * a * b / s * (ellipeinc(phi, m) * s**2 + ellipkinc(phi, m) * np.cos(phi)**2)


def _ellipsoid_mean_curvature(a, b, c, n=128):
    '''
    integral of mean curvature of an ellipsoid, as the integral of its support function over the unit sphere (Gauss-Legendre quadrature in the polar angle cosine, trapezoidal in the azimuth).

    :param a: semi-axis along x
    :param b: semi-axis along y
    :param c: semi-axis along z
    :param n: number of Gauss-Legendre nodes
    :returns: integral of mean curvature
    '''
    t, w = np.polynomial.legendre.leggauss(n)
    phi = np.arange(2 * n) * np.pi / n
    st = np.sqrt(1 - t**2)[:, None]
    h = np.sqrt((a * st * np.cos(phi))**2 + (b * st * np.sin(phi))**2 + (c * t[:, None])**2)
    return np.sum(w[:, None] * h) * np.pi / n


def _check_ellipsoid(a, b, c, pr):
    '''
    raise an exception if the points radius does not allow points to touch the surface of the ellipsoid everywhere.

    :param a: first semi-axis
    :param b: second semi-axis
    :param c: third semi-axis
    :param pr: points radius
    '''
    if min(a, b, c) <= 0:
        raise Exception("ERROR: ellipsoid semi-axes must be positive, found %s, %s, %s" % (a, b, c))
    rmin = min(a, b, c)**2 / max(a, b, c)
    if pr >= rmin:
        raise Exception("ERROR: points radius %s A must be smaller than the minimal radius of curvature of the ellipsoid with semi-axes %s, %s, %s A (%s A)" % (pr, a, b, c, rmin))


def _ellipsoid_inner_points(unit, a, b, c, pr):
    '''
    move points of an ellipsoid surface inward by pr along the surface normal.

    :param unit: points on the unit sphere (n x 3 numpy array), mapped on the ellipsoid surface by scaling by the semi-axes
    :param a: semi-axis along x
    :param b: semi-axis along y
    :param c: semi-axis along z
    :param pr: inward displacement
    :returns: n x 3 numpy array of points
    '''
    axes = np.array([a, b, c], dtype=float)
    x = unit * axes
    nrm = x / axes**2
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    return x - pr * nrm


def _polytope_measures(vertices):
    '''
    surface, integral of mean curvature and volume of the convex hull of a set of points. The integral of mean curvature is half the sum over edges of edge length times exterior dihedral angle.

    :param vertices: n x 3 numpy array
    :returns: surface, integral of mean curvature, volume
    '''
    hull = ConvexHull(vertices)
    nrm = hull.equations[:, :3]
    i = np.repeat(np.arange(len(hull.simplices)), 3)
    k = np.tile(np.arange(3), len(hull.simplices))
    j = hull.neighbors[i, k]
    keep = j > i
    i, k, j = i[keep], k[keep], j[keep]
    # the edge shared with neighbour k is made of the two vertices other than vertex k
    others = np.array([[1, 2], [0, 2], [0, 1]])[k]
    p1 = hull.points[hull.simplices[i, others[:, 0]]]
    p2 = hull.points[hull.simplices[i, others[:, 1]]]
    angle = np.arccos(np.clip(np.sum(nrm[i] * nrm[j], axis=1), -1.0, 1.0))
    M = 0.5 * np.sum(np.linalg.norm(p1 - p2, axis=1) * angle)
    return hull.area, M, hull.volume


class Prism(Structure):
    '''
    Create an ensemble of points arranged as a prism (polygonal bottom and top, flat sides).
    '''

    def __init__(self, r, h, n, skew=0.0, radius=1.1,
                 pts_density_u=np.pi / 32, pts_density_h=0.2):
        '''
        The nominal prism K has a regular polygonal bottom face of circumradius r centred at the origin of the xy plane (a vertex along x), and a top face at height h, shifted by skew along y.
        Points centres are placed on the boundary of K shrunk by the points radius (every face moved inward by radius), so that every point touches the surface of K. The shape is then centred at the origin.
        Properties r, h, n, skew and pt_radius hold K's dimensions and the points radius.

        :param r: distance of the vertices from the axis of symmetry (circumradius), points included, in Angstrom
        :param h: height, points included, in Angstrom
        :param n: number of side faces
        :param skew: shift of the top face along y with respect to the bottom one, in Angstrom
        :param radius: radius of the individual points composing it, in Angstrom. Must be smaller than half the height, and small enough for every side of the shrunk polygonal faces to keep a positive length
        :param pts_density_u: angular step between points around the prism axis (radians), to which the directions of the vertices are added, also used as radial step on bottom and top faces (Angstrom)
        :param pts_density_h: maximal step between points along the vertical axis, in Angstrom
        '''

        w, nrm, off, dC = self._section(r, h, n, skew, radius)
        H = h - 2 * radius

        # sides: rays from the centre of the bottom face (also through its vertices) reach its boundary, then follow the prism axis
        c = np.mean(w, axis=0)
        ulist = np.arange(0, 2 * np.pi, pts_density_u)
        vertices = np.mod(np.arctan2(w[:, 1] - c[1], w[:, 0] - c[0]), 2 * np.pi)
        missing = np.min(np.abs(np.mod(vertices[:, None] - ulist[None] + np.pi, 2 * np.pi) - np.pi), axis=1) > 1e-9
        ulist = np.sort(np.concatenate([ulist, vertices[missing]]))
        dirs = np.stack([np.cos(ulist), np.sin(ulist)], axis=1)
        proj = np.dot(dirs, nrm.T)
        dist = off - np.dot(nrm, c)
        with np.errstate(divide="ignore"):
            t = np.where(proj > 1e-12, dist[None] / proj, np.inf).min(axis=1)
        bnd = c + t[:, None] * dirs
        bnd = np.column_stack([bnd, np.ones(len(bnd)) * radius])
        center = np.array([c[0], c[1], radius])

        hlist = _steps(H, pts_density_h)
        side = bnd[:, None, :] + (hlist / H)[None, :, None] * dC

        # bottom and top: polygons scaled about the face centre
        rc = np.max(np.linalg.norm(w - c, axis=1))
        slist = np.arange(0, rc, pts_density_u) / rc
        bottom = center + slist[None, :, None] * (bnd - center)[:, None, :]
        top = bottom + dC

        p = np.concatenate([side.reshape(-1, 3), bottom.reshape(-1, 3), top.reshape(-1, 3)])

        super(Prism, self).__init__(p=p, r=radius)

        self.properties['r'] = r
        self.properties['h'] = h
        self.properties['n'] = n
        self.properties['skew'] = skew
        self.properties['pt_radius'] = radius

        self.center_to_origin()

    @staticmethod
    def _section(r, h, n, skew, pr):
        '''
        bottom face of the prism of points centres (nominal prism shrunk by pr).

        :param r: circumradius of the nominal prism
        :param h: height of the nominal prism
        :param n: number of side faces
        :param skew: shift of the nominal top face along y
        :param pr: points radius
        :returns: vertices of the bottom face (n x 2 numpy array, at height pr), outward normals (n x 2) and offsets (n) of its sides, as lines normal . x = offset, and the vector from the bottom face to the top one
        '''
        if h <= 2 * pr:
            raise Exception("ERROR: points radius %s A must be smaller than half the prism height (%s A)" % (pr, h / 2.0))

        ang = 2 * np.pi * np.arange(n) / n
        v = r * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        e = np.roll(v, -1, axis=0) - v
        d = np.array([0.0, skew, h])

        # side faces contain a bottom edge and the prism axis; at height pr they are moved inward by pr
        nk = np.cross(np.column_stack([e, np.zeros(n)]), d)
        nk /= np.linalg.norm(nk, axis=1)[:, None]
        off = np.sum(nk[:, :2] * v, axis=1) - pr - nk[:, 2] * pr

        # vertex k is the intersection of sides k-1 and k
        A = np.stack([np.roll(nk[:, :2], 1, axis=0), nk[:, :2]], axis=1)
        b = np.stack([np.roll(off, 1), off], axis=1)
        w = np.linalg.solve(A, b[..., None])[..., 0]

        lengths = np.sum((np.roll(w, -1, axis=0) - w) * e, axis=1) / np.linalg.norm(e, axis=1)
        if np.any(lengths <= 1e-9 * r):
            raise Exception("ERROR: points radius %s A is too large for a prism of circumradius %s A and %s sides" % (pr, r, n))

        return w, nk[:, :2], off, d * (h - 2 * pr) / h

    def _inner_measures(self):
        '''
        surface, integral of mean curvature and volume of the prism of points centres (an oblique prism with polygonal bottom face of area A and perimeter L, and axis vector d): V = A d_z, S = 2A + sum over sides of |edge x d|, M = pi |d| + pi L / 2.

        :returns: surface, integral of mean curvature, volume
        '''
        w, _, _, dC = self._section(self.properties['r'], self.properties['h'], self.properties['n'],
                                    self.properties['skew'], self.properties['pt_radius'])
        E = np.roll(w, -1, axis=0) - w
        area = 0.5 * np.abs(np.sum(w[:, 0] * np.roll(w[:, 1], -1) - np.roll(w[:, 0], -1) * w[:, 1]))
        perimeter = np.sum(np.linalg.norm(E, axis=1))
        lateral = np.sum(np.linalg.norm(np.cross(np.column_stack([E, np.zeros(len(E))]), dC), axis=1))
        return 2 * area + lateral, np.pi * np.linalg.norm(dC) + np.pi * perimeter / 2.0, area * dC[2]

    def get_surface(self):
        '''
        compute the surface of the body traced by the points (the nominal prism with edges and vertices rounded by the points radius).

        :returns: surface in A^2
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'])[0]

    def get_volume(self):
        '''
        compute the volume of the body traced by the points (the nominal prism with edges and vertices rounded by the points radius).

        :returns: volume in A^3
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'])[1]

    def ccs(self, gas=1):
        '''
        compute prism CCS in the projection approximation, as a quarter of the surface of the prism of points centres enlarged by points radius plus gas radius (see module documentation).

        :param gas: probe gas radius in Angstrom
        :returns: CCS in A^2
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'] + gas)[0] / 4.0


class Cylinder(Structure):
    '''
    Create an ensemble of points arranged as an elliptical cylinder.
    '''

    def __init__(self, r, h, squeeze=1.0, skew=0.0, radius=1.1, pts_density_u=np.pi / 32, pts_density_h=0.2):
        '''
        The nominal cylinder K has an elliptical bottom face of semi-axes r (along x) and r*squeeze (along y) centred at the origin of the xy plane, and a top face at height h, shifted by skew along y.
        Points centres are placed on the boundary of K shrunk by the points radius (faces and side moved inward by radius), so that every point touches the surface of K. The shape is then centred at the origin.
        Properties r1, r2, h, skew and pt_radius hold K's semi-axes, height, skew and the points radius.

        :param r: radius, points included, in Angstrom
        :param h: height, points included, in Angstrom
        :param squeeze: create an elliptical base, having the y semi-axis equal to squeeze times the x one
        :param skew: shift of the top face along y with respect to the bottom one, in Angstrom
        :param radius: radius of the individual points composing it, in Angstrom. Must be smaller than half the height, and than the minimal radius of curvature of the cylinder cross-section perpendicular to its axis
        :param pts_density_u: angular step between points along the u angle, in radians (using parametric function for cylinder)
        :param pts_density_h: maximal step between points along the height, in Angstrom
        '''

        self._check(r, r * squeeze, h, skew, radius)
        H = h - 2 * radius
        dC = np.array([0.0, skew, h]) * H / h

        # side: the rim of the bottom face follows the cylinder axis
        ulist = np.arange(0, 2 * np.pi, pts_density_u)
        rim = self._rim(ulist, r, r * squeeze, h, skew, radius)
        hlist = _steps(H, pts_density_h)
        side = rim[:, None, :] + (hlist / H)[None, :, None] * dC

        # bottom and top: rims scaled about the face centre
        ulist = np.arange(-np.pi / 2, np.pi / 2, pts_density_u)
        vlist = np.arange(-np.pi, np.pi, pts_density_u)
        rim = self._rim(vlist, r, r * squeeze, h, skew, radius)
        center = np.array([0.0, skew * radius / h, radius])
        bottom = center + np.cos(ulist)[:, None, None] * (rim - center)[None, :, :]
        top = bottom + dC

        p = np.concatenate([side.reshape(-1, 3), bottom.reshape(-1, 3), top.reshape(-1, 3)])

        super(Cylinder, self).__init__(p=p, r=radius)

        self.properties['r1'] = r
        self.properties['r2'] = r * squeeze
        self.properties['h'] = h
        self.properties['skew'] = skew
        self.properties['pt_radius'] = radius

        self.center_to_origin()

    @staticmethod
    def _section_axes(r1, r2, h, skew):
        '''
        semi-axes of the cross-section of the nominal cylinder perpendicular to its axis (an ellipse), and z component of the unit axis vector.

        :param r1: semi-axis of the bottom face along x
        :param r2: semi-axis of the bottom face along y
        :param h: height
        :param skew: shift of the top face along y
        :returns: semi-axis along x, the other semi-axis, z component of the unit axis vector
        '''
        dz = h / np.sqrt(h**2 + skew**2)
        return r1, r2 * dz, dz

    @staticmethod
    def _check(r1, r2, h, skew, pr):
        '''
        raise an exception if the points radius does not allow points to touch the surface of the nominal cylinder everywhere.

        :param r1: semi-axis of the bottom face along x
        :param r2: semi-axis of the bottom face along y
        :param h: height
        :param skew: shift of the top face along y
        :param pr: points radius
        '''
        if min(r1, r2) <= 0:
            raise Exception("ERROR: cylinder semi-axes must be positive, found %s and %s" % (r1, r2))
        if h <= 2 * pr:
            raise Exception("ERROR: points radius %s A must be smaller than half the cylinder height (%s A)" % (pr, h / 2.0))
        a, b, _ = Cylinder._section_axes(r1, r2, h, skew)
        rmin = min(a, b)**2 / max(a, b)
        if pr >= rmin:
            raise Exception("ERROR: points radius %s A must be smaller than the minimal radius of curvature of the cylinder cross-section (%s A)" % (pr, rmin))

    @staticmethod
    def _rim(theta, r1, r2, h, skew, pr):
        '''
        boundary of the bottom face of the cylinder of points centres: every point of the nominal side is moved inward by pr along its normal, and taken at height pr.

        :param theta: angles parametrising the nominal bottom ellipse (numpy array)
        :param r1: semi-axis of the bottom face along x
        :param r2: semi-axis of the bottom face along y
        :param h: height
        :param skew: shift of the top face along y
        :param pr: points radius
        :returns: len(theta) x 3 numpy array of points
        '''
        d = np.array([0.0, skew, h])
        b = np.stack([r1 * np.cos(theta), r2 * np.sin(theta), np.zeros(len(theta))], axis=1)
        nrm = np.stack([r2 * h * np.cos(theta), r1 * h * np.sin(theta), -r1 * skew * np.sin(theta)], axis=1)
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        t0 = pr * (1 + nrm[:, 2]) / h
        return b - pr * nrm + t0[:, None] * d

    def _inner_measures(self):
        '''
        surface, integral of mean curvature and volume of the cylinder of points centres.

        The cross-section perpendicular to the axis is the inner parallel curve at distance pr of an ellipse of area A and perimeter L (complete elliptic integral), having area A' = A - L pr + pi pr^2 and perimeter L' = L - 2 pi pr.
        With l the axis length and d_z the z component of the unit axis vector: V = A' l, S = 2 A' / d_z + L' l, M = pi l + pi L_h / 2, L_h being the perimeter of the bottom face (L' for a right cylinder, otherwise computed by trapezoidal quadrature).

        :returns: surface, integral of mean curvature, volume
        '''
        r1, r2, h, skew, pr = [self.properties[k] for k in ['r1', 'r2', 'h', 'skew', 'pt_radius']]
        a, b, dz = self._section_axes(r1, r2, h, skew)
        L = _ellipse_perimeter(a, b)
        area = np.pi * a * b - L * pr + np.pi * pr**2
        perimeter = L - 2 * np.pi * pr
        length = (h - 2 * pr) / dz

        if skew == 0:
            Lh = perimeter
        else:
            # the bottom face is the cross-section stretched by 1/d_z along y
            th = np.arange(4096) * 2 * np.pi / 4096
            kappa = a * b / (a**2 * np.sin(th)**2 + b**2 * np.cos(th)**2)**1.5
            Lh = 2 * np.pi * np.mean((1 - pr * kappa) * np.sqrt((r1 * np.sin(th))**2 + (r2 * np.cos(th))**2))

        return 2 * area / dz + perimeter * length, np.pi * length + np.pi * Lh / 2.0, area * length

    def get_surface(self):
        '''
        Compute the surface of the body traced by the points (the nominal cylinder with the rims of bottom and top faces rounded by the points radius).

        :returns: surface in A^2
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'])[0]

    def get_volume(self):
        '''
        Compute the volume of the body traced by the points (the nominal cylinder with the rims of bottom and top faces rounded by the points radius).

        :returns: volume in A^3
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'])[1]

    def ccs(self, gas=1):
        '''
        Compute cylinder CCS in the projection approximation, as a quarter of the surface of the cylinder of points centres enlarged by points radius plus gas radius (see module documentation).

        :param gas: probe gas radius in Angstrom
        :returns: CCS in A^2
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'] + gas)[0] / 4.0


class Cone(Structure):
    '''
    Create an ensemble of points arranged as a cone.
    '''

    def __init__(self, r, h, skew=0, radius=1.1,
                 pts_density_r=np.pi / 32, pts_density_h=0.2):
        '''
        The nominal cone K has a circular base of radius r centred at the origin of the xy plane, and its apex at height h, shifted by skew along y.
        Points centres are placed on the boundary of K shrunk by the points radius (base and side moved inward by radius), so that every point touches the surface of K. The shape is then centred at the origin.
        Properties r, h, skew and pt_radius hold K's dimensions and the points radius.

        :param r: base radius, points included, in Angstrom
        :param h: height, points included, in Angstrom
        :param skew: shift of the apex along y with respect to the base, in Angstrom
        :param radius: radius of the individual points composing it, in Angstrom. Must be small enough for the shrunk cone not to be empty
        :param pts_density_r: angular step between points around the vertical axis, in radians
        :param pts_density_h: maximal step between points along the height, and step along the radius of the base, in Angstrom
        '''

        ulist = np.arange(0, 2 * np.pi, pts_density_r)
        q0, g, t0, tmax = self._generators(ulist, r, h, skew, radius)

        # side: from the base up to the apex, or to the ridge the shrunk side ends on
        p = []
        for i in range(len(ulist)):
            for t in t0[i] + _steps(tmax[i] - t0[i], pts_density_h / h):
                p.append(q0[i] + t * g[i])

        # base: rim scaled about its centre
        rim = q0 + t0[:, None] * g
        center = np.mean(rim, axis=0)
        rc = np.max(np.linalg.norm(rim - center, axis=1))
        slist = np.arange(0, rc, pts_density_h) / rc
        base = center + slist[None, :, None] * (rim - center)[:, None, :]

        p = np.concatenate([np.array(p), base.reshape(-1, 3)])

        super(Cone, self).__init__(p=p, r=radius)

        self.properties['r'] = r
        self.properties['h'] = h
        self.properties['skew'] = skew
        self.properties['pt_radius'] = radius

        self.center_to_origin()

    @staticmethod
    def _side(theta, r, h, skew):
        '''
        side of the nominal cone: base rim points, generator vectors (from base rim to apex) and outward unit normals.

        :param theta: angles around the vertical axis (numpy array)
        :param r: base radius
        :param h: height
        :param skew: shift of the apex along y
        :returns: three len(theta) x 3 numpy arrays
        '''
        b = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros(len(theta))], axis=1)
        g = np.array([0.0, skew, h]) - b
        nrm = np.stack([r * h * np.cos(theta), r * h * np.sin(theta), r**2 - r * skew * np.sin(theta)], axis=1)
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        return b, g, nrm

    @staticmethod
    def _generators(theta, r, h, skew, pr, n_check=4096):
        '''
        side of the cone of points centres: every nominal generator is moved inward by pr along its normal, and runs from height pr up to where it leaves the shrunk cone (tested against n_check side planes).

        :param theta: angles around the vertical axis (numpy array)
        :param r: base radius
        :param h: height
        :param skew: shift of the apex along y
        :param pr: points radius
        :param n_check: number of side planes the generators are tested against
        :returns: starting points, generator vectors, and parameters of the first and last generator points (points are start + t * vector)
        '''
        if r <= 0 or h <= 0:
            raise Exception("ERROR: cone radius and height must be positive, found %s and %s" % (r, h))
        b, g, nrm = Cone._side(theta, r, h, skew)
        bc, _, nc = Cone._side(np.arange(n_check) * 2 * np.pi / n_check, r, h, skew)

        q0 = b - pr * nrm
        t0 = pr * (1 + nrm[:, 2]) / h

        # generator points must satisfy n . x <= n . b - pr for every side plane
        D = np.dot(g, nc.T)
        rhs = np.sum(nc * bc, axis=1)[None, :] - pr - np.dot(q0, nc.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = np.where(D > 1e-9 * np.linalg.norm(g, axis=1)[:, None], rhs / D, np.inf)
        tmax = bound.min(axis=1)

        if np.min(rhs - t0[:, None] * D) < -1e-9 * (r + h) or np.any(tmax <= t0):
            raise Exception("ERROR: points radius %s A is too large for a cone of base radius %s A and height %s A" % (pr, r, h))

        return q0, g, t0, tmax

    def _inner_measures(self):
        '''
        surface, integral of mean curvature and volume of the cone of points centres.

        For a right cone (no skew) this is a cone with base radius R and height H (same apex half-angle a as the nominal cone): S = pi R^2 + pi R sqrt(R^2 + H^2), V = pi R^2 H / 3, M = pi H + pi R (pi/2 + a).
        For an oblique cone, the measures of polytopes bounded by the base and 2048 or 4096 side planes are extrapolated (Richardson, error decreasing as the inverse square of the number of planes).

        :returns: surface, integral of mean curvature, volume
        '''
        r, h, skew, pr = [self.properties[k] for k in ['r', 'h', 'skew', 'pt_radius']]
        key = (r, h, skew, pr)
        if getattr(self, "_measures_cache", (None,))[0] == key:
            return self._measures_cache[1]

        if skew == 0:
            ls = np.sqrt(r**2 + h**2)
            H = h - pr * ls / r - pr
            if H <= 0:
                raise Exception("ERROR: points radius %s A is too large for a cone of base radius %s A and height %s A" % (pr, r, h))
            R = H * r / h
            alpha = np.arctan(r / h)
            res = (np.pi * R**2 + np.pi * R * np.sqrt(R**2 + H**2), np.pi * H + np.pi * R * (np.pi / 2 + alpha), np.pi * R**2 * H / 3.0)
        else:
            res = self._numeric_measures(r, h, skew, pr)

        self._measures_cache = (key, res)
        return res

    @staticmethod
    def _numeric_measures(r, h, skew, pr, n=2048):
        '''
        surface, integral of mean curvature and volume of the cone of points centres, extrapolated from polytopes bounded by the base and n or 2n side planes.

        :param r: base radius
        :param h: height
        :param skew: shift of the apex along y
        :param pr: points radius
        :param n: number of side planes of the coarser polytope
        :returns: surface, integral of mean curvature, volume
        '''
        # a point inside: between the centre of the base and the top of the side
        theta = np.arange(64) * 2 * np.pi / 64
        q0, g, t0, tmax = Cone._generators(theta, r, h, skew, pr)
        inside = 0.75 * np.mean(q0 + t0[:, None] * g, axis=0) + 0.25 * np.mean(q0 + tmax[:, None] * g, axis=0)

        res = []
        for m in [n, 2 * n]:
            b, _, nrm = Cone._side(np.arange(m) * 2 * np.pi / m, r, h, skew)
            A = np.vstack([nrm, [0.0, 0.0, -1.0]])
            off = np.append(np.sum(nrm * b, axis=1) - pr, -pr)
            hs = HalfspaceIntersection(np.column_stack([A, -off]), inside)
            res.append(np.array(_polytope_measures(hs.intersections)))

        return tuple((4 * res[1] - res[0]) / 3.0)

    def get_surface(self):
        '''
        compute the surface of the body traced by the points (the nominal cone with base rim and apex rounded by the points radius).

        :returns: surface in A^2
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'])[0]

    def get_volume(self):
        '''
        Compute the volume of the body traced by the points (the nominal cone with base rim and apex rounded by the points radius).

        :returns: volume in A^3
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'])[1]

    def ccs(self, gas=1):
        '''
        compute cone CCS in the projection approximation, as a quarter of the surface of the cone of points centres enlarged by points radius plus gas radius (see module documentation).

        :param gas: probe gas radius in Angstrom
        :returns: CCS in A^2
        '''
        S, M, V = self._inner_measures()
        return _steiner(S, M, V, self.properties['pt_radius'] + gas)[0] / 4.0


class Sphere(Structure):
    '''
    Create an ensemble of points arranged as a sphere, which can be squeezed into an ellipsoid.

    Uses a golden spiral to approximate an even distribution.
    '''

    def __init__(self, r, radius=1.9, n_sphere_point=960):
        '''
        The nominal sphere K has radius r and is centred at the origin. Points centres are placed on a sphere of radius r - radius, so that every point touches the surface of K.
        Properties r (radius of K), p1, p2 and p3 (squeezing coefficients along x, y and z), a, b and c (semi-axes of K, r times the squeezing coefficients) and pt_radius (the points radius) describe the shape.

        :param r: radius of the sphere, points included, in Angstrom
        :param radius: radius of the individual points composing it, in Angstrom. Must be smaller than r
        :param n_sphere_point: number of points in the sphere
        '''

        _check_ellipsoid(r, r, r, radius)

        pts = []
        inc = np.pi * (3 - np.sqrt(5))
        offset = 2 / float(n_sphere_point)
        for k in range(int(n_sphere_point)):
            y = k * offset - 1 + (offset / 2)
            r2 = np.sqrt(1 - y * y)
            phi = k * inc
            pts.append([np.cos(phi) * r2, y, np.sin(phi) * r2])
        self._directions = np.array(pts)

        super(Sphere, self).__init__(p=self._directions * (r - radius), r=np.ones(n_sphere_point)*radius)

        self.properties['p1'] = 1.0  # squeezing coeff on x axis
        self.properties['p2'] = 1.0  # squeezing coeff on y axis
        self.properties['p3'] = 1.0  # squeezing coeff on z axis
        self.properties['r'] = r
        self.properties['a'] = r  # semi-axis along x
        self.properties['b'] = r  # semi-axis along y
        self.properties['c'] = r  # semi-axis along z
        self.properties['pt_radius'] = radius

    def get_surface(self):
        '''
        compute the surface of the nominal ellipsoid (semi-axes a, b and c), which the points touch from inside. Uses the exact expression with incomplete elliptic integrals.

        :returns: surface in A^2
        '''
        return _ellipsoid_surface(self.properties['a'], self.properties['b'], self.properties['c'])

    def get_volume(self):
        '''
        compute the volume of the nominal ellipsoid (semi-axes a, b and c), which the points touch from inside.

        :returns: volume in A^3
        '''
        return 4 * np.pi * self.properties['a'] * self.properties['b'] * self.properties['c'] / 3.0

    def ccs(self, gas=1):
        '''
        compute spheroid CCS in the projection approximation, as a quarter of the surface of the nominal ellipsoid enlarged by the gas radius: (S + 2 M gas + 4 pi gas^2) / 4, with S the ellipsoid surface and M its integral of mean curvature (numerical quadrature). For an unsqueezed sphere this is pi (r + gas)^2.

        :param gas: probe gas radius in Angstrom
        :returns: CCS in A^2
        '''
        a, b, c = self.properties['a'], self.properties['b'], self.properties['c']
        return _steiner(_ellipsoid_surface(a, b, c), _ellipsoid_mean_curvature(a, b, c), 0, gas)[0] / 4.0

    def squeeze(self, deformation, preserve_volume=True):
        '''
        squeeze the sphere into an ellipsoid according to deformation coefficient(s), stored as properties p1, p2 and p3, and applied to the nominal radius r (semi-axes a, b and c).
        Points are rebuilt from the nominal sphere at every call (so that coefficients do not compound), with their centres moved inward by the points radius along the ellipsoid normal, aligned with the x, y and z axes and with the centre of the nominal ellipsoid kept in place.

        :param deformation: deformation coefficient. Can be a number (deformation of the x axis), or a list of 2 (x and y axes) or 3 (x, y and z axes) numbers.
        :param preserve_volume: If true and deformation is either a number or a list of 2 numbers, correct remaining axes to preserve the volume of the nominal ellipsoid (the product of the coefficients is 1). Otherwise, the remaining coefficients keep their previous value.
        '''
        d = np.atleast_1d(np.asarray(deformation, dtype=float))
        if d.ndim != 1 or len(d) not in [1, 2, 3]:
            raise Exception("ERROR: expected a number, or a list of 2 or 3 numbers, but %s was found" % deformation)

        p = [self.properties['p1'], self.properties['p2'], self.properties['p3']]
        if len(d) == 1:
            p[0] = d[0]
            if preserve_volume:
                p[1] = 1.0 / np.sqrt(d[0])
                p[2] = 1.0 / np.sqrt(d[0])
        elif len(d) == 2:
            p[0], p[1] = d
            if preserve_volume:
                p[2] = 1.0 / (d[0] * d[1])
        else:
            p = list(d)

        a, b, c = [self.properties['r'] * float(x) for x in p]
        pr = self.properties['pt_radius']
        _check_ellipsoid(a, b, c, pr)

        # centre of the nominal ellipsoid, from the current points and the shape they were built with
        old = _ellipsoid_inner_points(self._directions, self.properties['a'], self.properties['b'], self.properties['c'], pr)
        center = self.get_center() - np.mean(old, axis=0)

        self.properties['p1'], self.properties['p2'], self.properties['p3'] = [float(x) for x in p]
        self.properties['a'], self.properties['b'], self.properties['c'] = a, b, c

        self.set_xyz(_ellipsoid_inner_points(self._directions, a, b, c, pr) + center)
        self.get_center()

    def check_inclusion(self, p):
        '''
        test which points in the array p lie inside the nominal ellipsoid (semi-axes a, b and c along x, y and z), centred at the current center of geometry.

        .. note:: the semi-axes are taken along x, y and z, as built: the orientation of a squeezed sphere is not tracked, so after a rotation the test is wrong. An unsqueezed sphere is unaffected.

        :param p: points, as an (n, 3) numpy array
        :returns: boolean array of length n, True for points inside
        '''
        center = self.get_center()
        a, b, c = self.properties['a'], self.properties['b'], self.properties['c']

        test = (p[:, 0] - center[0])**2 / a**2 + (p[:, 1] - center[1])**2 / b**2 + (p[:, 2] - center[2])**2 / c**2
        return test < 1.0

    def get_sphericity(self):
        '''
        compute sphericity of the nominal ellipsoid (makes sense only for squeezed spheres, obviously..)

        :returns: shape sphericity
        '''
        return (np.pi**(1. / 3) * (6 * self.get_volume()) ** (2. / 3)) / self.get_surface()


class Ellipsoid(Structure):
    '''
    Create an ensemble of points arranged as an ellipsoid.
    '''

    def __init__(self, a, b, c, radius=1.9, pts_density_u=np.pi /
                 36, pts_density_v=np.pi / 36):
        '''
        The nominal ellipsoid K has semi-axes a, b and c along x, y and z and is centred at the origin. Points centres are obtained by moving points of the surface of K inward by radius along the surface normal, so that every point touches the surface of K.
        Properties a, b, c and pt_radius hold K's semi-axes and the points radius.

        :param a: x radius of the ellipsoid, points included, in Angstrom
        :param b: y radius of the ellipsoid, points included, in Angstrom
        :param c: z radius of the ellipsoid, points included, in Angstrom
        :param radius: radius of the individual points composing it, in Angstrom. Must be smaller than the minimal radius of curvature of the ellipsoid, min(a, b, c)^2 / max(a, b, c)
        :param pts_density_u: angular step between rings of points along the u angle (latitude), in radians (using parametric function for ellipsoid). Rings are symmetric about the equator, so that the center of geometry of the points is the center of the ellipsoid
        :param pts_density_v: angular step between points along the v angle (longitude), in radians (using parametric function for ellipsoid)
        '''

        _check_ellipsoid(a, b, c, radius)

        n_u = max(1, int(round(np.pi / pts_density_u)))
        n_v = max(1, int(round(2 * np.pi / pts_density_v)))
        ulist = -np.pi / 2 + (np.arange(n_u) + 0.5) * np.pi / n_u
        vlist = -np.pi + np.arange(n_v) * 2 * np.pi / n_v
        u, v = [x.ravel() for x in np.meshgrid(ulist, vlist, indexing="ij")]

        # parametric function for ellipsoid surface, then moved inward
        unit = np.stack([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)], axis=1)
        p = _ellipsoid_inner_points(unit, a, b, c, radius)

        super(Ellipsoid, self).__init__(p=p, r=radius)
        self.properties['a'] = a
        self.properties['b'] = b
        self.properties['c'] = c
        self.properties['pt_radius'] = radius

        self.center_to_origin()

    def check_inclusion(self, p):
        '''
        test which points in the array p lie inside the nominal ellipsoid (semi-axes a, b and c along x, y and z), centred at the current center of geometry.

        .. note:: the semi-axes are taken along x, y and z, as built: the orientation of the ellipsoid is not tracked, so after a rotation the test is wrong.

        :param p: points, as an (n, 3) numpy array
        :returns: boolean array of length n, True for points inside
        '''
        center = self.get_center()
        test = (p[:, 0] - center[0])**2 / self.properties['a']**2 + (p[:, 1] - center[1])**2 / self.properties['b']**2 + (p[:, 2] - center[2])**2 / self.properties['c']**2
        return test < 1.0

    def get_surface(self):
        '''
        compute the surface of the nominal ellipsoid (semi-axes a, b and c), which the points touch from inside. Uses the exact expression with incomplete elliptic integrals.

        :returns: surface in A^2
        '''
        return _ellipsoid_surface(self.properties['a'], self.properties['b'], self.properties['c'])

    def get_volume(self):
        '''
        compute the volume of the nominal ellipsoid (semi-axes a, b and c), which the points touch from inside.

        :returns: volume in A^3
        '''

        return 4 * np.pi * (self.properties['a'] * self.properties['b'] * self.properties['c']) / 3

    def get_sphericity(self):
        '''
        compute sphericity of the nominal ellipsoid.

        :returns: ellipsoid sphericity
        '''

        return (np.pi**(1. / 3) * (6 * self.get_volume()) ** (2. / 3)) / self.get_surface()

    def ccs(self, gas=1):
        '''
        compute ellipsoid CCS in the projection approximation, as a quarter of the surface of the nominal ellipsoid enlarged by the gas radius: (S + 2 M gas + 4 pi gas^2) / 4, with S the ellipsoid surface and M its integral of mean curvature (numerical quadrature).

        :param gas: probe gas radius in Angstrom
        :returns: CCS in A^2
        '''
        a, b, c = self.properties['a'], self.properties['b'], self.properties['c']
        return _steiner(_ellipsoid_surface(a, b, c), _ellipsoid_mean_curvature(a, b, c), 0, gas)[0] / 4.0
