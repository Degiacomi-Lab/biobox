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

from biobox.classes.structure import Structure
import numpy as np


class Prism(Structure):
    '''
    Create an ensemble of points arranged as a prism (polygonal bottom and top, rectangular sides).
    '''

    def __init__(self, r, h, n, skew=0.0, radius=1.1,
                 pts_density_u=np.pi / 32, pts_density_h=0.2):
        '''
        :param r: distance of sides from center of symmetry
        :param h: height
        :param n: number of side faces
        :param skew: skewing with respect of vertical axis
        :param radius: size of the individual points composing it
        :param pts_density_u: density of points along the r axis (polar coordinates)
        :param pts_density_h: the density of points along the vertical axis
        '''

        super(Prism, self).__init__(r=radius)

        new_r = r - radius
        new_h = h - radius * 2

        ulist = np.arange(0, 2 * np.pi, pts_density_u)
        hlist = np.arange(0, new_h + pts_density_h, pts_density_h)

        # parametric function for prism surface (side)
        p = []
        for u in ulist:
            for v in hlist:
                radial = np.cos(np.pi / n) / np.cos(u - np.pi / n * (2 * np.floor(n * u / (2 * np.pi)) + 1)) * r
                p.append([radial * np.cos(u), radial * np.sin(u) + skew * v / hlist.max(), v])

        # parametric function for prism bottom and top
        rlist = np.arange(0, new_r, pts_density_u)
        for u in ulist:
            for r1 in rlist:
                radial = np.cos(np.pi / n) / np.cos(u - np.pi / n * (2 * np.floor(n * u / (2 * np.pi)) + 1)) * r1
                p.append([radial * np.cos(u), radial * np.sin(u), hlist.min()])
                p.append([radial * np.cos(u), radial * np.sin(u) + skew, hlist.max()])

        super(Prism, self).__init__(p=np.array(p), r=radius)

        self.properties['r'] = r - radius
        self.properties['h'] = h - radius * 2
        self.properties['n'] = n
        self.properties['skew'] = skew

        self.center_to_origin()

    def get_surface(self):
        '''
        compute prism surface.

        :returns: surface in A^2
        '''
        side = 2 * self.properties['r'] * np.sin(np.pi / self.properties['n'])
        apothem = side / (2 * np.tan(np.pi / self.properties['n']))
        return self.properties['n'] * side * apothem / 2.0 + self.properties['n'] * side * self.properties['h']

    def get_volume(self):
        '''
        compute prism volume.

        :returns: volume in A^3
        '''
        side = 2 * self.properties['r'] * np.sin(np.pi / self.properties['n'])
        apothem = side / (2 * np.tan(np.pi / self.properties['n']))
        return self.properties['n'] * side * apothem / 2.0 * self.properties['h']

    def ccs(self, gas=1):
        '''
        compute prism CCS.

        :returns: CCS in A^2
        '''
        side = 2 * (self.properties['r'] + gas) * np.sin(np.pi / self.properties['n'])
        apothem = side / (2 * np.tan(np.pi / self.properties['n']))
        return (self.properties['n'] * side * apothem / 2.0 + self.properties['n'] * side * (self.properties['h'] + 2 * gas)) / 4.0


class Cylinder(Structure):
    '''
    Create an ensemble of points arranged as an elliptical cylinder.
    '''

    def __init__(self, r, h, squeeze=1.0, skew=0.0, radius=1.1, pts_density_u=np.pi / 32, pts_density_h=0.2):
        '''
        :param r: radius
        :param h: height
        :param squeeze: create an elliptical base, having axes equal to r and squeeze*r
        :param skew: skewing with respect of vertical axis
        :param radius: size of the individual points composing it
        :param pts_density_u density: of points along the u angle (using parametric function for cylinder)
        :param pts_density_h density: of points along the v angle (using parametric function for cylinder)
        '''

        super(Cylinder, self).__init__(r=radius)

        r1 = r - radius
        r2 = (r - radius) * squeeze
        new_h = h - radius * 2

        p = []
        ulist = np.arange(0, 2 * np.pi, pts_density_u)
        hlist = np.arange(0.0, new_h + pts_density_h, pts_density_h)

        # parametric function for cylinder surface (side)
        for u in ulist:
            for v in hlist:
                p.append([r1 * np.cos(u),
                          r2 * np.sin(u) + skew * v / hlist.max(),
                          v])

        # parametric function for elliptical surface (bottom and top)
        ulist = np.arange(-np.pi / 2, np.pi / 2, pts_density_u)
        vlist = np.arange(-np.pi, np.pi, pts_density_u)
        for u in ulist:
            for v in vlist:
                p.append([r1 * np.cos(u) * np.cos(v),
                          r2 * np.cos(u) * np.sin(v),
                          hlist.min()])
                p.append([r1 * np.cos(u) * np.cos(v),
                          r2 * np.cos(u) * np.sin(v) + skew,
                          hlist.max()])


        super(Cylinder, self).__init__(p=np.array(p), r=radius)

        self.properties['r1'] = r1
        self.properties['r2'] = r2
        self.properties['h'] = new_h
        self.properties['skew'] = skew

        self.center_to_origin()

    def get_surface(self):
        '''
        Compute cylinder surface.

        Uses Ramanujan approximation for base perimeter. Good, but not perfect for very elliptical cylinders!

        :returns: surface in A^2
        '''
        basis_area = np.pi * self.properties['r1'] * self.properties['r2']
        perimeter = np.pi * (3 * (self.properties['r1'] + self.properties['r2']) - np.sqrt((3 * self.properties['r1'] + self.properties['r2']) * (self.properties['r1'] + 3 * self.properties['r2'])))
        return 2 * basis_area + perimeter * self.properties['h']

    def get_volume(self):
        '''
        Compute cylinder volume.

        :returns: volume in A^3
        '''
        return np.pi * self.properties['r1'] * \
            self.properties['r2'] * self.properties['h']

    def ccs(self, gas=1):
        '''
        Compute cylinder CCS.

        Uses Ramanujan approximation for base perimeter. Good, but not perfect for very elliptical cylinders!

        :returns: CCS in A^2
        '''
        r1 = self.properties['r1'] + gas
        r2 = self.properties['r2'] + gas
        h = self.properties['h'] + gas

        basis_area = np.pi * r1 * r2
        perimeter = np.pi * (3 * (r1 + r2) - np.sqrt((3 * r1 + r2) * (r1 + 3 * r2)))
        return (2 * basis_area + perimeter * h) / 4.0


class Cone(Structure):
    '''
    Create an ensemble of points arranged as a cone.
    '''

    def __init__(self, r, h, skew=0, radius=1.1,
                 pts_density_r=np.pi / 32, pts_density_h=0.2):
        '''
        :param r: radius
        :param h: height
        :param skew: skewing with respect of vertical axis
        :param radius: size of the individual points composing it
        :param pts_density_r: density of points along the rotation axis (using parametric function for cylinder)
        :param pts_density_h: density of points along height (using parametric function for cylinder)
        '''

        # @todo allow to squeeze the base

        new_r = r - radius
        new_h = h - radius * 2

        p = []
        ulist = np.arange(0, 2 * np.pi, pts_density_r)
        hlist = np.arange(0, new_h + pts_density_h, pts_density_h)

        # parametric function for ellipsoid surface (side)
        for u in ulist:
            for v in hlist:
                p.append([(new_h - v) / new_h * new_r *  np.cos(u),
                          (new_h - v) / new_h * new_r *  np.sin(u) + skew * v / hlist.max(),
                          v])

        # parametric function for ellipsoid surface (bottom and top)
        rlist = np.arange(0, new_r, pts_density_h)
        for u in ulist:
            for r in rlist:
                p.append([r * np.cos(u), r * np.sin(u), hlist.min()])

        super(Cone, self).__init__(p=np.array(p), r=radius)

        self.properties['r'] = new_r
        self.properties['h'] = new_h
        self.properties['skew'] = skew

        self.center_to_origin()

    def get_surface(self):
        '''
        compute cone surface.

        :returns: surface in A^2
        '''
        lateral_height = np.sqrt(self.properties['r']**2 + self.properties['h']**2)
        return np.pi * self.properties['r'] * (self.properties['r'] + lateral_height)

    def get_volume(self):
        '''
        Compute cone volume.

        :returns: volume in A^3
        '''
        return np.pi * self.properties['r']**2 * self.properties['h'] / 3

    def ccs(self, gas=1):
        '''
        compute cone CCS (use analytical solution using surface and gas effect)

        :returns: CCS in A^2
        '''
        lateral_height = np.sqrt((self.properties['r'] + gas)**2 + (self.properties['h'] + 2 * gas)**2)
        return np.pi * self.properties['r'] * (self.properties['r'] + lateral_height) / 4.0


class Sphere(Structure):
    '''
    Create an ensemble of points arranged as a sphere.

    using golden spiral to approximate an even distribution
    '''

    def __init__(self, r, radius=1.9, n_sphere_point=960):
        '''
        :param r: radius of the ellipsoid
        :param radius: size of the individual points composing it
        :param n_sphere_point: This parameter defines the amount of points in the sphere
        '''
 
        pts = []
        inc = np.pi * (3 - np.sqrt(5))
        offset = 2 / float(n_sphere_point)
        for k in range(int(n_sphere_point)):
            y = k * offset - 1 + (offset / 2)
            r2 = np.sqrt(1 - y * y)
            phi = k * inc
            pts.append([np.cos(phi) * r2, y, np.sin(phi) * r2])

        rad  = r - radius

        super(Sphere, self).__init__(p=np.array(pts) * rad, r=np.ones(n_sphere_point)*radius)

        self.properties['p1'] = 1.0  # squeezing coeff on x axis
        self.properties['p2'] = 1.0  # squeezing coeff on y axis
        self.properties['p3'] = 1.0  # squeezing coeff on z axis
        self.properties['r'] = rad
        self.properties['a'] = rad  # length of first axis (initially like overall radius)
        self.properties['b'] = rad  # length of second axis (initially like overall radius)
        self.properties['c'] = rad  # length of third axis (initially like overall radius)

        self.properties['pt_radius'] = radius  # squeezing coeff on x axis

       
    def _old_get_surface(self):
        '''
        compute sphere surface.

        :returns: surface in A^2
        '''
        return 4 * np.pi * self.properties['r']**2

    def _old_get_volume(self):
        '''
        compute sphere volume.

        :returns: volume in A^3
        '''
        return 4 * np.pi * np.power(self.properties['r'], 3) / 3.0

    def get_surface(self):
        '''
        compute sphere surface.

        :returns: surface in A^2
        '''
        a = (self.properties['r']+self.properties["pt_radius"]) * self.properties['p1']
        b = (self.properties['r']+self.properties["pt_radius"]) * self.properties['p2']
        c = (self.properties['r']+self.properties["pt_radius"]) * self.properties['p3']
        p = 1.6075
        return 4 * np.pi * np.power((a**p * b**p + a**p * c**p + b**p * c**p) / 3.0, 1.0 / p)

    def get_volume(self):
        '''
        compute ellipsoid volume.

        :returns: volume in A^3
        '''
        return 4 * np.pi * (self.properties['r'] * self.properties['a'] * self.properties['r'] * self.properties['b'] * self.properties['r'] * self.properties['c']) / 3


    #def ccs(self, gas=1):
    #    '''
    #    compute sphere CCS.
    #
    #    :returns: surface in A^2
    #    '''
    #    return np.pi * (self.properties['r'] + gas)**2

    def ccs(self, gas=1):
        '''
        compute spheroid CCS.

        Uses analytical approximation to surface area.

        :returns: surface in A^2
        '''

        a = self.properties['a']+self.properties["pt_radius"] + gas
        b = self.properties['b']+self.properties["pt_radius"] + gas
        c = self.properties['c']+self.properties["pt_radius"] + gas
        p = 1.6075
        return np.pi * np.power((a**p * b**p + a**p * c**p + b**p * c**p) / 3.0, 1.0 / p)


    def squeeze(self, deformation, preserve_volume=True):
        '''
        squeeze sphere according to deformation coefficient(s)

        :param deformation: deformation coefficient. Can be a float (deformation of one axis), or a list of 2 or 3 floats.
        :param preserve_volume: If true and deformation is either a float or a list of 2 floats, correct remaining axes to preserve volume.
        '''

        if isinstance(deformation, float):
            # squeeze one axis
            self.properties["p1"] = float(deformation)
            self.properties['a'] = self.properties["r"]*self.properties["p1"]
            if preserve_volume:
                # use the two others to correct volume
                self.properties['p2'] = 1.0 / np.sqrt(float(deformation))
                self.properties['p3'] = 1.0 / np.sqrt(float(deformation))
                self.properties['b'] = self.properties["r"]*self.properties["p2"]
                self.properties['c'] = self.properties["r"]*self.properties["p3"]

        elif len(deformation) == 2:
                # squeeze two axes and preserve volume correcting the third one
                self.properties["p1"] = float(deformation[0])
                self.properties["p2"] = float(deformation[1])
                self.properties['a'] = self.properties["r"]*self.properties["p1"]
                self.properties['b'] = self.properties["r"]* self.properties["p2"]
                if preserve_volume:
                    self.properties['p3'] = 1.0 / float(deformation[0]*deformation[1])
                    self.properties['c'] = self.properties["r"]*self.properties["p3"]

        else:
                # general deformation without volume preservation
                self.properties["p1"] = float(deformation[0])
                self.properties["p2"] = float(deformation[1])
                self.properties["p3"] = float(deformation[2])
                self.properties['a'] = self.properties["r"]*self.properties["p1"]
                self.properties['b'] = self.properties["r"]*self.properties["p2"]
                self.properties['c'] = self.properties["r"]*self.properties["p3"]

        c = self.get_center()
        self.center_to_origin()

        points = self.get_xyz()
        points[:, 0] *= self.properties['p1']
        points[:, 1] *= self.properties['p2']
        points[:, 2] *= self.properties['p3']
        self.set_xyz(points)

        self.translate(c[0], c[1], c[2])

    def check_inclusion(self, p):
        '''
        count how many points in the array p are included in the sphere.

        overloading of superclass function, which is slower (here we can use the ellipsoid functional form to speed up things)

        :param p: list of points (numpy array)
        :returns: quantity of points located inside the sphere
        '''
        self.get_center()

        test = (p[:, 0] - self.properties['center'][0])**2 / (self.properties['r'] * self.properties['a'])**2 + (p[:, 1] - self.properties['center'][1])**2 / (self.properties['r'] * self.properties['b'])**2 + (p[:, 2] - self.properties['center'][2])**2 / (self.properties['c'] * self.properties['r'])**2
        #return len(np.where(test < 1.0)[0])
        return test < 1.0 #is True, inside ellipsoid, if False, outside


    def get_sphericity(self):
        '''
        compute sphericity (makes sense only for squeezed spheres, obviously..)

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
        :param a: x radius of the ellipsoid
        :param b: y radius of the ellipsoid
        :param c: z radius of the ellipsoid
        :param radius: size of the individual points composing it
        :param pts_density_u: This parameter defines the density of points along the u angle (using parametric function for ellipsoid)
        :param pts_density_v: This parameter defines the density of points along the v angle (using parametric function for ellipsoid)
        '''

        new_a = a - radius
        new_b = b - radius
        new_c = c - radius

        p = []
        ulist = np.arange(-np.pi / 2, np.pi / 2, pts_density_u)
        vlist = np.arange(-np.pi, np.pi, pts_density_v)

        # parametric function for ellipsoid surface
        for u in ulist:
            for v in vlist:
                p.append([new_a * np.cos(u) * np.cos(v),
                          new_b * np.cos(u) * np.sin(v),
                          new_c * np.sin(u)])


        super(Ellipsoid, self).__init__(p=np.array(p), r=radius)
        self.properties['a'] = new_a
        self.properties['b'] = new_b
        self.properties['c'] = new_c

        self.center_to_origin()

        self.properties['pt_radius'] = radius  # squeezing coeff on x axis


    def check_inclusion(self, p):
        '''
        count how many points in the array p are included in the ellipsoid.

        overloading of superclass function, which is slower (here we can use the ellipsoid functional form to speed up things)

        :param p: list of points (numpy array)
        :returns: quantity of points located inside the ellipsoid
        '''
        test = (p[:, 0] - self.properties['center'][0])**2 / self.properties['a']**2 + (p[:, 1] - self.properties['center'][1])**2 / self.properties['b']**2 + (p[:, 2] - self.properties['center'][2])**2 / self.properties['c']**2
        return len(np.where(test < 1.0)[0])

    def get_surface(self):
        '''
        compute ellipsoid surface.

        Note: using analytical value, uses analytical approximation to surface area

        :returns: surface in A^2
        '''
        a = self.properties['a']
        b = self.properties['b']
        c = self.properties['c']
        p = 1.6075
        return 4 * np.pi * np.power((a**p * b**p + a**p * c**p + b**p * c**p) / 3.0, 1.0 / p)

    def get_volume(self):
        '''
        compute ellipsoid volume.

        :returns: volume in A^3
        '''

        return 4 * np.pi * (self.properties['a'] * self.properties['b'] * self.properties['c']) / 3

    def get_sphericity(self):
        '''
        compute ellipsoid sphericity.

        :returns: ellipsoid sphericity
        '''

        return (np.pi**(1. / 3) * (6 * self.get_volume()) ** (2. / 3)) / self.get_surface()

    def ccs(self, gas=1):
        '''
        compute ellipsoid CCS.

        Uses analytical approximation to surface area.

        :returns: surface in A^2
        '''
        a = self.properties['a'] + self.properties["pt_radius"] + gas
        b = self.properties['b'] + self.properties["pt_radius"] + gas
        c = self.properties['c'] + self.properties["pt_radius"] + gas
        p = 1.6075
        return np.pi * np.power((a**p * b**p + a**p * c**p + b**p * c**p) / 3.0, 1.0 / p)
