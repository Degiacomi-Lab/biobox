import unittest

import biobox as bb


class test_density(unittest.TestCase):

    def setUp(self):
        self.D = bb.Density()
        self.D.import_map("EMD-1080.mrc", "mrc")

    def test_density_points(self):

        print("\n> density: placing points")
 
        try:       
            self.D.place_points(5)            
        except Exception as ex:
            assert False

    def test_density_CCS(self):

        print("\n> density: CCS calculation")
 
        try:          
            self.D.threshold_vol_ccs(sampling_points=1, append=False, noise_filter=0)
        except Exception as ex:
            assert False


class test_structures(unittest.TestCase):
    
    def setUp(self):
        self.M = bb.Molecule()
        self.M.import_pdb("HSP.pdb")
    
    def test_len(self):
        print("\n> testing magic methods")
        print(">> atom count: %s"%len(self.M))
        print(">> frames count: %s"%len(self.M, "frames"))

        M2 = self.M + self.M
        print(">> atom count of extension: %s"%len(M2))
        print M2[:, 20:22]

    def test_xlink(self):
    
        print("\n> testing shortest path")
    
        try:
            #extract indices of atoms to connect
            idx = self.M.atomselect("*", "LYS", "NZ", use_resname=True, get_index=True)[1]
    
            #prepare xlink measurer
            XL = bb.Xlink(self.M)
            XL.set_clashing_atoms(atoms=["CA", "C", "N", "O", "CB"], densify=True, atoms_vdw=False)
        
            XL.setup_global_search(maxdist=14, use_hull=False)  
            #XL.setup_local_search(maxdist=24)
    
            distance2, paths = XL.distance_matrix(idx, method="theta", get_path=True, smooth=True, verbose=False, flexible_sidechain=True, test_los=True)

        except Exception as ex:
            assert False
             

    def test_SASA(self):

        print("\n> testing molecule's SASA")
        try:
            [sasa, mesh, surf_idx] = bb.sasa(self.M, n_sphere_point=400)
        except Exception as ex:
            assert False


    def test_monomer_CCS(self):

        print("\n> testing CCS")
        
        try:
            ccs1 = bb.ccs(self.M)
            ccs2 = bb.ccs(self.M, use_lib=False)
        except Exception as ex:
            assert False

        self.assertAlmostEqual(ccs1, ccs2, delta=ccs2/10.0) #max 10% difference


    def test_multimer_CCS(self):

        print("\n> testing multimer CCS")

        try:
            A = bb.Multimer()
            A.load(self.M, 3)
            A.make_circular_symmetry(30)
            bb.ccs(A)
        except Exception as ex:
            assert False

    def test_multimer_selections(self):

        print("\n> testing multimer atomselect and query")
        try:        
            A = bb.Multimer()
            A.load_list([self.M, self.M], ["1", "2"])
        except Exception as ex:
            assert False

        pts_test = self.M.atomselect("*", "LYS", "CA", use_resname = True)

        try:
            pts = A.query('unit == "1" and resname == "LYS" and name == "CA"')
        except Exception as ex:
            assert False

        self.assertEqual(len(pts), len(pts_test))

        try:
            pts = A.atomselect("1", "*", "LYS", "CA", use_resname = True)
        except Exception as ex:
            assert False

        self.assertEqual(len(pts), len(pts_test))


    def test_multimer_rototranslations(self):
        
        print("\n> testing multimer rototranslations")
        try:
            P = bb.Multimer()
            P.load(self.M, 6)
            P.rotate(0, 0, 90)
            P.make_prism(25, 15, 180, 45, 90)
            P.rotate(10, 10, 10, [1, 2])
        except Exception as ex:
            assert False


    #test rototranslations on double disks (prism method)
    def test_monomers_rototranslations(self):

        print("\n> testing monomer rototranslations")
        try:
            self.M.align_axes()
            self.M.rotate(10, 10, 10)
            self.M.translate(20, 20, 20)
                    
        except Exception as ex:
            assert False


    #test assembly of multiple polyhedral architectures, and RMSD evaluation
    def test_polyRMSD(self):

        print("\n> assemblying Polyhedra")
        try:    
            #setup desired polyhedron
            P = bb.Multimer()
            P.setup_polyhedron("Octahedron", self.M)
        
            #try creation and and deletion of some polyhedra
            P.generate_polyhedron(40, 180, 0, 0)
            P.generate_polyhedron(42, 180, 5, 0, add_conformation=True)
            P.generate_polyhedron(40, 200, 5, 0, add_conformation=True)
            P.generate_polyhedron(40, 180, 10, 10, add_conformation=True)
            P.generate_polyhedron(40, 180, 5, 5, add_conformation=True)
            P.delete_xyz(2)
            
            #test atomselects on alternate conformations
            P.set_current(0)
            a1 = P.atomselect(["1", "2"], "*", 90, "CA")
            P.set_current(1)
            a2 = P.atomselect(["1", "2"], "*", 90, "CA")
            
        except Exception as ex:
            assert False


    #create all convex shapes
    def test_shapes(self):

        print("\n> testing convex shapes")
        try:
            C1 = bb.Prism(10, 20, 5)
            C1.get_surface()
            C1.get_volume()
            
            C2 = bb.Cylinder(10, 50)
            C1.get_surface()
            C1.get_volume()
    
            C3 = bb.Cone(10, 30)
            C1.get_surface()
            C1.get_volume()
    
            C4 = bb.Ellipsoid(10, 20, 30)
            C4.get_surface()
            C4.get_volume()
    
            C5 = bb.Sphere(10)
            C5.get_surface()
            C5.get_volume()
        
        except Exception as ex:
            assert False


if __name__ == '__main__':
    unittest.main()
