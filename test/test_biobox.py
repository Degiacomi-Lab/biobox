import unittest

from biobox import Multimer, Molecule, Structure, Sphere, Cone, Cylinder, Ellipsoid, Prism, Density, Xlink
import os

class test_biobox(unittest.TestCase):
    
    def setUp(self):
        self.M = Molecule()
        self.M.import_pdb("HSP.pdb")
    

    def test_xlink(self):
    
        print "\n> testing shortest path"
    
        try:
            #extract indices of atoms to connect
            idx = self.M.atomselect("*", "LYS", "NZ", use_resname=True, get_index=True)[1]
    
            #prepare xlink measurer
            XL = Xlink(self.M)
            XL.set_clashing_atoms(atoms=["CA", "C", "N", "O", "CB"], densify=True, atoms_vdw=False)
        
            XL.setup_global_search(maxdist=14, use_hull=False)  
            #XL.setup_local_search(maxdist=24)
    
            distance2, paths = XL.distance_matrix(idx, method="theta", get_path=True, smooth=True, verbose=False, flexible_sidechain=True, test_los=True)

        except Exception, ex:
            assert False
 

    def test_density(self):

        print "\n> testing EM map handling"
 
        try:       
            D = Density()
            D.import_map("EMD-1080.mrc", "mrc")
            D.place_points(0.1)
        except Exception, ex:
            assert False
 

    def test_SASA(self):

        print "\n> testing molecule's SASA"
        try:
            [sasa, mesh, surf_idx] = self.M.get_surface(n_sphere_point=400)
        except Exception, ex:
            assert False


    def test_monomer_CCS(self):

        print "\n> testing CCS"
        
        try:
            ccs1 = self.M.ccs()
            ccs2 = self.M.ccs(use_lib=False)
        except Exception, ex:
            assert False

        self.assertAlmostEqual(ccs1, ccs2, delta=ccs2/10.0) #max 10% difference


    def test_multimer_CCS(self):

        print "\n> testing multimer CCS"        

        try:
            A = Multimer()
            A.load(self.M, 3)
            A.make_circular_symmetry(30)
            A.ccs()
        except Exception, ex:
            assert False


    def test_multimer_rototranslations(self):
        
        print "\n> testing multimer rototranslations"
        try:
            P = Multimer()
            P.load(self.M, 6)
            P.rotate(0, 0, 90)
            P.make_prism(25, 15, 180, 45, 90)
            P.rotate(10, 10, 10, [1, 2])
        except Exception, ex:
            assert False


    #test rototranslations on double disks (prism method)
    def test_monomers_rototranslations(self):

        print "\n> testing monomer rototranslations"
        try:
            self.M.align_axes()
            self.M.rotate(10, 10, 10)
            self.M.translate(20, 20, 20)
                    
        except Exception, ex:
            assert False


    #test assembly of multiple polyhedral architectures, and RMSD evaluation
    def test_polyRMSD(self):

        print "\n> assemblying Polyhedra"
        try:    
            #setup desired polyhedron
            P = Multimer()
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
            
        except Exception, ex:
            assert False


    #create all convex shapes
    def test_shapes(self):

        #print "\n> testing convex shapes
        try:
            C1 = Prism(10, 20, 5)
            s1 = C1.get_surface()
            v1 = C1.get_volume()
            
            C2 = Cylinder(10, 50)
            s2 = C1.get_surface()
            v2 = C1.get_volume()
    
            C3 = Cone(10, 30)
            s3 = C1.get_surface()
            v3 = C1.get_volume()
    
            C4 = Ellipsoid(10, 20, 30)
            s4 = C4.get_surface()
            v4 = C4.get_volume()
    
            C5 = Sphere(10)
            s5 = C5.get_surface()
            v5 = C5.get_volume()
        
        except Exception, ex:
            assert False


if __name__ == '__main__':
    unittest.main()


