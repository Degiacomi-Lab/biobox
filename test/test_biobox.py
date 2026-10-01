import unittest
import sys, os
import numpy as np
if 'CONDA_BUILD_STATE' in os.environ and os.environ['CONDA_BUILD_STATE']=='TEST':
    pass
else:
    sys.path.insert(0, os.sep.join(os.getcwd().split(os.sep)[:-1])+os.sep+'src')
import biobox as bb

class test_density(unittest.TestCase):

    def setUp(self):
        self.D = bb.Density()
        self.D.import_map("EMD-1080.mrc", "mrc")

    def test_density_points(self):

        print("\n> density: placing points")

        try:
            self.D.place_points(5)
        except Exception:
            assert False

    def test_density_CCS(self):

        print("\n> density: CCS calculation")

        try:
            self.D.threshold_vol_ccs(sampling_points=1, append=False, noise_filter=0)
        except Exception:
            assert False


class test_structures(unittest.TestCase):

    def setUp(self):
        self.M = bb.Molecule()
        self.M.import_pdb("HSP.pdb")

    def test_len(self):
        print("\n> testing magic method:")
        print(">> atom count of monomer: %s"%len(self.M))

        M2 = self.M + self.M
        print(">> atom count of dimer: %s"%len(M2))
        #print M2[:, 20:22]

    def test_pdb_occupancy_beta_roundtrip(self):

        print("\n> testing that occupancy and beta factor survive writing a pdb")
        # distinct, atom-dependent values, so that a swap or a shift cannot go unnoticed
        import tempfile
        from copy import deepcopy
        M = deepcopy(self.M)
        n = len(M.data)
        occupancy = np.round(np.linspace(0.1, 1.0, n), 2)
        beta = np.round(np.linspace(10.0, 90.0, n), 2)
        M.data["occupancy"] = occupancy
        M.data["beta"] = beta

        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "roundtrip.pdb")
            M.write_pdb(fname)

            # the columns as written: occupancy in 55-60, beta factor in 61-66
            with open(fname) as f:
                lines = [l for l in f if l.startswith(("ATOM", "HETATM"))]
            np.testing.assert_allclose([float(l[54:60]) for l in lines], occupancy)
            np.testing.assert_allclose([float(l[60:66]) for l in lines], beta)

            # and as read back
            M2 = bb.Molecule()
            M2.import_pdb(fname)
            np.testing.assert_allclose(M2.data["occupancy"].astype(float), occupancy)
            np.testing.assert_allclose(M2.data["beta"].astype(float), beta)

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

        except Exception:
            assert False


    def test_SASA(self):

        print("\n> testing molecule's SASA")
        try:
            [sasa, mesh, surf_idx] = bb.sasa(self.M, n_sphere_point=400)
        except Exception:
            assert False

    def test_SASA_isolated_atom(self):

        print("\n> testing SASA of an isolated atom")
        # an atom must not occlude its own mesh, so alone it exposes its whole sphere
        S = self.M.get_subset(idxs=[1])
        r = S.data['radius'].values[0]
        asa = bb.sasa(S, probe=1.4, n_sphere_point=960, threshold=0)[0]
        self.assertAlmostEqual(asa, 4*np.pi*(r+1.4)**2, places=6)

    def test_SASA_atom_pair(self):

        print("\n> testing SASA of an atom pair against direct counting")
        # 6.0 A lies beyond radii.max()+2*probe but within r_i+r_j+2*probe, a neighbour the
        # former neighbour search missed; 7.0 A is out of reach
        from biobox.measures.calculators import _golden_spiral
        S = self.M.get_subset(idxs=[1, 2])
        R = S.data['radius'].values
        for d in [3.0, 5.0, 6.0, 7.0]:
            S.coordinates[0] = np.array([[0, 0, 0], [d, 0, 0]], dtype=float)
            S.set_current(0)
            mesh = _golden_spiral(960)*(R[0]+1.4)
            exposed = np.count_nonzero(np.linalg.norm(mesh-[d, 0, 0], axis=1)-R[1] >= 1.4)
            expected = 4*np.pi/960*exposed*(R[0]+1.4)**2
            asa = bb.sasa(S, targets=[0], probe=1.4, n_sphere_point=960, threshold=0)[0]
            self.assertAlmostEqual(asa, expected, places=6)

    def test_SASA_targets(self):

        print("\n> testing SASA of a subset of targets")
        # atoms outside the targets still occlude, and the area of a set of targets is the
        # sum of their individual areas
        idx = [10, 20, 30]
        together = bb.sasa(self.M, targets=idx, threshold=0)[0]
        apart = sum(bb.sasa(self.M, targets=[i], threshold=0)[0] for i in idx)
        alone = bb.sasa(self.M.get_subset(idxs=idx), threshold=0)[0]
        self.assertAlmostEqual(together, apart, places=6)
        self.assertLess(together, alone)

    def test_rmsd_one_vs_all_no_reflection(self):

        print("\n> testing that alignment never mirrors a structure")
        # the mirror image of a protein is best matched by a reflection, which alignment must
        # not apply: a proper rotation keeps the handedness of every group of four atoms
        from copy import deepcopy
        P = deepcopy(self.M)
        mirror = P.coordinates[0].copy()
        mirror[:, 0] = -mirror[:, 0]
        P.add_xyz(mirror)

        def volume(X):
            return np.dot(X[1]-X[0], np.cross(X[2]-X[0], X[3]-X[0]))

        before = volume(P.coordinates[1, :4])
        rmsd = P.rmsd_one_vs_all(0, align=True)
        self.assertEqual(np.sign(volume(P.coordinates[1, :4])), np.sign(before))

        # and the RMSD returned is the one of the coordinates the alignment left behind
        direct = np.sqrt(np.mean(np.sum((P.coordinates[1]-P.coordinates[0])**2, axis=1)))
        self.assertAlmostEqual(rmsd[1], direct, places=4)

    def test_SASA_c(self):

        print("\n> testing that sasa_c agrees with sasa")
        self.assertEqual(bb.sasa_c(self.M, n_sphere_point=400)[0],
                         bb.sasa(self.M, n_sphere_point=400)[0])


    def test_monomer_CCS(self):
        if 'IMPACTPATH' in os.environ:
            print("\n> testing CCS")
        else:
            print("\n\n> IMPACTPATH not set therefore can't test CCS. \n WARNING: If you want CCS calculations you need to set IMPACTPATH")
            return

        try:
            ccs1 = bb.ccs(self.M)
        except Exception:
            assert False

        try:
            ccs2 = bb.ccs(self.M, use_lib=False)
        except Exception:
            assert False

        self.assertAlmostEqual(ccs1, ccs2, delta=ccs2/10.0) #max 10% difference


    def test_multimer_CCS(self):
        if 'IMPACTPATH' in os.environ:
            print("\n> testing multimer CCS")
        else:
            print("\n\n> IMPACTPATH not set therefore can't test CCS. \n WARNING: If you want CCS calculations you need to set IMPACTPATH")
            return


        try:
            A = bb.Multimer()
            A.load(self.M, 3)
            A.make_circular_symmetry(30)
            bb.ccs(A)
        except Exception:
            assert False

    def test_multimer_selections(self):

        print("\n> testing multimer atomselect and query")
        try:
            A = bb.Multimer()
            A.load_list([self.M, self.M], ["1", "2"])
        except Exception:
            assert False

        pts_test = self.M.atomselect("*", "LYS", "CA", use_resname = True)

        try:
            pts = A.query('unit == "1" and resname == "LYS" and name == "CA"')
        except Exception:
            assert False

        self.assertEqual(len(pts), len(pts_test))

        try:
            pts = A.atomselect("1", "*", "LYS", "CA", use_resname = True)
        except Exception:
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
        except Exception:
            assert False


    #test rototranslations on double disks (prism method)
    def test_monomers_rototranslations(self):

        print("\n> testing monomer rototranslations")
        try:
            self.M.align_axes()
            self.M.rotate(10, 10, 10)
            self.M.translate(20, 20, 20)

        except Exception:
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

            #different conformations must be selected
            self.assertNotAlmostEqual(a1[0, 0], a2[0, 0])

        except Exception:
            assert False


    #create all convex shapes
    def test_shapes(self):

        print("\n> testing convex shapes")
        try:
            C1 = bb.Prism(10, 20, 5)
            C1.get_surface()
            C1.get_volume()

            C2 = bb.Cylinder(10, 50)
            C2.get_surface()
            C2.get_volume()

            C3 = bb.Cone(10, 30)
            C3.get_surface()
            C3.get_volume()

            C4 = bb.Ellipsoid(10, 20, 30)
            C4.get_surface()
            C4.get_volume()

            C5 = bb.Sphere(10)
            C5.get_surface()
            C5.get_volume()

        except Exception:
            assert False


if __name__ == '__main__':
    unittest.main()
