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

    def test_import_without_end(self):

        print("\n> testing that charges are loaded from files without an END statement")
        import tempfile
        pdb = ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N\n",
               "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C\n"]
        pqr = ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  0.1414 1.8240\n",
               "ATOM      2  CA  ALA A   1       1.458   0.000   0.000 -0.0597 1.9080\n"]

        with tempfile.TemporaryDirectory() as tmp:
            for end in ["", "END\n"]:
                fname = os.path.join(tmp, "test.pdb")
                with open(fname, "w") as f:
                    f.writelines(pdb + [end])
                M = bb.Molecule()
                M.import_pdb(fname)
                np.testing.assert_allclose(M.data["charge"].values, [0.0, 0.0])

                fname = os.path.join(tmp, "test.pqr")
                with open(fname, "w") as f:
                    f.writelines(pqr + [end])
                M = bb.Molecule()
                M.import_pqr(fname)
                np.testing.assert_allclose(M.data["charge"].values, [0.1414, -0.0597])
                np.testing.assert_allclose(M.data["radius"].values, [1.8240, 1.9080])

    def test_import_gro(self):

        print("\n> testing selections on a molecule loaded from a gro file")
        import tempfile
        gro = ["two residues\n", "    3\n",
               "    1ALA      N    1   0.000   0.000   0.000\n",
               "    1ALA     CA    2   0.100   0.000   0.000\n",
               "    2ALA      N    3   0.200   0.000   0.000\n",
               "   1.00000   1.00000   1.00000\n"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.gro")
            with open(fname, "w") as f:
                f.writelines(gro)
            M = bb.Molecule()
            M.import_gro(fname)

        self.assertEqual(list(M.data["index"]), [0, 1, 2])
        self.assertEqual(len(M.atomselect("A", 1, "CA")), 1)
        self.assertEqual(len(M.atomselect("A", [1, 2], "*")), 3)
        self.assertEqual(len(M.query("resid == 1")), 2)
        self.assertEqual(len(M.get_subset([0], flip=True)), 2)
        np.testing.assert_allclose(M.points[:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(M.data["occupancy"].values, [1.0, 1.0, 1.0])

        print("\n> testing elements and radii guessed from gro atom names")
        # an ion is named as its residue, protein and water names come from the atomtype
        # table, other names from their first letter, and unknown names get the default radius
        atoms = [("ALA", "CA"), ("CA", "CA"), ("NA", "NA"), ("SOD", "SOD"), ("SOL", "OW"),
                 ("SOL", "HW1"), ("LIG", "C12"), ("LIG", "1HD1"), ("LIG", "XX1")]
        lines = ["guesses\n", "%5d\n" % len(atoms)]
        for i, (resname, name) in enumerate(atoms):
            lines.append("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (i+1, resname, name, i+1, 0.3*i, 0, 0))
        lines.append("   5.00000   5.00000   5.00000\n")
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "guess.gro")
            with open(fname, "w") as f:
                f.writelines(lines)
            M = bb.Molecule()
            M.import_gro(fname)

        self.assertEqual(list(M.data["atomtype"]), ["C", "CA", "NA", "NA", "O", "H", "C", "H", ""])
        np.testing.assert_allclose(M.data["radius"].values, [1.70, 2.31, 2.27, 2.27, 1.52, 1.20, 1.70, 1.20, 1.80])

    def test_atomselect_resid_types(self):

        print("\n> testing atomselect with numpy and string residue IDs")
        expected = self.M.atomselect("*", 33, "CA", get_index=True)[1]
        self.assertEqual(len(expected), 2)
        r = np.unique(self.M.data["resid"])[0]
        self.assertIsInstance(r, np.integer)
        for res in [r, "33", [r], np.array([33]), ["33"]]:
            np.testing.assert_array_equal(self.M.atomselect("*", res, "CA", get_index=True)[1], expected)

        both = self.M.atomselect("*", [33, 34], "CA", get_index=True)[1]
        self.assertEqual(len(both), 4)

        lys = self.M.atomselect("*", np.str_("LYS"), "NZ", use_resname=True, get_index=True)[1]
        self.assertEqual(len(lys), len(self.M.atomselect("*", "LYS", "NZ", use_resname=True, get_index=True)[1]))
        self.assertGreater(len(lys), 0)

        with self.assertRaises(Exception):
            self.M.atomselect("*", "LYS", "NZ")

    def test_same_residue_list(self):

        print("\n> testing same_residue with a list of atoms")
        first = self.M.same_residue(0, get_index=True)[1]
        second = self.M.same_residue(20, get_index=True)[1]
        both = self.M.same_residue([0, 20], get_index=True)[1]
        np.testing.assert_array_equal(both, np.union1d(first, second))
        np.testing.assert_array_equal(self.M.same_residue([0, 1], get_index=True)[1], first)

    def test_get_subset_mask(self):

        print("\n> testing get_subset with a boolean mask")
        mask = self.M.data["name"].values == "CA"
        idx = np.where(mask)[0]
        S = self.M.get_subset(mask)
        self.assertEqual(len(S), len(idx))
        np.testing.assert_allclose(S.points, self.M.points[idx])
        self.assertEqual(len(self.M.get_subset(mask, flip=True)), len(self.M) - len(idx))
        self.assertEqual(len(self.M.get_subset([])), 0)

    def _write_biomt_pdb(self, fname, remarks, models):
        # write a pdb with the given REMARK lines and one MODEL per list of (chain, resid, xyz, occupancy, beta)
        with open(fname, "w") as f:
            f.writelines(remarks)
            for i, atoms in enumerate(models):
                f.write("MODEL     %4d\n" % (i+1))
                for j, (chain, resid, xyz, occ, beta) in enumerate(atoms):
                    f.write("ATOM  %5d  CA  ALA %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           C\n" % (j+1, chain, resid, xyz[0], xyz[1], xyz[2], occ, beta))
                f.write("ENDMDL\n")
            f.write("END\n")

    def _biomt_lines(self, mats, record="REMARK 350   BIOMT"):
        lines = []
        for n, m in enumerate(mats):
            for row in range(3):
                lines.append("%s%d %3d%10.6f%10.6f%10.6f%15.5f\n" % (record, row+1, n+1, m[row, 0], m[row, 1], m[row, 2], m[row, 3]))
        return lines

    def test_apply_biomatrix(self):

        print("\n> testing biological assembly construction from BIOMT")
        import tempfile
        R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        identity = np.hstack([np.eye(3), np.zeros((3, 1))])
        rototranslation = np.hstack([R, [[10.], [0.], [0.]]])
        shift = np.hstack([np.eye(3), [[0.], [0.], [5.]]])

        # biomolecule 1: both operators on chains A and B (the chain list continues on an AND line)
        # biomolecule 2: chain A unchanged, chain B shifted. Chain C is not part of any assembly
        remarks = ["REMARK 350 BIOMOLECULE: 1\n",
                   "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A,\n",
                   "REMARK 350                    AND CHAINS: B\n"] + self._biomt_lines([identity, rototranslation]) + \
                  ["REMARK 350 BIOMOLECULE: 2\n",
                   "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A\n"] + self._biomt_lines([identity]) + \
                  ["REMARK 350 APPLY THE FOLLOWING TO CHAINS: B\n"] + self._biomt_lines([shift])
        frame1 = [("A", 1, [1, 0, 0], 0.25, 77.0), ("A", 2, [2, 1, 0], 0.5, 11.0),
                  ("B", 1, [0, 3, 1], 0.75, 22.0), ("C", 1, [9, 9, 9], 1.0, 33.0)]
        frame2 = [(c, r, np.array(x) + 1, o, b) for c, r, x, o, b in frame1]

        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "biomt.pdb")
            self._write_biomt_pdb(fname, remarks, [frame1, frame2])
            M = bb.Molecule()
            M.import_pdb(fname)

        self.assertEqual(sorted(M.properties["biomatrix"]), [1, 2])
        self.assertEqual(M.properties["biomatrix"][1][0][0], ["A", "B"])

        B = M.apply_biomatrix()
        xyz = M.coordinates[:, :3]
        expected = np.concatenate([xyz, np.dot(xyz, R.T) + [10, 0, 0]], axis=1)
        np.testing.assert_allclose(B.coordinates, expected, atol=1e-6)
        np.testing.assert_allclose(B.points, B.coordinates[0])
        self.assertEqual(list(B.data["chain"]), ["A", "A", "B", "D", "D", "E"])
        np.testing.assert_allclose(B.data["occupancy"].values, [0.25, 0.5, 0.75] * 2)
        np.testing.assert_allclose(B.data["beta"].values, [77.0, 11.0, 22.0] * 2)
        self.assertEqual(list(B.data["index"]), list(range(6)))
        self.assertEqual(list(B.data.index), list(range(6)))
        self.assertIn("radius", B.data.columns)
        self.assertEqual(len(B.query("chain == 'D'")), 2)
        self.assertEqual(len(B.get_subset([0, 3])), 2)

        B2 = M.apply_biomatrix(2)
        self.assertEqual(list(B2.data["chain"]), ["A", "A", "B"])
        np.testing.assert_allclose(B2.points, xyz[0] + [[0, 0, 0], [0, 0, 0], [0, 0, 5]])

        with self.assertRaises(Exception):
            M.apply_biomatrix(3)

    def test_apply_matrices_chain_names(self):

        print("\n> testing chain names of many copies, and SMTRY operators")
        import tempfile
        mats = [np.hstack([np.eye(3), [[10.*i], [0.], [0.]]]) for i in range(70)]
        remarks = self._biomt_lines(mats[:2], record="REMARK 290   SMTRY") + self._biomt_lines(mats)
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "copies.pdb")
            self._write_biomt_pdb(fname, remarks, [[("A", 1, [0, 0, 0], 1.0, 0.0)]])
            M = bb.Molecule()
            M.import_pdb(fname)

        # matrices without a chain list apply to all chains, and names run on to two characters
        B = M.apply_biomatrix()
        self.assertEqual(len(set(B.data["chain"])), 70)
        self.assertEqual(B.data["chain"].values[0], "A")
        self.assertTrue(any(len(c) == 2 for c in B.data["chain"]))
        np.testing.assert_allclose(B.points[:, 0], 10.*np.arange(70))

        S = M.apply_symmetry()
        np.testing.assert_allclose(S.points, [[0, 0, 0], [10, 0, 0]])

    def _molecule_from_atoms(self, atoms):
        # build a molecule from a list of (name, element, chain, resid, xyz)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "atoms.pdb")
            with open(fname, "w") as f:
                for i, (name, element, chain, resid, xyz) in enumerate(atoms):
                    f.write("ATOM  %5d  %-3s ALA %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n" % (i+1, name, chain, resid, xyz[0], xyz[1], xyz[2], element))
                f.write("END\n")
            M = bb.Molecule()
            M.import_pdb(fname)
        return M

    def _voxel_xyz(self, D, idx):
        return D.properties["origin"] + np.array(idx) * np.diag(D.properties["delta"])

    def test_density_origin(self):

        print("\n> testing that density maps place atoms at their coordinates")
        M = self._molecule_from_atoms([("CA", "C", "A", 1, [1.0, 2.0, 3.0]), ("CB", "C", "A", 1, [5.0, 2.0, 3.0])])
        for D in [M.get_density(step=1.0), M.get_vdw_density(step=0.5)]:
            dens = D.properties["density"]
            first = np.unravel_index(np.argmax(dens[:dens.shape[0]//2]), dens.shape)
            np.testing.assert_allclose(self._voxel_xyz(D, first), [1.0, 2.0, 3.0], atol=1e-9)
            np.testing.assert_allclose(D.properties["origin"], [1.0 - 3, 2.0 - 3, 3.0 - 3])

    def test_vdw_density_per_type(self):

        print("\n> testing that each atom type contributes only its own atoms")
        C = self._molecule_from_atoms([("CA", "C", "A", 1, [0, 0, 0]), ("CB", "C", "A", 1, [3, 0, 0])])
        D = C.get_vdw_density(step=0.5)
        axes = C._grid_axes(C.points, 0.5, 3)
        np.testing.assert_allclose(D.properties["density"], C._density_on_grid(C.points, axes, 0.5, 1.455, 10))

        # without carbons, the hydrogen map alone is returned
        H = self._molecule_from_atoms([("H", "H", "A", 1, [0, 0, 0])])
        D = H.get_vdw_density(step=0.5)
        axes = H._grid_axes(H.points, 0.5, 3)
        np.testing.assert_allclose(D.properties["density"], H._density_on_grid(H.points, axes, 0.5, 0.72, 10))

        # in a mixed molecule, the map is the sum of the carbon and the hydrogen maps
        CH = self._molecule_from_atoms([("CA", "C", "A", 1, [0, 0, 0]), ("H", "H", "A", 1, [3, 0, 0])])
        D = CH.get_vdw_density(step=0.5)
        axes = CH._grid_axes(CH.points, 0.5, 3)
        expected = CH._density_on_grid(CH.points[:1], axes, 0.5, 1.455, 10) + CH._density_on_grid(CH.points[1:], axes, 0.5, 0.72, 10)
        np.testing.assert_allclose(D.properties["density"], expected)

    def test_electrostatics(self):

        print("\n> testing electrostatic maps")
        atoms = [("CA", "C", "A", 1, [0, 0, 0]), ("CB", "C", "A", 1, [0.2, 0, 0]),
                 ("CA", "C", "B", 1, [17, 0, 0])]

        # charges falling in the same voxel add up
        M = self._molecule_from_atoms(atoms[:2])
        M.data["charge"] = [1.0, 1.0]
        P1 = M.get_electrostatics(clear_mass=False)[0].properties["density"]
        M.data["charge"] = [2.0, 0.0]
        P2 = M.get_electrostatics(clear_mass=False)[0].properties["density"]
        M.data["charge"] = [0.0, 2.0]
        P3 = M.get_electrostatics(clear_mass=False)[0].properties["density"]
        np.testing.assert_allclose(P1, P2)
        np.testing.assert_allclose(P2, P3)
        self.assertGreater(P1.sum(), 0)

        # the potential of a single charge is centred on it, whatever the voxel size
        M = self._molecule_from_atoms(atoms[:1])
        M.data["charge"] = [1.0]
        for step in [1.0, 0.5]:
            D = M.get_electrostatics(step=step, clear_mass=False)[0]
            dens = D.properties["density"]
            grid = np.indices(dens.shape).reshape(3, -1).T
            centroid = self._voxel_xyz(D, (grid * dens.reshape(-1, 1)).sum(axis=0) / dens.sum())
            np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-6)

        # a chain gives the same maps whether selected or alone
        M = self._molecule_from_atoms(atoms)
        M.data["charge"] = [1.0, 0.0, -1.0]
        B = M.get_subset(M.atomselect("B", "*", "*", get_index=True)[1])
        for selected, alone in zip(M.get_electrostatics(chain="B"), B.get_electrostatics()):
            np.testing.assert_allclose(selected.properties["density"], alone.properties["density"])
            np.testing.assert_allclose(selected.properties["origin"], alone.properties["origin"])
        np.testing.assert_allclose(M.get_electrostatics(chain="B")[1].properties["origin"], [17 - 3, -3, -3])

    def test_write_pdb_columns(self):

        print("\n> testing atom name columns and serial numbers in written pdb files")
        import tempfile
        M = self._molecule_from_atoms([("CA", "C", "A", 1, [0, 0, 0]), ("CB", "C", "A", 1, [1, 0, 0]), ("CG", "C", "A", 1, [2, 0, 0])])
        M.data["name"] = ["CA", "HD11", "1HD1"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "names.pdb")
            M.write_pdb(fname)
            lines = [l for l in open(fname) if l.startswith("ATOM")]
        # columns 13-16 hold the name, column 17 the (empty) altloc
        self.assertEqual([l[12:17] for l in lines], [" CA  ", "HD11 ", "1HD1 "])
        self.assertEqual([l[6:11] for l in lines], ["    1", "    2", "    3"])

        h36 = bb.Molecule._hybrid36
        self.assertEqual([h36(v) for v in [1, 99999, 100000, 100001, 100000 + 26*36**4]], ["1", "99999", "A0000", "A0001", "a0000"])

    def test_write_pdb_split(self):

        print("\n> testing TER records of a written subset")
        import tempfile
        from copy import deepcopy
        M = deepcopy(self.M)
        M.add_xyz(M.coordinates[0] + 1.0)
        chains = M.data["chain"].values.copy()
        index = np.arange(1000, len(M))
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "split.pdb")
            M.write_pdb(fname, index=index, split_struc=True)
            models = open(fname).read().split("ENDMDL")[:-1]

        np.testing.assert_array_equal(M.data["chain"].values, chains)
        self.assertEqual(len(models), 2)
        for model in models:
            lines = model.splitlines()
            ter = [i for i, l in enumerate(lines) if l.startswith("TER")]
            atoms_before = [len([l for l in lines[:i] if l.startswith("ATOM")]) for i in ter]
            self.assertEqual(atoms_before, [618, 642, 666])
        self.assertEqual(models[0].count("TER"), models[1].count("TER"))
        chains_1 = [l[21] for l in models[0].splitlines() if l.startswith("ATOM")]
        chains_2 = [l[21] for l in models[1].splitlines() if l.startswith("ATOM")]
        self.assertEqual(chains_1, chains_2)

    def test_write_gro(self):

        print("\n> testing atom numbers and box of written gro files")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "out.gro")
            self.M.write_gro(fname)
            lines = open(fname).readlines()
            self.M.write_gro(fname, index=[5, 9])
            sub = open(fname).readlines()

        self.assertEqual([int(l[15:20]) for l in lines[2:5]], [1, 2, 3])
        span = (self.M.points.max(axis=0) - self.M.points.min(axis=0)) / 10.0
        np.testing.assert_allclose([float(x) for x in lines[-1].split()], span, atol=1e-5)
        self.assertEqual([int(l[15:20]) for l in sub[2:4]], [1, 2])
        span = (self.M.points[[5, 9]].max(axis=0) - self.M.points[[5, 9]].min(axis=0)) / 10.0
        np.testing.assert_allclose([float(x) for x in sub[-1].split()], span, atol=1e-5)

    def test_write_pqr(self):

        print("\n> testing charge and radius columns of written pqr files")
        import tempfile
        import pandas as pd
        M = self._molecule_from_atoms([("N", "N", "A", 1, [0, 0, 0]), ("CA", "C", "A", 1, [1, 0, 0]), ("C", "C", "A", 1, [2, 0, 0])])
        charges = [0.1414, -0.0597, -0.3821]
        radii = [1.824, 1.908, 1.9]
        M.pdb2pqr = lambda: pd.DataFrame({"charge": charges, "radius": radii})
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "out.pqr")
            M.write_pqr(fname, index=[1, 2])
            lines = [l for l in open(fname) if l.startswith("ATOM")]
            M2 = bb.Molecule()
            M2.import_pqr(fname)

        # whitespace-separated fields, as read by APBS and PDB2PQR
        self.assertEqual([len(l.split()) for l in lines], [12, 12])
        self.assertEqual([float(l.split()[9]) for l in lines], charges[1:])
        self.assertEqual([float(l.split()[10]) for l in lines], radii[1:])
        np.testing.assert_allclose(M2.data["charge"].values, charges[1:])
        np.testing.assert_allclose(M2.data["radius"].values, radii[1:])

    def test_pdb_altloc_icode(self):

        print("\n> testing alternate locations, insertion codes and residue numbers in pdb files")
        import tempfile
        lines = ["ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 10.00           N\n",
                 "ATOM      2  CA AALA A  10       1.000   0.000   0.000  0.60 10.00           C\n",
                 "ATOM      3  CA BALA A  10       1.100   0.000   0.000  0.40 10.00           C\n",
                 "ATOM      4  CA  GLY A  52       3.000   0.000   0.000  1.00 10.00           C\n",
                 "ATOM      5  CA  SER A  52A      4.000   0.000   0.000  1.00 10.00           C\n",
                 "ATOM      6  CA  THR A-100       5.000   0.000   0.000  1.00 10.00           C\n",
                 "ATOM      7  CA  VAL BA000       6.000   0.000   0.000  1.00 10.00           C\n",
                 "END\n"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "std.pdb")
            with open(fname, "w") as f:
                f.writelines(lines)
            M = bb.Molecule()
            M.import_pdb(fname)

            self.assertEqual(list(M.data["name"]), ["N", "CA", "CA", "CA", "CA", "CA", "CA"])
            self.assertEqual(list(M.data["altloc"]), ["", "A", "B", "", "", "", ""])
            self.assertEqual(list(M.data["icode"]), ["", "", "", "", "A", "", ""])
            self.assertEqual(list(M.data["chain"]), ["A"] * 6 + ["B"])
            self.assertEqual(list(M.data["resid"]), [10, 10, 10, 52, 52, -100, 10000])

            # both alternate locations are kept, and a residue number selects all its insertion codes
            self.assertEqual(len(M.atomselect("A", 10, "CA")), 2)
            np.testing.assert_array_equal(M.atomselect("A", 52, "*", get_index=True)[1], [3, 4])
            np.testing.assert_array_equal(M.atomselect("A", "52A", "*", get_index=True)[1], [4])
            np.testing.assert_array_equal(M.same_residue(4, get_index=True)[1], [4])
            np.testing.assert_array_equal(M.same_residue_unique(3, get_index=True)[1], [3])
            self.assertEqual(M.get_fasta(), "AGST/V")

            # writing restores every column, and reading it back gives the same data
            out = os.path.join(tmp, "out.pdb")
            M.write_pdb(out)
            written = [l for l in open(out) if l.startswith("ATOM")]
            self.assertEqual([l[12:27] for l in written[1:3]], [" CA AALA A  10 ", " CA BALA A  10 "])
            self.assertEqual(written[4][12:27], " CA  SER A  52A")
            M2 = bb.Molecule()
            M2.import_pdb(out)
            for col in ["name", "altloc", "icode", "chain", "occupancy"]:
                self.assertEqual(list(M2.data[col]), list(M.data[col]), col)
            # residue 10000 does not fit 4 columns and keeps its last 4 digits
            self.assertEqual(list(M2.data["resid"]), [10, 10, 10, 52, 52, -100, 0])

        # columns survive merging molecules
        both = M + M
        self.assertEqual(list(both.data["altloc"]), list(M.data["altloc"]) * 2)
        self.assertEqual(list(both.data["icode"]), list(M.data["icode"]) * 2)

    def test_write_pdb_limits(self):

        print("\n> testing values the pdb format cannot hold")
        import tempfile
        M = self._molecule_from_atoms([("CA", "C", "A", 1, [0, 0, 0])])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.pdb")
            for xyz in [[-1000.0, 0, 0], [0, 0, 10000.0]]:
                M.coordinates[0, 0] = xyz
                with self.assertRaises(Exception):
                    M.write_pdb(out)
                self.assertFalse(os.path.exists(out))

            M.coordinates[0, 0] = [0, 0, 0]
            M.data["chain"] = "AB"
            with self.assertRaises(Exception):
                M.write_pdb(out)

            # residue numbers that do not fit keep their last 4 digits
            M.data["chain"] = "A"
            M.data["resid"] = 12345
            M.write_pdb(out)
            line = [l for l in open(out) if l.startswith("ATOM")][0]
            self.assertEqual(line[22:26], "2345")

    def test_element_from_atom_name(self):

        print("\n> testing element assignment when the element column is blank")
        # HSP.pdb has no element column, so elements and radii come from the atom names
        CA = self.M.atomselect("*", "*", "CA", get_index=True)[1]
        N = self.M.atomselect("*", "*", "N", get_index=True)[1]
        self.assertTrue(np.all(self.M.data["atomtype"].values[CA] == "C"))
        np.testing.assert_allclose(self.M.data["radius"].values[CA], 1.70)
        np.testing.assert_allclose(self.M.data["radius"].values[N], 1.55)

        # a right-justified one-letter element differs from a two-letter element in column 13
        names = {" CA ": "C", "CA  ": "CA", "FE  ": "FE", " OXT": "O", "OXT ": "O",
                 "HE21": "H", "1HD1": "H", "HB2 ": "H", "ZN  ": "ZN", " QQ ": ""}
        for name, element in names.items():
            self.assertEqual(self.M._guess_element(name), element, name)

        # an element column, when present, takes precedence over the atom name
        import tempfile
        lines = ["HETATM    1 CA    CA A   1       0.000   0.000   0.000  1.00 10.00          CA\n",
                 "HETATM    2 CA    CA A   2       5.000   0.000   0.000  1.00 10.00            \n",
                 "ATOM      3  CA  ALA A   3      10.000   0.000   0.000  1.00 10.00           C\n", "END\n"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "calcium.pdb")
            with open(fname, "w") as f:
                f.writelines(lines)
            M = bb.Molecule()
            M.import_pdb(fname, include_hetatm=True)
        self.assertEqual(list(M.data["atomtype"]), ["CA", "CA", "C"])
        np.testing.assert_allclose(M.data["radius"].values, [2.31, 2.31, 1.70])

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

    def test_principal_axes(self):

        print("\n> testing principal axes and alignment on them")
        # the axes must be real (NumPy 2.5 made linalg.eig always return complex arrays) and
        # orthonormal, and after align_axes they must lie on x, y and z, signs included
        from copy import deepcopy
        P = deepcopy(self.M)
        axes = P.get_principal_axes()
        self.assertFalse(np.iscomplexobj(axes))
        np.testing.assert_allclose(np.dot(axes, axes.T), np.eye(3), atol=1e-10)

        P.align_axes()
        np.testing.assert_allclose(P.get_principal_axes(), np.eye(3), atol=1e-6)

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


    def test_convex_formulas(self):

        print("\n> testing convex shape volumes, surfaces and CCS")
        from scipy.spatial import ConvexHull

        # sphere: the envelope of radius r, so an unsqueezed sphere has sphericity 1
        S = bb.Sphere(10, radius=1.9)
        self.assertAlmostEqual(S.get_volume(), 4 * np.pi * 10**3 / 3, places=6)
        self.assertAlmostEqual(S.get_surface(), 4 * np.pi * 10**2, places=6)
        self.assertAlmostEqual(S.get_sphericity(), 1.0, places=6)
        inside = S.check_inclusion(np.array([[0, 0, 0], [9.9, 0, 0], [10.1, 0, 0], [15, 0, 0]]) + S.get_center())
        self.assertEqual(list(inside), [True, True, False, False])
        S.squeeze(2.0)
        self.assertAlmostEqual(S.get_volume(), 4 * np.pi * 10**3 / 3, places=6)
        self.assertLess(S.get_sphericity(), 1.0)

        # prism: two bases plus the sides, and points on the same radius as the volume
        P = bb.Prism(10, 20, 6, radius=1.1)
        r, h, n = P.properties["r"], P.properties["h"], P.properties["n"]
        side = 2 * r * np.sin(np.pi / n)
        base = n * side * (r * np.cos(np.pi / n)) / 2
        self.assertAlmostEqual(P.get_surface(), 2 * base + n * side * h, places=6)
        self.assertAlmostEqual(P.get_volume(), base * h, places=6)
        self.assertAlmostEqual(np.max(np.linalg.norm(P.points[:, :2], axis=1)), r, places=6)
        hull = ConvexHull(P.points)
        self.assertAlmostEqual(hull.volume / P.get_volume(), 1.0, delta=0.02)

        # cylinder: CCS inflates every face by the gas radius
        C = bb.Cylinder(10, 20, radius=1.1)
        r1, h = C.properties["r1"], C.properties["h"]
        self.assertAlmostEqual(C.ccs(gas=1), (2 * np.pi * (r1 + 1)**2 + 2 * np.pi * (r1 + 1) * (h + 2)) / 4, places=6)

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
