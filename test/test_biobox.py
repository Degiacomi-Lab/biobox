import unittest
import sys, os
import numpy as np
import pandas as pd
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

    def _write_mrc(self, fname, data, mode=2, voxel=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), nstart=(0, 0, 0), label=b""):
        # MRC2014 file of a map given in x, y, z order (columns, rows, sections)
        nx, ny, nz = data.shape
        dtype = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}[mode]
        ints = np.zeros(256, dtype="<i4")
        floats = ints.view("<f4")
        ints[0:3] = (nx, ny, nz)
        ints[3] = mode
        ints[4:7] = nstart
        ints[7:10] = (nx, ny, nz)
        floats[10:13] = (voxel[0] * nx, voxel[1] * ny, voxel[2] * nz)
        floats[13:16] = (90.0, 90.0, 90.0)
        ints[16:19] = (1, 2, 3)
        floats[19:22] = (data.min(), data.max(), data.mean())
        floats[49:52] = origin
        header = bytearray(ints.tobytes())
        header[52 * 4:53 * 4] = b"MAP "
        header[53 * 4:54 * 4] = bytes([0x44, 0x44, 0, 0])
        if label:
            ints2 = np.frombuffer(bytes(header), dtype="<i4").copy()
            ints2[55] = 1
            header = bytearray(ints2.tobytes())
            header[224:224 + len(label)] = label
        with open(fname, "wb") as f:
            f.write(bytes(header))
            f.write(np.ascontiguousarray(data.transpose(2, 1, 0)).astype(dtype).tobytes())

    def test_mrc_reading(self):

        print("\n> density: MRC axes, voxel size, origin and data types")
        import tempfile
        data = np.zeros((7, 5, 3), dtype=np.float32)
        data[4, 1, 2] = 100.0
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "map.mrc")

            # the ORIGIN field of MRC2000/2014 files
            self._write_mrc(fname, data, voxel=(2, 3, 4), origin=(10, -20, 30))
            D = bb.Density()
            D.import_map(fname, "mrc")
            self.assertEqual(D.properties["density"].shape, (7, 5, 3))
            np.testing.assert_array_equal(np.argwhere(D.properties["density"] == 100), [[4, 1, 2]])
            # values derived from float32 header fields, compared to well below a picometre
            np.testing.assert_allclose(np.diag(D.properties["delta"]), [2, 3, 4], atol=1e-5)
            np.testing.assert_allclose(D.properties["origin"], [10, -20, 30], atol=1e-5)
            self.assertEqual(D.properties["format"], "mrc")

            # nstart, when the ORIGIN field is empty
            self._write_mrc(fname, data, voxel=(2, 3, 4), nstart=(1, 2, 3))
            D.import_map(fname, "mrc")
            np.testing.assert_allclose(D.properties["origin"], [2, 6, 12], atol=1e-5)

            # mode 0 is signed, mode 6 unsigned 16-bit, single sections are kept
            signed = np.full((3, 3, 3), -5)
            self._write_mrc(fname, signed, mode=0)
            D.import_map(fname, "mrc")
            np.testing.assert_allclose(D.properties["density"], -5)
            self._write_mrc(fname, data * 600, mode=6)
            D.import_map(fname, "mrc")
            self.assertAlmostEqual(D.properties["density"][4, 1, 2], 60000)
            self._write_mrc(fname, data[:, :, :1], voxel=(2, 3, 4))
            D.import_map(fname, "mrc")
            self.assertEqual(D.properties["density"].shape, (7, 5, 1))

            # a Chimera rotation label does not prevent loading
            self._write_mrc(fname, data, label=b"Chimera rotation: 0 0 1 90")
            D.import_map(fname, "mrc")
            self.assertEqual(D.properties["density"].shape, (7, 5, 3))

            # index to coordinates, and back
            import biobox.classes.density_MRC as MRC
            self._write_mrc(fname, data, voxel=(2, 3, 4), nstart=(1, 2, 3))
            G = MRC.MRC_Grid(fname, "mrc")
            np.testing.assert_allclose(G.ijk_to_xyz((4, 1, 2)), (10, 9, 20), atol=1e-5)
            np.testing.assert_allclose(G.xyz_to_ijk((10, 9, 20)), (4, 1, 2), atol=1e-5)

            # load failures are reported
            with open(fname, "wb") as f:
                f.write(b"not a map")
            with self.assertRaises(Exception):
                bb.Density().import_map(fname, "mrc")

    def test_density_points_placement(self):

        print("\n> density: points arrangement shrunk by the sphere radius")
        origin = np.array([1.0, 2.0, 3.0])
        for delta, corner in [(1.0, 20), (2.0, 20), (2.0, 40)]:
            data = np.zeros((60, 60, 60))
            data[corner:corner + 4, corner:corner + 4, corner:corner + 4] = 1.0
            D = bb.Density()
            D.import_numpy(data, origin=origin, delta=np.identity(3) * delta)
            D.place_points(sigma=0.5, noise_filter=0)
            r = D.properties["radius"]
            if delta == 1:
                scale = 1.0
            else:
                scale = (delta - 1) / (3 * delta - 3) * (3 * delta - r)
            np.testing.assert_allclose(D.points.min(axis=0), origin + corner * scale + r)
            np.testing.assert_allclose(D.points.max(axis=0), origin + (corner + 3) * scale + r)

        # a single voxel has no extent to shrink, and sits at its voxel shifted by the radius
        data = np.zeros((10, 10, 10))
        data[5, 5, 5] = 1.0
        D = bb.Density()
        D.import_numpy(data, delta=np.identity(3) * 2)
        D.place_points(sigma=0.5, noise_filter=0)
        np.testing.assert_allclose(D.points, [np.ones(3) * (10 + D.properties["radius"])])

        # a blob holding 3% of the points survives a noise filter of 1% but not one of 5%
        data = np.zeros((40, 40, 40))
        data[2:22, 2:22, 2:10] = 1.0
        data[30:35, 30:35, 30:35] = 1.0
        D = bb.Density()
        D.import_numpy(data)
        D.place_points(sigma=0.5, noise_filter=0.01)
        self.assertEqual(len(D.points), 3200 + 125)
        D.place_points(sigma=0.5, noise_filter=0.05)
        self.assertEqual(len(D.points), 3200)

    def test_density_scan(self):

        print("\n> density: thresholds, volumes and appended scans")
        data = np.zeros((20, 20, 20))
        data[5:15, 5:15, 5:15] = 10.0
        D = bb.Density()
        D.import_numpy(data)
        sigma = D.get_sigma_from_thresh(5.0)

        # the threshold is given in sigma units, and the volume survives a failed CCS
        row = D.find_data_from_sigma(sigma, noise_filter=0, append=True)
        self.assertAlmostEqual(row[0], sigma)
        self.assertAlmostEqual(row[1], 1000.0)
        self.assertEqual(D.properties["scan"].shape, (1, 3))
        D.find_data_from_sigma(sigma, noise_filter=0, append=True)
        self.assertEqual(D.properties["scan"].shape, (2, 3))
        D.threshold_vol_ccs(low=sigma, high=sigma, sampling_points=1, noise_filter=0, append=True)
        self.assertEqual(D.properties["scan"].shape, (3, 3))
        np.testing.assert_allclose(D.properties["scan"][:, 1], 1000.0)

        # a threshold above the maximum gives an empty map
        self.assertEqual(list(D.find_data_from_sigma(D.get_sigma_from_thresh(20.0))[1:]), [0.0, 0.0])

    def test_density_blur_and_dx(self):

        print("\n> density: isotropic blur, and dx files")
        import tempfile
        data = np.zeros((9, 9, 9))
        data[4, 4, 4] = 1.0
        D = bb.Density()
        D.import_numpy(data)
        D.blur(dimension=5, sigma=0.8)
        d = D.properties["density"]
        np.testing.assert_allclose(d[2:7, 4, 4], d[4, 2:7, 4])
        np.testing.assert_allclose(d[2:7, 4, 4], d[4, 4, 2:7])
        self.assertGreater(d[4, 4, 4], d[4, 4, 5])

        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "map.dx")
            D.write_dx(fname)
            text = open(fname).read().replace("data follows\n", "data follows\n\n")
            with open(fname, "w") as f:
                f.write(text)
            E = bb.Density()
            E.import_map(fname, "dx")
            np.testing.assert_allclose(E.properties["density"], d)


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

        # only the current conformation is copied
        M.set_current(1)
        B = M.apply_biomatrix()
        xyz = M.coordinates[1, :3]
        expected = np.concatenate([xyz, np.dot(xyz, R.T) + [10, 0, 0]])
        self.assertEqual(B.coordinates.shape, (1, 6, 3))
        np.testing.assert_allclose(B.points, expected, atol=1e-6)
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
        np.testing.assert_allclose(B2.points, xyz + [[0, 0, 0], [0, 0, 0], [0, 0, 5]])

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

        # a selected chain gives the same potential as the chain alone, on the same grid
        M = self._molecule_from_atoms(atoms)
        M.data["charge"] = [1.0, 0.0, -1.0]
        B = M.get_subset(M.atomselect("B", "*", "*", get_index=True)[1])
        for selected, alone in zip(M.get_electrostatics(chain="B", clear_mass=False), B.get_electrostatics(clear_mass=False)):
            np.testing.assert_allclose(selected.properties["density"], alone.properties["density"])
            np.testing.assert_allclose(selected.properties["origin"], alone.properties["origin"])
        np.testing.assert_allclose(M.get_electrostatics(chain="B")[1].properties["origin"], [17 - 3, -3, -3])

        # the mass mask also holds atoms of other chains, including those just outside the grid
        for xa in [15.0, 13.0]:
            near = self._molecule_from_atoms([("CA", "C", "A", 1, [xa, 0, 0]), ("CA", "C", "B", 1, [17, 0, 0])])
            far = self._molecule_from_atoms([("CA", "C", "A", 1, [50, 0, 0]), ("CA", "C", "B", 1, [17, 0, 0])])
            mass_near = near.get_electrostatics(chain="B")[2]
            mass_far = far.get_electrostatics(chain="B")[2]
            np.testing.assert_allclose(mass_near.properties["origin"], mass_far.properties["origin"])
            self.assertGreater(mass_near.properties["density"].sum(), mass_far.properties["density"].sum())

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

    def test_pdb_two_character_chains(self):

        print("\n> testing two-character chain names in pdb files")
        import tempfile
        M = self._molecule_from_atoms([("CA", "C", "A", 1, [0, 0, 0]), ("CA", "C", "B", 2, [1, 0, 0]), ("CA", "C", "C", 1000, [2, 0, 0])])
        M.data["chain"] = ["A", "AB", "AC"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "chains.pdb")
            M.write_pdb(fname)
            lines = [l for l in open(fname) if l.startswith("ATOM")]
            # column 22 holds the first character, columns 73-76 the full name, and the residue number keeps columns 23-26
            self.assertEqual([l[21] for l in lines], ["A", "A", "A"])
            self.assertEqual([l[72:76] for l in lines], ["    ", "AB  ", "AC  "])
            self.assertEqual([l[22:26] for l in lines], ["   1", "   2", "1000"])
            R = bb.Molecule()
            R.import_pdb(fname)
            self.assertEqual(list(R.data["chain"]), ["A", "AB", "AC"])
            self.assertEqual(list(R.data["resid"]), [1, 2, 1000])

            # segment identifiers that do not extend the chain are not chain names
            fname = os.path.join(tmp, "segid.pdb")
            with open(fname, "w") as f:
                for j, (chain, segid) in enumerate([("A", "PROA"), ("A", "B1"), ("B", "B")]):
                    f.write("ATOM  %5d  CA  ALA %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s   C\n" % (j+1, chain, 1, j, 0, 0, 1, 0, segid))
            R = bb.Molecule()
            R.import_pdb(fname)
            self.assertEqual(list(R.data["chain"]), ["A", "A", "B"])

            M.data["chain"] = ["A", "ABC", "AC"]
            with self.assertRaises(Exception):
                M.write_pdb(os.path.join(tmp, "long.pdb"))

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
            M.data["chain"] = "ABC"
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


    def _wall_path(self, step=1.0):
        # a 13 x 13 sheet of points at x = 0, in a global grid spanning -14 to 14 A
        from biobox.measures.path import Path
        ys = np.arange(-6, 6.01, 1.0)
        wall = np.array([[0, y, z] for y in ys for z in ys], float)
        P = Path(wall)
        P.setup_global_search(step=step, maxdist=60, use_hull=False, boundaries=[[-14, 14]] * 3)
        return P

    def _path_clear(self, P, wp):
        # every segment of a path stays in accessible space, except next to its two ends
        exempt = P._target_exemptions(wp[0], wp[-1])
        return all(P._segment_clear(a, b, exempt) for a, b in zip(wp[:-1], wp[1:]))

    def test_line_of_sight(self):

        print("\n> testing line of sight along every dominant axis")
        import biobox.lib.fastmath as FM
        for axis in range(3):
            grid = np.ones((20, 20, 20), dtype=bool)
            a = np.array([5, 5, 5])
            b = a.copy()
            b[axis] += 10
            b[(axis + 1) % 3] += 1
            on = a.copy()
            on[axis] += 2
            off = on.copy()
            off[(axis + 1) % 3] += 1
            grid[tuple(on)] = False
            self.assertFalse(FM.c_line_of_sight(grid, a, b))
            grid[tuple(on)] = True
            grid[tuple(off)] = False
            self.assertTrue(FM.c_line_of_sight(grid, a, b))
            np.testing.assert_array_equal(a, [5, 5, 5])

    def test_path_grid(self):

        print("\n> testing path grid coordinates, costs and extent")
        from biobox.lib.graph import Graph
        P = self._wall_path()
        g = P.graph
        np.testing.assert_allclose(g.get_points_from_idx(np.array([0., 0., 0.])), [-14, -14, -14])
        np.testing.assert_allclose(g.get_points_from_idx(np.array([14., 14., 14.])), [0, 0, 0])
        self.assertFalse(g.is_accessible(np.array([[0., 0., 0.]]))[0])
        self.assertTrue(g.is_accessible(np.array([[-8., 0., 0.]]))[0])

        # euclidean step costs and heuristic, in grid steps
        flat = lambda i: int(g.get_flat_index(np.array(i)))
        self.assertAlmostEqual(g.cost(flat([1, 1, 1]), flat([2, 2, 2])), np.sqrt(3))
        self.assertAlmostEqual(g.heuristic(np.array(flat([1, 1, 1])), np.array(flat([4, 5, 1]))), 5.0)

        # the default grid encloses a lopsided cloud
        rng = np.random.default_rng(0)
        cloud = np.vstack([rng.normal(0, 2, (200, 3)), [[20, 0, 0]] * 4])
        G = Graph(cloud)
        G.make_grid(step=1.0)
        lo = G.get_points_from_idx(np.array([0., 0., 0.]))
        hi = G.get_points_from_idx(np.array(G.access_grid_shape - 1, dtype=float)) if G.access_grid_shape is not None else None
        G.make_global_grid(step=1.0)
        hi = G.get_points_from_idx(np.array(G.access_grid_shape - 1, dtype=float))
        self.assertTrue(np.all(cloud.min(axis=0) >= lo) and np.all(cloud.max(axis=0) <= hi))

    def test_astar_optimal(self):

        print("\n> testing that A* returns the shortest path on the grid")
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        P = self._wall_path()
        g = P.graph
        acc = g.access_grid
        shape = acc.shape
        rows, cols, w = [], [], []
        idx = np.array(np.where(acc)).T
        for d in [np.array(v) for v in np.ndindex(3, 3, 3)]:
            d = d - 1
            if not np.any(d):
                continue
            nb = idx + d
            ok = np.all((nb >= 0) & (nb < shape), axis=1)
            ok[ok] = acc[tuple(nb[ok].T)]
            rows.extend(np.ravel_multi_index(tuple(idx[ok].T), shape))
            cols.extend(np.ravel_multi_index(tuple(nb[ok].T), shape))
            w.extend([np.linalg.norm(d)] * int(ok.sum()))
        n = int(np.prod(shape))
        A = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()

        for s, e in [([-5, 0, 0], [5, 0, 0]), ([-6, 2, 3], [7, -1, -2])]:
            s = np.array(s, float)
            e = np.array(e, float)
            si = int(np.ravel_multi_index(tuple(g.get_idx_from_points(np.array([s]))[0]), shape))
            ei = int(np.ravel_multi_index(tuple(g.get_idx_from_points(np.array([e]))[0]), shape))
            d, wp = P.search_path(s, e, method="astar", get_path=False, test_los=False)
            nodes = wp[1:-1]
            chain = np.sum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))
            self.assertAlmostEqual(chain, dijkstra(A, indices=si)[ei], places=6)

    def test_path_search(self):

        print("\n> testing shortest paths around and through obstacles")
        P = self._wall_path()

        # paths around the wall stay in accessible space, before and after smoothing, and are
        # never shorter than the shortest way around its edge
        bound = 2 * np.sqrt(5**2 + 6**2)
        for method in ["theta", "astar", "old_theta"]:
            for los in [False, True]:
                d, wp = P.search_path(np.array([-5., 0, 0]), np.array([5., 0, 0]), method=method, get_path=False, test_los=los)
                self.assertTrue(self._path_clear(P, wp), (method, los))
                self.assertGreaterEqual(d, bound)
                ds, chain = P.smooth(P._get_trails(wp))
                self.assertTrue(self._path_clear(P, chain), (method, los))
                self.assertGreaterEqual(ds, bound)

        # points seeing each other are joined by a straight line
        rng = np.random.default_rng(0)
        for k in range(20):
            s = np.array([-8., 0, 0]) + rng.uniform(-3, 3, 3)
            e = np.array([-8., 0, 0]) + rng.uniform(-3, 3, 3)
            d, wp = P.search_path(s, e, get_path=False)
            self.assertAlmostEqual(d, np.linalg.norm(s - e))

        # a point inside a sealed shell cannot be reached
        from biobox.measures.path import Path
        n = int(4 * np.pi * 49 / 0.25)
        k = np.arange(n)
        y = k * 2.0 / n - 1 + 1.0 / n
        r = np.sqrt(1 - y * y)
        shell = 7 * np.column_stack((np.cos(k * np.pi * (3 - np.sqrt(5))) * r, y, np.sin(k * np.pi * (3 - np.sqrt(5))) * r))
        Q = Path(shell)
        Q.setup_global_search(step=1.0, maxdist=60, use_hull=False, boundaries=[[-14, 14]] * 3)
        for method in ["theta", "astar"]:
            d, wp = Q.search_path(np.array([11., 0, 0]), np.array([0.3, 0.2, 0.1]), method=method, get_path=False)
            self.assertEqual(d, -1)
            self.assertEqual(len(wp), 0)

        # a local grid without obstacles is fully accessible
        Q = Path(np.array([[100., 100, 100], [101, 100, 100]]))
        Q.setup_local_search(step=1.0, maxdist=28)
        self.assertAlmostEqual(Q.search_path(np.array([0., 0, 0]), np.array([5., 0, 0]))[0], 5.0)

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

    def test_SASA_threshold(self):

        print("\n> testing that the SASA threshold only selects surface atoms")
        # atoms exposed below the threshold still contribute to the area and the mesh
        idx = range(200)
        asa0, mesh0, surf0 = bb.sasa(self.M, targets=idx, n_sphere_point=200, threshold=0)
        asa1, mesh1, surf1 = bb.sasa(self.M, targets=idx, n_sphere_point=200, threshold=0.05)
        self.assertEqual(asa0, asa1)
        self.assertEqual(mesh0.shape, mesh1.shape)
        self.assertLess(len(surf1), len(surf0))
        self.assertTrue(set(surf1) <= set(surf0))

    def test_SASA_edge_cases(self):

        print("\n> testing SASA of an empty structure and of missing radii")
        S = bb.Structure(p=np.zeros((0, 3)))
        S.data['radius'] = np.zeros(0)
        asa, mesh, surf = bb.sasa(S)
        self.assertEqual(asa, 0.0)
        self.assertEqual(mesh.shape, (0, 3))
        self.assertEqual(len(surf), 0)

        S = bb.Structure(p=np.array([[0., 0, 0], [2, 0, 0]]))
        S.data['radius'] = np.array([1.5, np.nan])
        with self.assertRaises(ValueError):
            bb.sasa(S)

    def test_saxs_files(self):

        print("\n> testing that saxs leaves the input file and removes its own")
        import tempfile, shutil
        from unittest import mock
        import biobox.measures.calculators as C

        calls = []
        def fake_crysol(cmd, **kwargs):
            # crysol writes its output in the working directory, named after the input file
            calls.append(cmd)
            base = os.path.splitext(os.path.basename(cmd[-1]))[0]
            np.savetxt(base + "00.int", np.array([[0.01, 1.0, 0], [0.02, 0.9, 0]]), header="crysol")
            for ext in ["00.alm", "00.log"]:
                open(base + ext, "w").close()

        here = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            self.M.write_pdb("myprotein.pdb")
            with mock.patch.object(C.subprocess, "check_call", side_effect=fake_crysol):
                curve = C.saxs(self.M, crysol_path="atsas", pdbname="myprotein.pdb")
                self.assertEqual(sorted(os.listdir(tmp)), ["myprotein.pdb"])
                C.saxs(self.M, crysol_path="atsas")
                self.assertEqual(sorted(os.listdir(tmp)), ["myprotein.pdb"])
        finally:
            os.chdir(here)
            shutil.rmtree(tmp)

        self.assertEqual(curve.shape, (2, 2))
        # no shell redirection, which Windows cmd cannot resolve
        self.assertEqual(calls[0], [os.path.join("atsas", "crysol"), "-lm", "20", "-ns", "500", "myprotein.pdb"])

    def test_ccs_library_name(self):

        print("\n> testing the IMPACT library name on each platform")
        from unittest import mock
        import biobox.measures.calculators as C

        seen = []
        def fake_ccs(libfile):
            seen.append(libfile)
            raise RuntimeError("stop")

        with mock.patch.object(C, "CCS", side_effect=fake_ccs):
            for platform in ["darwin", "linux", "win32"]:
                with mock.patch.object(C.sys, "platform", platform):
                    with self.assertRaises(Exception):
                        C.ccs(self.M, impact_path="impact")
        self.assertEqual([os.path.basename(f) for f in seen], ["libimpact.so", "libimpact.so", "libimpact.dll"])

    def test_dipole_density_centred(self):

        print("\n> testing that the dipole density is centred on the fluctuating voxel")
        from unittest import mock
        import biobox.lib.e_density as E
        from biobox.classes.density import Density

        nx, c = 15, 7
        dm = np.zeros((2, nx, nx, nx, 3), np.float32)
        dm[0, c, c, c] = [1, 0, 0]
        dm[1, c, c, c] = [-1, 0, 0]
        orig = np.array([np.arange(nx) * 1.0] * 3)

        captured = []
        def capture(self, fname):
            captured.append(self.properties['density'].copy())

        with mock.patch.object(Density, "write_dx", capture):
            for vox in [3.0, 4.0]:
                E.c_get_dipole_density(dm, orig, [0., 0., 0.], 5e-27, "x.dx", vox_in_window=vox)

        # odd and even windows give kernels of 3 and 5 points per axis, centred on the voxel
        for d, width in zip(captured, [3, 5]):
            self.assertEqual(np.unravel_index(np.argmax(d), d.shape), (c, c, c))
            com = [np.sum(d * np.indices(d.shape)[k]) / d.sum() for k in range(3)]
            np.testing.assert_allclose(com, [c, c, c], atol=1e-6)
            self.assertEqual(np.count_nonzero(d[:, c, c]), width)

    def test_predict_without_scan(self):

        print("\n> testing that CCS and mass prediction ask for a scan when none is stored")
        D = bb.Density()
        D.import_numpy(np.arange(27.).reshape(3, 3, 3))
        with self.assertRaisesRegex(IOError, "threshold_vol_ccs"):
            D.predict_ccs_from_mass(10.0, 100.0)
        with self.assertRaisesRegex(IOError, "threshold_vol_ccs"):
            D.predict_mass_from_ccs(10.0, 1000.0)

    def test_best_threshold_places_best(self):

        print("\n> testing that best_threshold places points at the threshold with the smallest mass error")
        from unittest import mock
        from biobox.classes.density import Density

        D = bb.Density()
        D.import_numpy(np.arange(27.).reshape(3, 3, 3))

        # volume as a step function of the threshold, so that bisection ends on a repeated error
        placed = []
        def fake_place(self, sigma=0, noise_filter=0.01):
            placed.append(sigma)
        def fake_volume(self):
            return float(np.floor(10 * (2.0 - placed[-1])))

        with mock.patch.object(Density, "place_points", fake_place), mock.patch.object(Density, "get_volume", fake_volume):
            r = D.best_threshold(7.3, density=1.0)

        best = r[np.argmin(np.abs(r[:, 1]))]
        self.assertNotAlmostEqual(abs(r[-1, 1]), abs(best[1]))
        self.assertEqual(placed[-1], best[0])
        self.assertAlmostEqual(best[1], -0.3)

    def test_dipole_density_eqn(self):

        print("\n> testing that the dipole density rejects unknown functions")
        from unittest import mock
        import biobox.lib.e_density as E
        from biobox.classes.density import Density

        dm = np.zeros((2, 5, 5, 5, 3), np.float32)
        dm[0, 2, 2, 2] = [1, 0, 0]
        dm[1, 2, 2, 2] = [-1, 0, 0]
        orig = np.array([np.arange(5) * 1.0] * 3)

        written = []
        with mock.patch.object(Density, "write_dx", lambda self, fname: written.append(fname)):
            with self.assertRaisesRegex(ValueError, "gauss.*slater"):
                E.c_get_dipole_density(dm, orig, [0., 0., 0.], 5e-27, "x.dx", eqn="lorentz")
            for eqn in ["gauss", "slater"]:
                E.c_get_dipole_density(dm, orig, [0., 0., 0.], 5e-27, "x.dx", eqn=eqn)
        self.assertEqual(len(written), 2)

    def test_dipole_map_cones(self):

        print("\n> testing that dipole map cones are selected by the dipole magnitude")
        import tempfile, shutil
        import biobox.lib.e_density as E

        # a dipole of 0.9 e*A along z, seen by the 27 voxels whose window contains both charges
        crd = np.array([[[2., 2., 2.45], [2., 2., 1.55]]] * 2)
        charges = np.array([1., -1.])
        orig = np.array([np.arange(5) * 1.0] * 3)

        tmp = tempfile.mkdtemp()
        try:
            fname = os.path.join(tmp, "dipole_map.tcl")
            dm = E.c_get_dipole_map(crd, orig, charges, 0, 2, 1.0, 3, True, fname)
            with open(fname) as f:
                cones = [l for l in f if l.startswith("draw cone")]
        finally:
            shutil.rmtree(tmp)

        magnitude = np.linalg.norm(np.mean(dm, axis=0), axis=3)
        self.assertEqual(np.count_nonzero(magnitude > 0.7), 27)
        self.assertEqual(len(cones), 27)
        # cones point along z
        for l in cones:
            w = l.split()
            self.assertEqual(w[3:5], w[8:10])
            self.assertAlmostEqual(float(w[10]) - float(w[5]), 0.9, places=5)

    def test_dipole_density_no_fluctuation(self):

        print("\n> testing that a dipole map without fluctuations gives an error and no file")
        from unittest import mock
        import biobox.lib.e_density as E
        from biobox.classes.density import Density

        dm = np.zeros((2, 5, 5, 5, 3), np.float32)
        dm[:, 2, 2, 2] = [1, 0, 0]
        orig = np.array([np.arange(5) * 1.0] * 3)

        written = []
        with mock.patch.object(Density, "write_dx", lambda self, fname: written.append(fname)):
            with self.assertRaisesRegex(ValueError, "fluctuat"):
                E.c_get_dipole_density(dm, orig, [0., 0., 0.], 5e-27, "x.dx")
        self.assertEqual(written, [])

    def test_dipole_density_grid_edges(self):

        print("\n> testing that the dipole density clips functions at the grid edges")
        from unittest import mock
        import biobox.lib.e_density as E
        from biobox.classes.density import Density

        class LowMemory(np.ndarray):
            # the first ufunc call raises MemoryError, sending the calculation to its chunked path
            calls = 0
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                LowMemory.calls += 1
                if LowMemory.calls == 1:
                    raise MemoryError
                inputs = [np.asarray(i) for i in inputs]
                return getattr(ufunc, method)(*inputs, **kwargs)

        nx, c = 9, 4
        def fluctuating(ix, iy, iz):
            dm = np.zeros((2, nx, nx, nx, 3), np.float32)
            dm[0, ix, iy, iz] = [1, 0, 0]
            dm[1, ix, iy, iz] = [-1, 0, 0]
            return dm
        orig = np.array([np.arange(nx) * 1.0] * 3)

        captured = []
        def capture(self, fname):
            captured.append(self.properties['density'].copy())

        with mock.patch.object(Density, "write_dx", capture):
            for vox in [3.0, 5.0]:
                for low_memory in [False, True]:
                    captured.clear()
                    for idx in [(c, c, c), (0, 0, 0), (nx-1, 0, nx-1)]:
                        dm = fluctuating(*idx)
                        if low_memory:
                            LowMemory.calls = 0
                            dm = dm.view(LowMemory)
                        E.c_get_dipole_density(dm, orig, [0., 0., 0.], 5e-27, "x.dx", vox_in_window=vox)
                        if low_memory:
                            self.assertGreater(LowMemory.calls, 1)

                    centre, corner, edge = captured
                    half = int(vox) // 2
                    # each clipped function is the part of the centred function that lies inside the grid
                    np.testing.assert_allclose(corner[:half+1, :half+1, :half+1], centre[c:c+half+1, c:c+half+1, c:c+half+1])
                    np.testing.assert_allclose(edge[nx-1-half:, :half+1, nx-1-half:], centre[c-half:c+1, c:c+half+1, c-half:c+1])
                    self.assertEqual(np.count_nonzero(corner), (half+1)**3)
                    self.assertEqual(np.count_nonzero(edge), (half+1)**3)

    def test_ccs_executable_files(self):

        print("\n> testing that the IMPACT executable mode removes its params file and rejects mixed radii")
        import tempfile, shutil
        from unittest import mock
        import biobox.measures.calculators as C

        params = []
        def fake_impact(cmd, **kwargs):
            # impact writes its report to the file the command redirects to
            p = cmd.split("-param ")[1].split('"')[1]
            with open(p) as f:
                params.append((p, f.read()))
            with open(cmd.split(">")[-1].strip(), "w") as f:
                f.write("CCS PA (A^2): 1234.5 0 0 1300.0 0\n")

        points = np.array([[0., 0., 0.], [5., 0., 0.], [0., 5., 0.]])
        here = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            with mock.patch.object(C.subprocess, "check_call", side_effect=fake_impact) as call:
                v = C.ccs(bb.Structure(points, 2.0), use_lib=False, impact_path="impact")
                self.assertEqual(os.listdir(tmp), [])

                with self.assertRaisesRegex(ValueError, "use_lib=True"):
                    C.ccs(bb.Structure(points, [2.0, 2.0, 3.0]), use_lib=False, impact_path="impact")
                self.assertEqual(os.listdir(tmp), [])
                self.assertEqual(call.call_count, 1)

            failed = []
            def failing_impact(cmd, **kwargs):
                failed.append(cmd.split("-param ")[1].split('"')[1])
                raise OSError("impact failed")
            with mock.patch.object(C.subprocess, "check_call", side_effect=failing_impact):
                with self.assertRaises(Exception):
                    C.ccs(bb.Structure(points, 2.0), use_lib=False, impact_path="impact")
            self.assertFalse(os.path.exists(os.path.dirname(failed[0])))
        finally:
            os.chdir(here)
            shutil.rmtree(tmp)

        self.assertEqual(v, 1234.5)
        p, text = params[0]
        self.assertNotEqual(os.path.dirname(os.path.abspath(p)), os.path.abspath(tmp))
        self.assertFalse(os.path.exists(os.path.dirname(p)))
        self.assertTrue(text.endswith(" Z 3.0"))

    def test_mrc_cached_submatrix(self):

        print("\n> testing that MRC submatrices are read from a cached larger matrix")
        from biobox.classes.density_MRC import MRC_Grid, Data_Cache

        grid = MRC_Grid("EMD-1080.mrc", "mrc")
        grid.data_cache = Data_Cache(size=0)
        full = grid.matrix()

        for origin, size, step in [((1, 2, 3), (4, 5, 6), (1, 1, 1)), ((2, 4, 6), (9, 7, 5), (2, 2, 2))]:
            cached = grid.matrix(origin, size, step, from_cache_only=True)
            self.assertIsNotNone(cached)
            np.testing.assert_array_equal(cached, grid.read_matrix(origin, size, step, None))

        # clearing the cache removes every matrix of the grid
        sub = grid.matrix((0, 0, 0), (3, 3, 3), (2, 2, 2))
        grid.cache_data(sub, (0, 0, 0), (3, 3, 3), (2, 2, 2))
        self.assertEqual(len(grid.data_cache.group_keys_and_data(grid)), 2)
        grid.clear_cache()
        self.assertEqual(grid.data_cache.group_keys_and_data(grid), [])
        self.assertIsNotNone(full)

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

    def test_principal_axes_frame(self):

        print("\n> testing that principal axes are taken about the center and form a rotation")
        rng = np.random.default_rng(0)
        for k in range(50):
            pts = rng.normal(size=(40, 3)) * [5, 3, 1]
            S = bb.Structure(p=pts)
            axes = S.get_principal_axes()
            self.assertAlmostEqual(np.linalg.det(axes), 1.0, places=10)

            # the axes do not depend on where the structure sits
            S.translate(30, 40, -20)
            np.testing.assert_allclose(S.get_principal_axes(), axes, atol=1e-8)

            # and they are the eigenvectors of the covariance, from largest to smallest variance
            w, v = np.linalg.eigh(np.cov(pts.T))
            np.testing.assert_allclose(np.abs(np.sum(axes * v.T[::-1], axis=1)), np.ones(3), atol=1e-8)

    def test_center_copy(self):

        print("\n> testing that a returned center does not change afterwards")
        S = bb.Structure(p=np.array([[0., 0, 0], [2, 0, 0]]))
        c = S.get_center()
        S.translate(5, 0, 0)
        np.testing.assert_allclose(c, [1, 0, 0])
        np.testing.assert_allclose(S.get_center(), [6, 0, 0])
        np.testing.assert_allclose(S.properties["center"], [6, 0, 0])

    def test_transformations_current_frame(self):

        print("\n> testing that transformations move only the current conformation")
        from copy import deepcopy
        rng = np.random.default_rng(1)
        frame = rng.normal(size=(30, 3)) * [5, 3, 1] + [10, -4, 2]
        S = bb.Structure(p=frame)
        S.add_xyz(frame + [1.0, 2.0, 3.0])
        S.add_xyz(frame - [4.0, 0.0, 1.0])
        S.set_current(1)
        untouched = [0, 2]

        transformations = {
            "rotate": lambda T: T.rotate(90, 0, 0),
            "translate": lambda T: T.translate(5, -2, 1),
            "apply_transformation": lambda T: T.apply_transformation(np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])),
            "center_to_origin": lambda T: T.center_to_origin(),
            "align_axes": lambda T: T.align_axes(),
        }
        for name, transform in transformations.items():
            T = deepcopy(S)
            transform(T)
            np.testing.assert_allclose(T.coordinates[untouched], S.coordinates[untouched], err_msg=name)
            self.assertFalse(np.allclose(T.coordinates[1], S.coordinates[1]), name)
            np.testing.assert_allclose(T.points, T.coordinates[1], err_msg=name)
            np.testing.assert_allclose(T.properties["center"], T.coordinates[1].mean(axis=0), atol=1e-10, err_msg=name)

        R = deepcopy(S)
        R.rotate(90, 0, 0)
        np.testing.assert_allclose(R.coordinates[1][:, 0], S.coordinates[1][:, 0])
        np.testing.assert_allclose(R.coordinates[1][:, 1], -S.coordinates[1][:, 2], atol=1e-10)

        A = deepcopy(S)
        A.align_axes()
        np.testing.assert_allclose(A.get_principal_axes(), np.eye(3), atol=1e-6)

        # assemblies move the current conformation of their units (that of the loaded structure)
        P = bb.Assembly()
        P.load(S, 2)
        P.translate(1, 0, 0)
        for u in P.unit:
            np.testing.assert_allclose(u.coordinates[1], S.coordinates[1] + [1, 0, 0])
            np.testing.assert_allclose(u.coordinates[untouched], S.coordinates[untouched])

    def test_structure_construction(self):

        print("\n> testing empty structures, radii and added frames")
        # an empty structure has no frames and no points, before and after clear
        for S in [bb.Structure(), bb.Molecule()]:
            self.assertEqual(len(S), 0)
            self.assertEqual(len(S.coordinates), 0)
        S = bb.Structure(p=np.zeros((3, 3)))
        S.clear()
        self.assertEqual(len(S), 0)
        S.add_xyz(np.ones((2, 3)))
        self.assertEqual(S.coordinates.shape, (1, 2, 3))

        # a radius can be any scalar, or one value per point
        pts = np.arange(9.0).reshape(3, 3)
        np.testing.assert_allclose(bb.Structure(pts, r=np.float64(1.5)).data["radius"], [1.5] * 3)
        np.testing.assert_allclose(bb.Structure(pts, r=[1, 2, 3]).data["radius"], [1, 2, 3])
        with self.assertRaises(Exception):
            bb.Structure(pts, r=[1, 2])

        # the first added frame becomes current, wherever the pointer was
        S = bb.Structure(pts)
        S.add_xyz(pts + 1)
        S.add_xyz(pts + 2)
        S.set_current(0)
        S.add_xyz(pts + 3)
        self.assertEqual(S.current, 3)
        np.testing.assert_allclose(S.points, pts + 3)
        S.set_current(0)
        S.add_xyz(np.array([pts + 4, pts + 5]))
        self.assertEqual(S.current, 4)

    def test_structure_index_arrays(self):

        print("\n> testing selections given as numpy arrays")
        rng = np.random.default_rng(2)
        S = bb.Structure(rng.normal(size=(4, 10, 3)))
        idx = np.array([1, 3, 5])
        np.testing.assert_allclose(S.get_xyz(idx), S.points[idx])
        np.testing.assert_allclose(S.get_xyz([1, 3, 5]), S.points[idx])
        np.testing.assert_allclose(S.get_xyz(), S.points)
        np.testing.assert_allclose(S.rmsf(idx), S.rmsf()[idx])
        proj, pca = S.pca(2, idx)
        self.assertEqual(proj.shape, (4, 2))

    def test_rmsf(self):

        print("\n> testing RMSF against its definition")
        # the square root of the mean squared displacement from the mean position, over frames
        rng = np.random.default_rng(4)
        X = rng.normal(size=(7, 5, 3)) * 2
        expected = np.sqrt(np.mean(np.sum((X - X.mean(axis=0))**2, axis=2), axis=0))
        S = bb.Structure(X)
        np.testing.assert_allclose(S.rmsf(), expected)
        np.testing.assert_allclose(S.rmsf(np.array([0, 3])), expected[[0, 3]])

        from copy import deepcopy
        M = deepcopy(self.M)
        M.add_xyz(M.coordinates[0] + rng.normal(size=M.coordinates[0].shape))
        idx = np.array([0, 10, 20])
        np.testing.assert_allclose(M.beta_factor_from_rmsf(idx), 8 * np.pi**2 * M.rmsf(idx)**2 / 3)

    def test_structure_hull_and_pdb(self):

        print("\n> testing convex hull and pdb output of a structure")
        import tempfile
        from scipy.spatial import ConvexHull
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(50, 3))
        H = bb.Structure(pts).convex_hull()
        self.assertEqual(len(H), len(ConvexHull(pts).vertices))
        self.assertTrue(all(any(np.allclose(p, q) for q in pts) for p in H.points))

        S = bb.Structure(pts[:3], r=[1.5, 2.0, 2.5])
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "spheres.pdb")
            S.write_pdb(fname)
            lines = [l for l in open(fname) if l.startswith("ATOM")]
        # radius in the beta factor column, occupancy 1
        np.testing.assert_allclose([float(l[60:66]) for l in lines], [1.5, 2.0, 2.5])
        np.testing.assert_allclose([float(l[54:60]) for l in lines], [1.0, 1.0, 1.0])

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


    def test_multimer_write_pdb(self):

        print("\n> testing that a multimer is written frame by frame")
        import tempfile
        from copy import deepcopy
        M = deepcopy(self.M)
        M.add_xyz(M.coordinates[0] + 1.0)
        P = bb.Multimer()
        P.load(M, 2)
        for u in P.unit:
            u.set_current(1)
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "multimer.pdb")
            P.write_pdb(fname)
            text = open(fname).read()
        models = text.split("ENDMDL")[:-1]
        self.assertEqual(len(models), 2)
        for f, model in enumerate(models):
            atoms = [l for l in model.splitlines() if l.startswith("ATOM")]
            self.assertEqual(len(atoms), 2 * len(M))
            self.assertAlmostEqual(float(atoms[0][30:38]), M.coordinates[f, 0, 0], places=3)
            self.assertEqual({l[21] for l in atoms}, {"A", "B"})
        self.assertEqual([u.current for u in P.unit], [1, 1])

    def test_multimer_make_molecule(self):

        print("\n> testing that merging units keeps the knowledge of all of them")
        from copy import deepcopy
        A = deepcopy(self.M)
        B = deepcopy(self.M)
        A.knowledge["atom_ccs"]["FE"] = 2.5
        B.knowledge["atom_ccs"]["ZN"] = 1.5
        P = bb.Multimer()
        P.load_list([A, B], ["A", "B"])
        merged = P.make_molecule()
        self.assertEqual(merged.knowledge["atom_ccs"]["FE"], 2.5)
        self.assertEqual(merged.knowledge["atom_ccs"]["ZN"], 1.5)

    def test_assembly_units(self):

        print("\n> testing assembly labels, loading and frames")
        S = [bb.Structure(np.array([[float(i), 0, 0]])) for i in range(3)]

        # labels refer to the unit they were given to
        A = bb.Assembly()
        self.assertEqual(A.append(S[0]), "0")
        self.assertEqual(A.append(S[1], "B"), "B")
        self.assertEqual(A.append(S[2]), "2")
        self.assertEqual(A.unit_labels, {"0": 0, "B": 1, "2": 2})
        A.translate(10, 0, 0, unit="B")
        np.testing.assert_allclose([u.points[0, 0] for u in A.unit], [0, 11, 2])

        # lists are loaded after existing units, and loaded units keep the current frame of their structure
        A.load_list([S[0], S[1]], ["C", "D"])
        self.assertEqual(A.unit_labels["C"], 3)
        self.assertEqual(A.unit_labels["D"], 4)
        two = bb.Structure(np.array([[[0.0, 0, 0]], [[5.0, 0, 0]]]))
        two.set_current(1)
        L = bb.Assembly()
        L.load(two, 1)
        self.assertEqual(L.unit[0].current, 1)
        np.testing.assert_allclose(L.unit[0].points, [[5.0, 0, 0]])
        L.translate(1, 0, 0)
        np.testing.assert_allclose(L.unit[0].points, L.unit[0].coordinates[1])

        # coordinates of units of different size
        H = bb.Assembly()
        H.load_list([bb.Structure(np.zeros((2, 3))), bb.Structure(np.zeros((3, 3)))], ["x", "y"])
        self.assertEqual([len(p) for p in H.get_uxyz()], [2, 3])

    def test_assembly_builders(self):

        print("\n> testing assembly builders")
        point = bb.Structure(np.array([[0.0, 0, 0]]))
        for build in ["make_stacked_rings", "make_prism"]:
            A = bb.Assembly()
            A.load(point, 4)
            if build == "make_stacked_rings":
                A.make_stacked_rings(10, 5)
            else:
                A.make_prism(10, 5, 0, 0, 0)
            np.testing.assert_allclose(sorted(u.points[0, 2] for u in A.unit), [0, 0, 5, 5], atol=1e-10)

        # the default grouping does not leak from one call to the next
        A = bb.Assembly()
        A.load(point, 2)
        A.make_curved_chain(10, 5)
        B = bb.Assembly()
        B.load(point, 4)
        B.make_curved_chain(10, 5)
        self.assertEqual(len({tuple(np.round(u.points[0], 6)) for u in B.unit}), 4)

        # circular symmetry with custom labels
        C = bb.Assembly()
        for name in ["a", "b", "c"]:
            C.append(bb.Structure(np.array([[0.0, 0, 0], [1.0, 0, 0]])), name)
        C.make_circular_symmetry(5)
        radii = [np.linalg.norm(u.points[0, :2]) for u in C.unit]
        np.testing.assert_allclose(radii, radii[0])

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


    def test_polyhedron_methods(self):

        print("\n> testing polyhedron neighbors, RMSD selection and colors")
        import tempfile
        rng = np.random.default_rng(5)
        block = bb.Structure(rng.normal(size=(6, 3)))

        # every edge of a dodecahedron touches four others
        D = bb.Polyhedron()
        D.setup_polyhedron("Dodecahedron", block)
        neigh = D.get_neighbors()
        self.assertEqual(len(neigh), len(D.conn))
        self.assertTrue(all(len(v) == 4 for v in neigh.values()))

        # the RMSD uses the selected points of every unit
        P = bb.Polyhedron()
        P.setup_polyhedron("Octahedron", block)
        P.generate_polyhedron(40, 180, 0, 0)
        P.generate_polyhedron(42, 180, 5, 0, add_conformation=True)
        P.set_current(1)
        sel = [[0, 2]] * len(P.unit)
        frames = []
        for f in range(2):
            frames.append(np.concatenate([u.coordinates[f][[0, 2]] for u in P.unit]))
        expected = bb.Structure(np.array(frames)).rmsd(0, 1)
        dist = P.rmsd_distance_matrix(sel)
        self.assertAlmostEqual(np.max(dist), expected, places=6)
        self.assertEqual(P.unit[0].current, 1)

        # colors follow the connection type, also beyond the 25 default colors
        P.conn_type = np.arange(len(P.conn)) * 3
        with tempfile.TemporaryDirectory() as tmp:
            P.write_poly_architecture(output=os.path.join(tmp, "arch"))
            lines = [l.split()[2] for l in open(os.path.join(tmp, "arch.tcl")) if l.startswith("draw color")]
        colors = ['blue', 'red', 'gray', 'orange', 'yellow', 'tan', 'silver', 'green', 'white', 'pink', 'cyan', 'purple', 'lime',
                  'mauve', 'ochre', 'iceblue', 'black', 'yellow2', 'green2', 'cyan2', 'blue2', 'violet', 'magenta', 'red2', 'orange2']
        self.assertEqual(lines, [colors[(3 * k) % 25] for k in range(len(P.conn))])
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

        # prism: points centres lie on the prism with every face moved inward by the points radius (apothem and half height shrink by it),
        # and the points trace that prism enlarged by their radius (Steiner formula, with M = pi H + pi perimeter / 2)
        P = bb.Prism(10, 20, 6, radius=1.1)
        pr = P.properties["pt_radius"]
        apothem = 10 * np.cos(np.pi / 6) - pr
        side = 2 * apothem * np.tan(np.pi / 6)
        base, H = 6 * side * apothem / 2, 20 - 2 * pr
        S, M, V = 2 * base + 6 * side * H, np.pi * H + np.pi * 6 * side / 2, base * H
        self.assertAlmostEqual(P.get_surface(), S + 2 * M * pr + 4 * np.pi * pr**2, places=6)
        self.assertAlmostEqual(P.get_volume(), V + S * pr + M * pr**2 + 4 * np.pi * pr**3 / 3, places=6)
        self.assertAlmostEqual(np.max(np.linalg.norm(P.points[:, :2], axis=1)), apothem / np.cos(np.pi / 6), places=6)
        hull = ConvexHull(P.points)
        self.assertAlmostEqual(hull.volume / V, 1.0, places=6)

        # cylinder: CCS of the cylinder of points centres (radius and half height shrunk by the points radius), enlarged by points radius plus gas radius
        C = bb.Cylinder(10, 20, radius=1.1)
        R, H, rho = 10 - 1.1, 20 - 2.2, 1.1 + 1
        self.assertAlmostEqual(C.ccs(gas=1), (2 * np.pi * R**2 + 2 * np.pi * R * H + 2 * (np.pi * H + np.pi**2 * R) * rho + 4 * np.pi * rho**2) / 4, places=6)

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

    def test_half_sphere(self):

        print("\n> testing the side chain half sphere")
        from biobox.measures.path import Xlink
        X = Xlink(self.M)
        i = self.M.atomselect("*", "*", "NZ", get_index=True)[1][0]

        # the atom's position comes first, and every other point keeps thresh from all atoms
        for kwargs in [{}, {"thresh": 3.0}, {"radii": []}, {"radii": [6.0]}]:
            s = X.get_half_sphere(i, **kwargs)
            np.testing.assert_array_equal(s[0], self.M.points[i])
            d = np.linalg.norm(s[1:, None] - self.M.points[None], axis=2)
            self.assertTrue(np.all(d >= kwargs.get("thresh", 2.0)))

        # distance_matrix passes every option on
        from unittest import mock
        seen = []
        def capture(idx, **kwargs):
            seen.append(kwargs)
            raise RuntimeError("stop")
        with mock.patch.object(X, "get_half_sphere", side_effect=capture):
            with self.assertRaises(Exception):
                X.distance_matrix([i], flexible_sidechain=True, sphere_pts_surf=3.0, sphere_thresh=2.5, sphere_radii=[6.0, 5.0])
        self.assertEqual(seen[0], {"pts_surf": 3.0, "thresh": 2.5, "radii": [6.0, 5.0]})

    def test_closest_nodes_placeholder(self):

        print("\n> testing closest grid nodes of a target without accessible neighbours")
        P = self._wall_path()
        g = P.graph
        t = np.array([-8., 0, 0])
        ti = g.get_idx_from_points(np.array([t]))[0]
        g.access_grid[ti[0] - 2:ti[0] + 3, ti[1] - 2:ti[1] + 3, ti[2] - 2:ti[2] + 3] = False

        # the blocked target gets the placeholder index and distance 10000, the other one its
        # own grid point at squared distance 0.25
        dists, idx = g.get_closest_nodes(np.array([t, [8., 0, 0.5]]))
        self.assertEqual(idx.shape, (2, 3))
        np.testing.assert_array_equal(idx[0], [-1, -1, -1])
        self.assertEqual(dists[0], 10000)
        np.testing.assert_array_equal(g.get_points_from_idx(idx[1].astype(float)), [8, 0, 0])
        self.assertAlmostEqual(dists[1], 0.25)

        # the blocked target is buried, whatever maxdist, and is not exempted from clash tests
        for maxdist in [60, 1e6]:
            P.maxdist = maxdist
            d, wp = P.search_path(t, np.array([-8., 6, 0]))
            self.assertEqual(d, -2)
            self.assertEqual(len(wp), 0)
        self.assertEqual(P._target_exemptions(t)[0][1], 0.0)

    def test_path_trails(self):

        print("\n> testing the filling of paths between waypoints")
        P = self._wall_path()
        wp = np.array([[0, 0, 0], [2.5, 0, 0], [2.5, 0.4, 0], [2.5, 0.4, 0], [2.5, 3.4, 0], [2.5, 3.4, 0.5]])
        tr = P._get_trails(wp)
        expected = [[0, 0, 0], [2.5 / 3, 0, 0], [5.0 / 3, 0, 0], [2.5, 0, 0], [2.5, 0.4, 0],
                    [2.5, 1.4, 0], [2.5, 2.4, 0], [2.5, 3.4, 0], [2.5, 3.4, 0.5]]
        np.testing.assert_allclose(tr, expected, atol=1e-12)
        np.testing.assert_array_equal(wp[0], [0, 0, 0])
        self.assertAlmostEqual(P._measure_path(tr), P._measure_path(wp))

        # a path search returns the same length and end points, filled or not
        for e in [[-8., 5.3, 0.2], [5., 0, 0]]:
            s = np.array([-8., 0, 0])
            e = np.array(e)
            d0, wp0 = P.search_path(s, e, get_path=False)
            d1, wp1 = P.search_path(s, e, get_path=True)
            self.assertAlmostEqual(d0, d1)
            np.testing.assert_array_equal(wp1[[0, -1]], wp0[[0, -1]])
            steps = np.linalg.norm(np.diff(wp1, axis=0), axis=1)
            self.assertTrue(np.all(steps <= 1 + 1e-9) and np.all(steps > 0))
            self.assertAlmostEqual(P._measure_path(wp1), d0)

    def test_flexible_paths(self):

        print("\n> testing paths of side chains in contact or impossible to link")
        from unittest import mock
        from biobox.measures.path import Xlink
        ys = np.arange(-6, 6.01, 1.0)
        wall = np.array([[0, y, z] for y in ys for z in ys], float)
        X = Xlink(self.M)
        X.set_clashing_atoms(points=wall)
        X.setup_global_search(step=1.0, maxdist=10, boundaries=[[-14, 14]] * 3)

        # sphere 1 is linkable to sphere 0, sphere 2 touches sphere 0, sphere 3 is too far from all
        spheres = [np.array([[-8., 0, 0], [-8, 1, 0]]), np.array([[-8., 6, 0]]),
                   np.array([[-8.3, -0.4, 0]]), np.array([[8., 0, 0]])]
        indices = [10, 20, 30, 40]
        lookup = dict(zip(indices, spheres))
        with mock.patch.object(X, "get_half_sphere", side_effect=lambda i, **kwargs: lookup[i]):
            distance, paths = X.distance_matrix(indices, get_path=True, flexible_sidechain=True)

        paths = {tuple(p[0]): p[1] for p in paths}
        self.assertEqual(sorted(paths), [(0, 1), (0, 2), (1, 2)])
        np.testing.assert_array_equal(distance[3, :3], [-1, -1, -1])

        self.assertAlmostEqual(distance[0, 1], 5.0)
        self.assertEqual(sorted(map(tuple, paths[(0, 1)][[0, -1]])), [(-8, 1, 0), (-8, 6, 0)])

        self.assertEqual(distance[0, 2], 1.0)
        np.testing.assert_array_equal(paths[(0, 2)], [[-8.3, -0.4, 0], [-8, 0, 0]])

    def _check_exclusion(self, g, obstacles, radii):
        # grid points closer to an obstacle than its radius are inaccessible, the others accessible
        w = np.array(np.indices(g.access_grid.shape)).reshape(3, -1).T
        pts = g.get_points_from_idx(w.astype(float))
        excess = np.full(len(pts), np.inf)
        for o, r in zip(obstacles, radii):
            near = np.all(np.abs(pts - o) < r + 1, axis=1)
            excess[near] = np.minimum(excess[near], np.linalg.norm(pts[near] - o, axis=1) - r)
        acc = g.access_grid[tuple(w.T)]
        clear = np.abs(excess) > 1e-6
        self.assertTrue(np.any(~acc[clear]) and np.any(acc[clear]))
        np.testing.assert_array_equal(acc[clear], excess[clear] > 0)

    def test_vdw_exclusion(self):

        print("\n> testing the hard sphere exclusion of the atoms_vdw grids")
        from biobox.measures.path import Path, Xlink

        # an isolated atom, in global and local grids of different steps
        atom = np.array([[0.3, 0.2, 0.1]])
        for step in [1.0, 0.5]:
            P = Path(atom)
            P.setup_global_search(step=step, use_hull=False, boundaries=[[-6, 6]] * 3, params=np.array([3.25]))
            self._check_exclusion(P.graph, atom, [3.25])
            P.setup_local_search(step=step, maxdist=12, params=np.array([3.25]))
            P.graph.place_local_grid(np.array([-1., 0.5, 0]), np.array([1., 1.5, 0]))
            self._check_exclusion(P.graph, atom, [3.25])

        # backbone atoms of HSP: van der Waals radius plus probe, in global and local grids
        vdw = self.M.knowledge["atom_vdw"]
        X = Xlink(self.M)
        sel = X.set_clashing_atoms(densify=False, atoms_vdw=True, probe=1.4)
        obstacles = self.M.points[sel]
        radii = np.array([vdw[a] for a in self.M.data["atomtype"].values[sel]]) + 1.4
        np.testing.assert_array_equal(X.params, radii)
        self.assertEqual(sorted(set(np.round(radii, 2))), [2.92, 2.95, 3.1])
        X.setup_global_search(step=1.0)
        self._check_exclusion(X.graph, obstacles, radii)
        X.setup_local_search(step=0.5, maxdist=12)
        X.graph.place_local_grid(obstacles[0], obstacles[30])
        self._check_exclusion(X.graph, obstacles, radii)

        # the default model is unchanged
        import hashlib
        X = Xlink(self.M)
        X.set_clashing_atoms(densify=False)
        idx = self.M.atomselect("*", "LYS", "NZ", use_resname=True, get_index=True)[1]
        grids = []
        X.setup_global_search(step=1.0, use_hull=False)
        grids.append(X.graph.access_grid)
        X.setup_global_search(step=0.5, use_hull=False)
        grids.append(X.graph.access_grid)
        X.setup_local_search(step=1.0, maxdist=20)
        X.graph.place_local_grid(self.M.points[idx[0]], self.M.points[idx[1]])
        grids.append(X.graph.access_grid)
        expected = [((57, 38, 52), 93229, "5011001f5b4694145dda3268c86a045879b2d12c"),
                    ((111, 72, 101), 783750, "ad51a46baff4b4969be7a2fdc330bc5816883b2e"),
                    ((21, 21, 21), 3898, "32b1a831b5405e6991aba3a7eae3b065ea67ee4e")]
        for g, e in zip(grids, expected):
            self.assertEqual((g.shape, int(g.sum()), hashlib.sha1(np.packbits(g).tobytes()).hexdigest()), e)

    def test_grid_boundaries(self):

        print("\n> testing global grids built within boundaries or around a cloud")
        from biobox.measures.path import Path
        atom = np.array([[12.2, 36.7, 73.6]])
        lo = atom[0] - 5
        hi = atom[0] + 5
        cloud = np.array([lo + 1, hi - 1])
        for params in [np.array([]), np.array([3.2])]:
            for kwargs in [{"boundaries": np.array([lo, hi]).T}, {"cloud": cloud}]:
                P = Path(atom)
                P.setup_global_search(step=1.0, use_hull=False, params=params, **kwargs)
                g = P.graph
                np.testing.assert_array_equal(g.access_grid.shape, [11, 11, 11])
                np.testing.assert_allclose(g.get_points_from_idx(np.array([0., 0, 0])), lo, atol=1e-9)
                np.testing.assert_allclose(g.get_points_from_idx(np.array([10., 10, 10])), hi, atol=1e-9)
                self.assertFalse(g.is_accessible(atom)[0])
                self.assertTrue(g.is_accessible(np.array([lo]))[0])
    def test_pdb2pqr_histidine(self):

        print("\n> testing pdb2pqr histidines and default forcefield")
        import pandas as pd
        NALA = "N H1 H2 H3 CA HA CB HB1 HB2 HB3 C O".split()
        CHID = "N H CA HA CB HB1 HB2 CG ND1 HD1 CE1 HE1 NE2 CD2 HD2 C O OXT".split()
        CALA = "N H CA HA CB HB1 HB2 HB3 C O OXT".split()

        def build(chains):
            rows = []
            xyz = []
            x = 0.0
            for ci, (cname, residues) in enumerate(chains):
                for ri, (rn, names) in enumerate(residues):
                    for a in names:
                        rows.append(["ATOM", 0, a, rn, cname, ri + 1, 1.0, 0.0, a[0], 1.5, 0.0])
                        xyz.append([x + (0.5 if a == "C" else 0.0), ci * 100.0, 0.0])
                    x += 1.5
            M = bb.Molecule()
            M.data = pd.DataFrame(rows, columns=["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype", "radius", "charge"])
            M.data["index"] = np.arange(len(rows))
            M.coordinates = np.array([xyz])
            M.current = 0
            M.points = M.coordinates[0]
            return M

        ff = np.loadtxt(os.path.join(os.path.dirname(bb.__file__), "data", "amber14sb.dat"), usecols=(0, 1, 2), dtype=str)
        q = {(r, n): float(c) for r, n, c in ff}

        # a C-terminal HID followed by another chain, with the default forcefield path
        M = build([("X", [("ALA", NALA), ("HIS", CHID)]), ("Y", [("ALA", NALA), ("ALA", CALA)])])
        pqr = M.pdb2pqr()
        first_Y = len(NALA) + len(CHID)
        self.assertEqual(list(M.data["resname"][len(NALA):first_Y].unique()), ["CHID"])
        self.assertEqual(M.data["resname"][first_Y], "NALA")
        self.assertAlmostEqual(pqr["charge"][first_Y], q[("NALA", "N")])
        expected = 2 * sum(q[("NALA", a)] for a in NALA) + sum(q[("CHID", a)] for a in CHID) + sum(q[("CALA", a)] for a in CALA)
        self.assertAlmostEqual(pqr["charge"].sum(), expected, places=4)

        # a C-terminal HID as the last residue of the structure
        M = build([("X", [("ALA", NALA), ("HIS", CHID)])])
        pqr = M.pdb2pqr()
        self.assertEqual(list(M.data["resname"][len(NALA):].unique()), ["CHID"])

    def test_renumber_resid_keep_chains(self):

        print("\n> testing residue renumbering")
        M = self.M
        A = M.data["chain"] == "A"
        M.data.loc[A, "resid"] = M.data["resid"][A] - M.data["resid"][A].min() - 2
        # the second residue of chain C takes the number of the first one
        C = np.flatnonzero(M.data["chain"].values == "C")
        r = M.data["resid"].values[C]
        M.data.loc[C[r == r.min() + 1], "resid"] = r.min()

        M.renumber_resid_keep_chains(start_from=100)
        CA = M.atomselect("*", "*", "CA", get_index=True)[1]
        for c in np.unique(M.data["chain"]):
            r = M.data["resid"].values[CA][M.data["chain"].values[CA] == c]
            np.testing.assert_array_equal(r, np.arange(100, 100 + len(r)))

        # every atom carries the number of its residue's CA
        for i in CA:
            idx = M.same_residue(i, get_index=True)[1]
            self.assertTrue(np.all(M.data["resid"].values[idx] == M.data["resid"].values[i]))

        M.renumber_resid_keep_chains(reset_resid_with_chain=False)
        np.testing.assert_array_equal(M.data["resid"].values[CA], np.arange(1, len(CA) + 1))

    def test_reorder_resid(self):

        print("\n> testing residue reordering")
        M = self.M
        CA = M.atomselect("A", "*", "CA", get_index=True)[1]
        resnames = M.data["resname"].values[CA]
        ca_xyz = M.points[CA].copy()
        chain_order = list(dict.fromkeys(M.data["chain"].values))
        k = len(CA)

        # 1-based values, as a list: the last 3 residues move to the front
        order = list(np.r_[np.arange(k - 2, k + 1), np.arange(1, k - 2)])
        M.reorder_resid(order, chain="A", renumber=False)

        CA2 = M.atomselect("A", "*", "CA", get_index=True)[1]
        expected = np.r_[np.arange(k - 3, k), np.arange(0, k - 3)]
        np.testing.assert_array_equal(M.data["resname"].values[CA2], resnames[expected])
        np.testing.assert_array_equal(M.points[CA2], ca_xyz[expected])
        self.assertEqual(list(dict.fromkeys(M.data["chain"].values)), chain_order)
        self.assertTrue(M.data.index.equals(pd.RangeIndex(len(M.data))))
        np.testing.assert_array_equal(M.data["index"].values, np.arange(len(M.data)))

        with self.assertRaises(ValueError):
            M.reorder_resid(order[:-1], chain="A")

    def test_match_residue(self):

        print("\n> testing residue matching between strands")
        A = self.M.get_subset(self.M.atomselect("A", "*", "*", get_index=True)[1])
        res = A.data["resid"].values[A.atomselect("*", "*", "CA", get_index=True)[1]]

        r1, r2 = A.match_residue(A.get_subset(np.flatnonzero(A.data["resid"].values != res[27])))
        self.assertEqual(list(r1), list(np.delete(res, 27)))
        self.assertEqual(list(r2), list(np.delete(res, 27)))

        r1, r2 = A.match_residue(A.get_subset(np.flatnonzero(A.data["resid"].values > res[1])))
        self.assertEqual(list(r1), list(res[2:]))

    def test_get_fasta_variants(self):

        print("\n> testing one-letter codes of residue variants")
        M = self.M.get_subset(self.M.atomselect("A", "*", "*", get_index=True)[1])
        reference = M.get_fasta()
        resnames = M.data["resname"].values.copy()
        for old, new in [("MET", "MSE"), ("HIS", "HIE"), ("CYS", "CYX")]:
            M.data.loc[resnames == old, "resname"] = new
        self.assertEqual(M.get_fasta(), reference)

        first = M.data["resid"].values == M.data["resid"].values[0]
        M.data.loc[first, "resname"] = "N" + resnames[0]
        self.assertEqual(M.get_fasta(), reference)
        M.data.loc[first, "resname"] = "UNK"
        self.assertEqual(M.get_fasta(), "X" + reference[1:])

    def _small_ensemble(self, frames):
        # first 5 atoms of HSP, with every frame shifted by its index along x
        S = self.M.get_subset(np.arange(5))
        S.coordinates = np.array([S.coordinates[0] + [k, 0.0, 0.0] for k in range(frames)])
        S.set_current(0)
        return S

    def test_addall_conformation_array(self):

        print("\n> testing addall with conformations given as list or numpy array")
        from copy import deepcopy
        S = self._small_ensemble(4)
        S2 = deepcopy(S)
        for conformations in [[0, 3], np.array([0, 3])]:
            N = S.addall(S2, conformations=conformations)
            self.assertEqual(N.coordinates.shape, (2, 10, 3))
            np.testing.assert_allclose(N.coordinates[1, :5], S.coordinates[3])
        self.assertEqual(S.addall(S2, conformations=np.array([], dtype=int)).coordinates.shape, (4, 10, 3))

    def test_atoms_ccs_array(self):

        print("\n> testing that atomic CCS radii are always returned as a numpy array")
        M = self.M.get_subset(np.arange(20))
        first = M.get_atoms_ccs()
        second = M.get_atoms_ccs()
        self.assertIsInstance(first, np.ndarray)
        self.assertIsInstance(second, np.ndarray)
        np.testing.assert_allclose(first, second)

        # a user-defined column is returned as is, as a copy
        custom = np.linspace(1.0, 2.0, len(M.data))
        M.data["atom_ccs"] = custom
        radii = M.get_atoms_ccs()
        self.assertIsInstance(radii, np.ndarray)
        np.testing.assert_allclose(radii, custom)
        radii[:] = 0.0
        np.testing.assert_allclose(M.data["atom_ccs"].values, custom)

    def test_import_md(self):

        print("\n> testing import of CASTEP md files")
        import tempfile
        elements = ["C", "H", "Fe"]
        steps = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
                 np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.5, 1.0]])]

        def vectors(xyz, tag):
            return [" %-2s%15d   %24.16E   %24.16E   %24.16E  <-- %s\n" % (e, 1, v[0], v[1], v[2], tag)
                    for e, v in zip(elements, xyz)]

        lines = [" BEGIN header\n", "  \n", " END header\n", "  \n"]
        for k, xyz in enumerate(steps):
            if k > 0:
                lines.append("  \n")
            lines.append("                      %24.16E\n" % (k * 1.0))
            lines.append("                      %24.16E  %24.16E  %24.16E  <-- E\n" % (-1.0, -1.0, 0.0))
            lines.append("                      %24.16E  <-- T\n" % 0.0)
            for row in np.identity(3) * 10.0:
                lines.append("                      %24.16E  %24.16E  %24.16E  <-- h\n" % tuple(row))
            lines += vectors(xyz, "R") + vectors(np.zeros((3, 3)), "V") + vectors(np.zeros((3, 3)), "F")

        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "test.md")
            with open(fname, "w") as f:
                f.writelines(lines)
            M = bb.Molecule()
            M.import_md(fname)

        np.testing.assert_allclose(M.coordinates, np.array(steps))
        self.assertEqual(list(M.data["name"]), elements)
        self.assertEqual(list(M.data["atomtype"]), elements)
        np.testing.assert_array_equal(M.data["index"].values, [0, 1, 2])
        np.testing.assert_array_equal(M.data["resid"].values, [0, 0, 0])
        self.assertTrue(np.issubdtype(M.data["index"].dtype, np.integer))
        self.assertTrue(np.issubdtype(M.data["resid"].dtype, np.integer))
        self.assertTrue(np.issubdtype(M.data["occupancy"].dtype, np.floating))
        self.assertTrue(np.issubdtype(M.data["beta"].dtype, np.floating))
        np.testing.assert_allclose(M.data["occupancy"].values, 1.0)
        np.testing.assert_allclose(M.data["beta"].values, 0.0)
        vdw = M.know("atom_vdw")
        np.testing.assert_allclose(M.data["radius"].values, [vdw["C"], vdw["H"], vdw.get("FE", vdw["."])])
        np.testing.assert_allclose(M.data["charge"].values, 0.0)
        self.assertEqual(list(M.data["altloc"]), [""] * 3)
        self.assertEqual(list(M.data["icode"]), [""] * 3)

    def test_pdb_model_records(self):

        print("\n> testing MODEL and END records of multi-model pdb files")
        import tempfile
        S = self._small_ensemble(12)
        A = bb.Multimer()
        A.load_list([S, S], ["1", "2"])

        with tempfile.TemporaryDirectory() as tmp:
            for writer, natoms in [(S, 5), (A, 10)]:
                fname = os.path.join(tmp, "models.pdb")
                writer.write_pdb(fname)
                with open(fname) as f:
                    lines = [l.rstrip("\n") for l in f]

                # model serial right-justified in columns 11-14, and END after the last ENDMDL
                models = [l for l in lines if l.startswith("MODEL")]
                self.assertEqual(models, ["MODEL     %4d" % (k + 1) for k in range(12)])
                self.assertEqual([l[10:14] for l in models[8:10]], ["   9", "  10"])
                self.assertEqual(lines[-2:], ["ENDMDL", "END"])

                M = bb.Molecule()
                M.import_pdb(fname)
                self.assertEqual(M.coordinates.shape, (12, natoms, 3))
                np.testing.assert_allclose(M.coordinates[:, :5], S.coordinates, atol=1e-3)

                try:
                    from Bio.PDB import PDBParser
                except ImportError:
                    continue
                structure = PDBParser(QUIET=True).get_structure("test", fname)
                self.assertEqual(len(structure), 12)
                self.assertEqual([m.serial_num for m in structure], list(range(1, 13)))

    def test_clean(self):

        print("\n> testing removal of alternate locations and non amino acid residues")
        import tempfile
        atoms = [("N", "", "ALA", 1, 1.00), ("CA", "A", "ALA", 1, 0.40), ("CB", "A", "ALA", 1, 0.40),
                 ("CA", "B", "ALA", 1, 0.60), ("CB", "B", "ALA", 1, 0.60), ("C", "", "ALA", 1, 1.00),
                 ("N", "", "SER", 2, 1.00), ("OG", "B", "SER", 2, 0.50), ("OG", "A", "SER", 2, 0.50),
                 ("O", "", "HOH", 3, 1.00)]
        lines = []
        for frame in range(2):
            lines.append("MODEL     %4d\n" % (frame + 1))
            for i, (name, alt, resname, resid, occ) in enumerate(atoms):
                lines.append("ATOM  %5d  %-3s%1s%3s A%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           %s\n"
                             % (i + 1, name, alt, resname, resid, i + 10.0 * frame, 0.0, 0.0, occ, 10.0, name[0]))
            lines.append("ENDMDL\n")
        lines.append("END\n")

        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "altloc.pdb")
            with open(fname, "w") as f:
                f.writelines(lines)
            M = bb.Molecule()
            M.import_pdb(fname)
            before = M.data.copy()
            files = sorted(os.listdir(tmp)), sorted(os.listdir("."))

            # B is kept in ALA 1 (higher occupancy), and B also in SER 2 (tie, first in the file)
            C = M.clean()
            self.assertEqual(sorted(os.listdir(tmp)), files[0])
            self.assertEqual(sorted(os.listdir(".")), files[1])

        kept = [0, 3, 4, 5, 6, 7]
        self.assertEqual(list(C.data["name"]), [atoms[i][0] for i in kept])
        self.assertEqual(list(C.data["altloc"]), [""] * len(kept))
        np.testing.assert_allclose(C.coordinates, M.coordinates[:, kept])
        np.testing.assert_array_equal(C.data["index"].values, np.arange(len(kept)))

        # the water is kept on request
        C2 = M.clean(remove_non_amino=False)
        self.assertEqual(list(C2.data["resname"]), ["ALA"] * 4 + ["SER"] * 2 + ["HOH"])
        np.testing.assert_allclose(C2.coordinates, M.coordinates[:, kept + [9]])
        self.assertEqual(list(C2.data["altloc"]), [""] * 7)

        # the molecule itself is unchanged
        pd.testing.assert_frame_equal(M.data, before)
        self.assertEqual(M.coordinates.shape, (2, 10, 3))

    def test_vdw_density_keeps_atomtypes(self):

        print("\n> testing that van der Waals densities fill in only empty atomtypes of the selected atoms")
        M = self.M.get_subset(np.arange(11))
        types = list(M.data["atomtype"].values)
        # atom 0 is an iron, atom 4 (CB) keeps an atomtype that its name does not give
        M.data.iloc[0, M.data.columns.get_loc("name")] = "FE"
        types[0] = "FE"
        types[4] = "S"
        # CA of THR 33 and of GLY 34 have no atomtype
        types[1] = ""
        types[8] = ""
        M.data["atomtype"] = types

        # only the selected atoms are filled in
        sel = np.arange(8)
        axes = M._grid_axes(M.points[sel], 1.0, 3)
        M._vdw_density_on_grid(sel, axes, 1.0, 5)
        expected = list(types)
        expected[1] = "C"
        self.assertEqual(list(M.data["atomtype"]), expected)

        # a density of all atoms fills in the rest
        M.get_vdw_density(step=1.0, kernel_half_width=5)
        expected[8] = "C"
        self.assertEqual(list(M.data["atomtype"]), expected)

        # an atom whose element cannot be guessed is still an error
        M.data.iloc[2, M.data.columns.get_loc("name")] = "XX"
        M.data.iloc[2, M.data.columns.get_loc("atomtype")] = ""
        with self.assertRaises(Exception):
            M.get_vdw_density(step=1.0, kernel_half_width=5)

    def test_pdb2pqr_n_terminus(self):

        print("\n> testing N-terminal residue names in pdb2pqr")
        import tempfile
        ala = ["N", "H1", "H2", "H3", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O"]

        def molecule(atoms, tmp):
            lines = ["ATOM  %5d %-4s %-3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                     % (i + 1, name if len(name) > 1 and resname == name else " " + name, resname, resid, 1.3 * i, 0.0, 0.0,
                        name if resname == name else name[0])
                     for i, (name, resname, resid) in enumerate(atoms)]
            fname = os.path.join(tmp, "nterm.pdb")
            with open(fname, "w") as f:
                f.writelines(lines + ["END\n"])
            M = bb.Molecule()
            M.import_pdb(fname)
            return M

        with tempfile.TemporaryDirectory() as tmp:
            # an ordinary N-terminal residue is renamed
            M = molecule([(name, "ALA", 1) for name in ala], tmp)
            M.pdb2pqr()
            self.assertEqual(list(M.data["resname"]), ["NALA"] * len(ala))

            # an ion before it is never renamed (it is not in the forcefield file)
            M = molecule([("ZN", "ZN", 1)] + [(name, "ALA", 2) for name in ala], tmp)
            with self.assertRaises(Exception):
                M.pdb2pqr()
            self.assertEqual(M.data["resname"].values[0], "ZN")
            self.assertNotIn("NZN", list(M.data["resname"]))

    def test_pdb_ter_records(self):

        print("\n> testing TER records and serial numbers in written pdb files")
        import tempfile
        M = self._molecule_from_atoms([("N", "N", "A", 1, [0, 0, 0]), ("CA", "C", "A", 1, [1, 0, 0]),
                                       ("CA", "C", "B", 5, [2, 0, 0]),
                                       ("CA", "C", "C", 7, [3, 0, 0]), ("CB", "C", "C", 8, [4, 0, 0])])
        M.data["icode"] = ["", "", "", "", "A"]
        M.add_xyz(M.coordinates[0] + 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "ter.pdb")
            M.write_pdb(fname)
            models = open(fname).read().split("ENDMDL")[:-1]
            for model in models:
                lines = [l for l in model.splitlines() if l.startswith(("ATOM", "TER"))]
                self.assertEqual([l[:6].strip() for l in lines], ["ATOM", "ATOM", "TER", "ATOM", "TER", "ATOM", "ATOM", "TER"])
                # TER takes the next serial number, and the atoms after it continue from the following one
                self.assertEqual([int(l[6:11]) for l in lines], list(range(1, 9)))
                ter = [l for l in lines if l.startswith("TER")]
                self.assertEqual(ter, ["TER       3      ALA A   1 ", "TER       5      ALA B   5 ", "TER       8      ALA C   8A"])

            # TER records are ignored when reading
            R = bb.Molecule()
            R.import_pdb(fname)
            self.assertEqual(list(R.data["chain"]), ["A", "A", "B", "C", "C"])
            self.assertEqual(list(R.data["name"]), ["N", "CA", "CA", "CA", "CB"])
            np.testing.assert_allclose(R.coordinates, M.coordinates, atol=1e-3)

            try:
                from Bio.PDB import PDBParser
                structure = PDBParser(QUIET=True).get_structure("ter", fname)
                for model in structure:
                    self.assertEqual([(c.id, len(list(c.get_atoms()))) for c in model], [("A", 2), ("B", 1), ("C", 2)])
            except ImportError:
                pass

            # split_struc closes each guessed chain once (here, a single chain)
            M.write_pdb(fname, split_struc=True)
            lines = [l for l in open(fname) if l.startswith(("ATOM", "TER"))]
            self.assertEqual([l[:6].strip() for l in lines], ["ATOM"] * 5 + ["TER"] + ["ATOM"] * 5 + ["TER"])
            self.assertEqual(lines[5][6:11], "    6")

            # a multimer closes every unit with a TER record taking the next serial number
            A = bb.Multimer()
            A.load_list([M.get_subset([0, 1]), M.get_subset([2])], ["1", "2"])
            A.write_pdb(fname)
            lines = [l for l in open(fname).read().split("ENDMDL")[0].splitlines() if l.startswith(("ATOM", "TER"))]
            self.assertEqual([l[:6].strip() for l in lines], ["ATOM", "ATOM", "TER", "ATOM", "TER"])
            self.assertEqual([int(l[6:11]) for l in lines], list(range(1, 6)))
            self.assertEqual([l for l in lines if l.startswith("TER")], ["TER       3      ALA A   1 ", "TER       5      ALA B   5 "])
            R = bb.Molecule()
            R.import_pdb(fname)
            self.assertEqual(list(R.data["chain"]), ["A", "A", "B"])
            self.assertEqual(R.coordinates.shape, (2, 3, 3))

    def test_pdb_formal_charge(self):

        print("\n> testing formal charges in pdb files")
        import tempfile
        lines = ["HETATM    1 ZN    ZN A   1       0.000   0.000   0.000  1.00  0.00          ZN2+\n",
                 "HETATM    2 CL    CL A   2       1.000   0.000   0.000  1.00  0.00          CL1-\n",
                 "HETATM    3  O   HOH A   3       2.000   0.000   0.000  1.00  0.00           O  \n",
                 "END\n"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "charges.pdb")
            with open(fname, "w") as f:
                f.writelines(lines)
            M = bb.Molecule()
            M.import_pdb(fname, include_hetatm=True)
            self.assertEqual(list(M.data["formal_charge"]), [2, -1, 0])
            self.assertTrue(np.issubdtype(M.data["formal_charge"].dtype, np.integer))

            out = os.path.join(tmp, "out.pdb")
            M.write_pdb(out)
            written = [l.rstrip("\n") for l in open(out) if l.startswith("HETATM")]
            self.assertEqual([l[76:80] for l in written], ["ZN2+", "CL1-", " O  "])
            R = bb.Molecule()
            R.import_pdb(out, include_hetatm=True)
            self.assertEqual(list(R.data["formal_charge"]), [2, -1, 0])

            # derived molecules carry the column, and a missing column is written blank
            self.assertEqual(list(M.get_subset([1, 2]).data["formal_charge"]), [-1, 0])
            self.assertEqual(list((M + M).data["formal_charge"]), [2, -1, 0] * 2)
            N = M.get_subset([0, 1])
            N.data = N.data.drop(columns="formal_charge")
            both = M + N
            self.assertEqual(list(both.data["formal_charge"]), [2, -1, 0, 0, 0])
            self.assertTrue(np.issubdtype(both.data["formal_charge"].dtype, np.integer))
            N.write_pdb(out)
            self.assertEqual([l[78:80] for l in open(out) if l.startswith("HETATM")], ["  ", "  "])

            M.data["formal_charge"] = [10, 0, 0]
            with self.assertRaises(Exception):
                M.write_pdb(os.path.join(tmp, "large.pdb"))

            # importers of other formats set formal charges to 0
            pqr = os.path.join(tmp, "test.pqr")
            with open(pqr, "w") as f:
                f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  0.1414 1.8240\nEND\n")
            P = bb.Molecule()
            P.import_pqr(pqr)
            self.assertEqual(list(P.data["formal_charge"]), [0])
            gro = os.path.join(tmp, "test.gro")
            with open(gro, "w") as f:
                f.writelines(["one atom\n", "    1\n", "    1ALA      N    1   0.000   0.000   0.000\n", "   1.00000   1.00000   1.00000\n"])
            G = bb.Molecule()
            G.import_gro(gro)
            self.assertEqual(list(G.data["formal_charge"]), [0])

    def test_guess_chain_split_capped(self):

        print("\n> testing chain splitting of capped peptides")
        import tempfile, shutil

        def peptide(gap):
            # ACE-ALA-ALA-NME along x, with a gap before the second ALA
            residues = [("ACE", ["CH3", "C", "O"]), ("ALA", ["N", "CA", "C", "O"]),
                        ("ALA", ["N", "CA", "C", "O"]), ("NME", ["N", "CH3"])]
            lines = []
            x = 0.0
            for r, (resname, names) in enumerate(residues):
                if r == 2:
                    x += gap
                for name in names:
                    lines.append("ATOM  %5d %-4s %-3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s\n"
                                 % (len(lines) + 1, " " + name if len(name) < 4 else name, resname, r + 1, x, 0.0, 0.0, name[0]))
                    x += 1.3
            return lines + ["END\n"]

        tmp = tempfile.mkdtemp()
        try:
            for gap, expected in [(0.0, [0, 13]), (20.0, [0, 7, 13])]:
                fname = os.path.join(tmp, "capped.pdb")
                with open(fname, "w") as f:
                    f.writelines(peptide(gap))
                M = bb.Molecule()
                M.import_pdb(fname)
                n, intervals, gaps = M.guess_chain_split()
                self.assertEqual(list(intervals), expected)
                self.assertEqual(list(M.data["chain"]), (["A"] * 7 + ["B"] * 6) if gap else ["A"] * 13)
        finally:
            shutil.rmtree(tmp)

    def test_assembly_radii_and_pdb(self):

        print("\n> testing Assembly radii, buried surface and PDB output")
        import tempfile
        A = bb.Assembly()
        A.load(bb.Sphere(10, radius=2.5), 2)
        A.translate(30, 0, 0, unit=["1"])

        # radii are carried into the merged structure, so units out of contact bury nothing
        np.testing.assert_array_equal(np.unique(A.make_structure().data["radius"]), [2.5])
        self.assertAlmostEqual(A.get_buried(), 0.0, places=6)

        # as in Structure.write_pdb: serials in file order, occupancy 1, radius in beta
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "assembly.pdb")
            A.write_pdb(fname)
            lines = open(fname).readlines()
        n = sum(len(u.points) for u in A.unit)
        self.assertEqual([int(l[6:11]) for l in lines], list(range(1, n + 1)))
        self.assertTrue(all(float(l[54:60]) == 1.0 and float(l[60:66]) == 2.5 for l in lines))

    def test_polyhedron_measures_and_deformation(self):

        print("\n> testing measures and deformation classes of a Polyhedron")
        import tempfile
        import biobox.measures.calculators as C
        block = bb.Structure(np.random.default_rng(5).normal(size=(6, 3)))
        P = bb.Polyhedron()
        P.setup_polyhedron("Octahedron", block)
        P.generate_polyhedron(40, 180, 0, 0)

        # a Polyhedron is measured through make_structure
        self.assertAlmostEqual(C.rgyr(P), C.rgyr(P.make_structure()), places=6)
        self.assertAlmostEqual(C.sasa(P)[0], C.sasa(P.make_structure())[0], places=6)

        # three vertices in two deformation classes: one coefficient per class, in both methods
        P.add_deformation([0, 1])
        P.add_deformation(2)
        P.generate_polyhedron(40, 180, 0, 0, deformation=[1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            P.write_poly_architecture(output=os.path.join(tmp, "arch"), deformation=[1, 2])
            with self.assertRaises(Exception):
                P.write_poly_architecture(output=os.path.join(tmp, "arch"), deformation=[1, 2, 3])
        with self.assertRaises(Exception):
            P.generate_polyhedron(40, 180, 0, 0, deformation=[1, 2, 3])

    def test_assembly_conformations_and_labels(self):

        print("\n> testing Assembly conformations and positional placement")
        rng = np.random.default_rng(7)

        # units already holding two frames: the added conformation becomes the current one everywhere
        A = bb.Assembly()
        A.load_list([bb.Structure(rng.normal(size=(2, 5, 3))), bb.Structure(rng.normal(size=(2, 4, 3)))], ["x", "y"])
        B = bb.Assembly()
        B.load_list([bb.Structure(rng.normal(size=(5, 3))), bb.Structure(rng.normal(size=(4, 3)))], ["x", "y"])
        A.add_conformation(B)
        self.assertEqual(A.current, 2)
        for u, v in zip(A.unit, B.unit):
            self.assertEqual(u.current, 2)
            np.testing.assert_array_equal(u.points, v.points)

        # a unit with a different number of points leaves the assembly untouched
        C = bb.Assembly()
        C.load_list([bb.Structure(rng.normal(size=(5, 3))), bb.Structure(rng.normal(size=(3, 3)))], ["x", "y"])
        with self.assertRaises(Exception):
            A.add_conformation(C)
        self.assertEqual([len(u.coordinates) for u in A.unit], [3, 3])

        # stacked rings and prisms place units by position, whatever their labels
        block = rng.normal(size=(6, 3))
        for method, args in [("make_stacked_rings", (20, 10)), ("make_prism", (20, 10, 10, 20, 30))]:
            placed = []
            for labels in [[], ["a", "b", "c", "d"]]:
                A = bb.Assembly()
                A.load_list([bb.Structure(block.copy()) for _ in range(4)], labels)
                getattr(A, method)(*args)
                placed.append(A.get_all_xyz())
            np.testing.assert_allclose(placed[0], placed[1])

    def test_polyhedron_current_and_deformation_axis(self):

        print("\n> testing Polyhedron current conformation and deformation axes")
        block = bb.Structure(np.random.default_rng(5).normal(size=(6, 3)))
        P = bb.Polyhedron()
        P.setup_polyhedron("Octahedron", block)
        P.generate_polyhedron(40, 180, 0, 0)
        P.generate_polyhedron(42, 180, 5, 0, add_conformation=True)

        # the selected conformation is kept by the polyhedron and restored after measuring
        P.set_current(0)
        self.assertEqual(P.current, 0)
        P.rmsd_distance_matrix()
        self.assertEqual([u.current for u in P.unit], [0] * len(P.unit))

        # the axis given is normalised as a copy, also when made of integers
        axis = np.array([0.0, 0.0, 2.0])
        P.add_deformation(0, vector=axis)
        np.testing.assert_array_equal(axis, [0.0, 0.0, 2.0])
        P.add_deformation(1, vector=np.array([0, 3, 4]))
        np.testing.assert_allclose(P.deform[-1, 2:5], [0, 0.6, 0.8])

    @staticmethod
    def _golden_spiral(n):
        # n unit vectors evenly spread on the sphere
        k = np.arange(n) + 0.5
        z = 1 - 2 * k / n
        phi = k * np.pi * (3 - np.sqrt(5))
        s = np.sqrt(1 - z**2)
        return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)

    @staticmethod
    def _convex_shapes():
        # shapes of every class, with a function measuring the distance of points from the surface of their nominal body K (in the frame K is built in).
        # K is the intersection of half-spaces n . y <= c, so the distance of a point x inside it from its surface is the minimum of c - n . x over them
        def planes(normals, offsets):
            normals = np.asarray(normals, dtype=float)
            return lambda X: np.min(np.asarray(offsets)[None] - np.dot(X, normals.T), axis=1)

        def tangent_planes(family):
            # one-parameter family of half-spaces: dense sampling, then refinement around the best sample
            def dist(X):
                th = np.arange(3600) * 2 * np.pi / 3600
                n, c = family(th)
                t = th[np.argmin(c[None] - np.dot(X, n.T), axis=1)]
                step = 2 * np.pi / 3600
                for _ in range(40):
                    cand = t[:, None] + step * np.linspace(-1, 1, 5)[None]
                    n, c = family(cand.ravel())
                    f = (c - np.sum(n * np.repeat(X, 5, axis=0), axis=1)).reshape(-1, 5)
                    t = cand[np.arange(len(t)), np.argmin(f, axis=1)]
                    step /= 2
                n, c = family(t)
                return c - np.sum(n * X, axis=1)
            return dist

        def through(b, n, inside):
            # unit normals of planes through points b, oriented away from the point inside
            n = n / np.linalg.norm(n, axis=1)[:, None]
            n *= np.sign(np.sum(n * (b - inside), axis=1))[:, None]
            return n, np.sum(n * b, axis=1)

        def ellipsoid(a, b, c):
            # tangent planes of an ellipsoid: offset sqrt(a^2 u_x^2 + b^2 u_y^2 + c^2 u_z^2) for unit normal u; minimum over u refined on local grids
            def support(U):
                return np.sqrt((a * U[..., 0])**2 + (b * U[..., 1])**2 + (c * U[..., 2])**2)

            def dist(X):
                U0 = test_structures._golden_spiral(4000)
                U = U0[np.argmin(support(U0)[None] - np.dot(X, U0.T), axis=1)]
                step = 0.06
                grid = np.stack(np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)), axis=-1).reshape(-1, 2)
                for _ in range(40):
                    t1 = np.cross(U, [0.6, 0.0, 0.8])
                    t1 /= np.linalg.norm(t1, axis=1)[:, None]
                    t2 = np.cross(U, t1)
                    cand = U[:, None, :] + step * (grid[:, 0, None] * t1[:, None, :] + grid[:, 1, None] * t2[:, None, :])
                    cand /= np.linalg.norm(cand, axis=-1)[..., None]
                    U = cand[np.arange(len(U)), np.argmin(support(cand) - np.einsum("nkj,nj->nk", cand, X), axis=1)]
                    step /= 2
                return support(U) - np.sum(U * X, axis=1)
            return dist

        def prism(r, h, n, skew):
            ang = 2 * np.pi * np.arange(n + 1) / n
            v = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n + 1)], axis=1)
            nrm, off = through(v[:-1], np.cross(v[1:] - v[:-1], [0, skew, h]), [0, skew / 2, h / 2])
            return planes(np.vstack([nrm, [[0, 0, -1], [0, 0, 1]]]), np.append(off, [0, h]))

        def cylinder(r1, r2, h, skew):
            def family(th):
                b = np.stack([r1 * np.cos(th), r2 * np.sin(th), np.zeros(len(th))], axis=1)
                tangent = np.stack([-r1 * np.sin(th), r2 * np.cos(th), np.zeros(len(th))], axis=1)
                return through(b, np.cross(tangent, [0, skew, h]), [0, skew / 2, h / 2])
            lateral, caps = tangent_planes(family), planes([[0, 0, -1], [0, 0, 1]], [0, h])
            return lambda X: np.minimum(lateral(X), caps(X))

        def cone(r, h, skew):
            def family(th):
                b = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(len(th))], axis=1)
                tangent = np.stack([-np.sin(th), np.cos(th), np.zeros(len(th))], axis=1)
                return through(b, np.cross(tangent, [0, skew, h] - b), [0, skew / 4, h / 4])
            lateral, base = tangent_planes(family), planes([[0, 0, -1]], [0])
            return lambda X: np.minimum(lateral(X), base(X))

        S = bb.Sphere(8, radius=1.5, n_sphere_point=1500)
        S.squeeze([1.3, 0.8])
        return [("squeezed sphere", S, ellipsoid(8 * 1.3, 8 * 0.8, 8 / 1.04)),
                ("ellipsoid", bb.Ellipsoid(6, 8, 10, pts_density_u=np.pi / 26, pts_density_v=np.pi / 26), ellipsoid(6, 8, 10)),
                ("cylinder", bb.Cylinder(6, 12, pts_density_u=np.pi / 24, pts_density_h=0.35), cylinder(6, 6, 12, 0)),
                ("skewed elliptic cylinder", bb.Cylinder(6, 12, squeeze=0.6, skew=3, pts_density_u=np.pi / 24, pts_density_h=0.35), cylinder(6, 3.6, 12, 3)),
                ("4-sided prism", bb.Prism(8, 12, 4, pts_density_u=np.pi / 16, pts_density_h=0.35), prism(8, 12, 4, 0)),
                ("skewed 6-sided prism", bb.Prism(8, 12, 6, skew=3, pts_density_u=np.pi / 16, pts_density_h=0.35), prism(8, 12, 6, 3)),
                ("cone", bb.Cone(6, 12, pts_density_r=np.pi / 24, pts_density_h=0.35), cone(6, 12, 0)),
                ("skewed cone", bb.Cone(6, 12, skew=3, pts_density_r=np.pi / 24, pts_density_h=0.35), cone(6, 12, 3))]

    def test_convex_touching_points(self):

        print("\n> testing that convex shape points touch the nominal surface from inside")
        from unittest import mock

        # shapes are kept in the frame of their nominal body
        with mock.patch.object(bb.Structure, "center_to_origin"):
            shapes = self._convex_shapes()

        rng = np.random.default_rng(3)
        for name, C, dist in shapes:
            X = C.points[rng.choice(len(C.points), 200, replace=False)]
            np.testing.assert_allclose(dist(X), C.properties["pt_radius"], atol=1e-6, err_msg=name)

    def test_convex_ccs_numerical(self):

        print("\n> testing convex shape CCS against the projected area of their points")
        # projection approximation CCS of a point cloud: mean over directions of the area of the union of the projected discs.
        # Each projection is cut in rows spaced by dy, and the union of the chords of every row is measured exactly.
        def projected_ccs(points, rho, dirs, dy=0.1):
            k = int(np.ceil(2 * rho / dy)) + 2
            areas = []
            for u in dirs:
                e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(u, e1)
                X = np.dot(points, e1)
                Y = np.dot(points, e2)
                Y = Y - Y.min() + rho
                rows = np.floor((Y - rho) / dy).astype(int)[:, None] + np.arange(k)[None]
                d2 = rho**2 - ((rows + 0.5) * dy - Y[:, None])**2
                m = d2 > 0
                half = np.sqrt(d2[m])
                x = np.broadcast_to(X[:, None], rows.shape)[m]
                # rows are placed one after the other along a single line, then chords are sorted by their start
                shift = rows[m] * (2 * (np.abs(X).max() + rho) + 1)
                order = np.argsort(x - half + shift)
                start = (x - half + shift)[order]
                end = (x + half + shift)[order]
                reach = np.concatenate([[-np.inf], np.maximum.accumulate(end)[:-1]])
                areas.append(np.sum(np.clip(end - np.maximum(start, reach), 0, None)) * dy)
            return np.mean(areas)

        dirs = self._golden_spiral(100)
        gas = 1.0
        for name, C, _ in self._convex_shapes():
            rho = C.properties["pt_radius"] + gas
            numerical = projected_ccs(C.points, rho, dirs)
            analytical = C.ccs(gas=gas)
            print("  %-25s %6d points, CCS analytical %8.2f, numerical %8.2f A^2 (%+.2f%%)" % (name, len(C.points), analytical, numerical, 100 * (numerical / analytical - 1)))
            # the union of the point spheres lies inside the body the analytical CCS describes: the numerical value can exceed it only by
            # the row discretisation (+0.12% for a single disc of radius 2 A), and falls short by the gaps between points (0.1-0.3% here,
            # decreasing with the square of the point spacing). 100 directions average the projected area within about 0.05%
            self.assertLess(numerical / analytical, 1.001, msg=name)
            self.assertGreater(numerical / analytical, 0.995, msg=name)

    def test_convex_measures(self):

        print("\n> testing convex shape surfaces, volumes and CCS limits")
        # a sphere: points radius and gas both enlarge the nominal radius
        S = bb.Sphere(10, radius=1.9)
        for gas in [0.0, 1.0, 2.5]:
            self.assertAlmostEqual(S.ccs(gas=gas), np.pi * (10 + gas)**2, places=8)
        self.assertAlmostEqual(S.get_surface(), 4 * np.pi * 10**2, places=8)
        self.assertAlmostEqual(S.get_volume(), 4 * np.pi * 10**3 / 3, places=8)
        np.testing.assert_allclose(np.linalg.norm(S.points, axis=1), 10 - 1.9)

        # smooth shapes measure their nominal body: a prolate spheroid, as a squeezed sphere and as an ellipsoid
        a, b = 15.0, 6.0
        e = np.sqrt(1 - b**2 / a**2)
        surface = 2 * np.pi * b**2 * (1 + a / (b * e) * np.arcsin(e))
        E = bb.Ellipsoid(b, b, a)
        S = bb.Sphere(10, radius=1.9)
        S.squeeze([b / 10, b / 10, a / 10])
        for shape in [E, S]:
            self.assertAlmostEqual(shape.get_surface() / surface, 1.0, places=10)
            self.assertAlmostEqual(shape.get_volume(), 4 * np.pi * a * b * b / 3, places=8)
            # integral of mean curvature of a prolate spheroid: 2 pi int_{-1}^{1} sqrt(b^2 + (a^2 - b^2) t^2) dt = 2 pi (a + b^2 asinh(c / b) / c), c = sqrt(a^2 - b^2)
            c = np.sqrt(a**2 - b**2)
            M = 2 * np.pi * (a + b**2 * np.arcsinh(c / b) / c)
            self.assertAlmostEqual(shape.ccs(gas=1.0) / ((surface + 2 * M + 4 * np.pi) / 4), 1.0, places=10)

        # other shapes: the body traced by the points has the measures of the hull of the points centres enlarged by their radius (Steiner formula).
        # The hull is exact for the prism, and inscribed in curved sides (relative error about 1e-5 with these densities)
        from scipy.spatial import ConvexHull
        shapes = [bb.Prism(10, 20, 6, skew=4, pts_density_u=np.pi / 64, pts_density_h=50),
                  bb.Cylinder(10, 20, squeeze=0.7, skew=-3, pts_density_u=np.pi / 256, pts_density_h=50),
                  bb.Cone(10, 20, skew=5, pts_density_r=np.pi / 256, pts_density_h=50)]
        for C in shapes:
            hull = ConvexHull(C.points)
            nrm = hull.equations[:, :3]
            M = 0
            for i, simplex in enumerate(hull.simplices):
                for k, j in enumerate(hull.neighbors[i]):
                    if j > i:
                        edge = hull.points[np.delete(simplex, k)]
                        M += 0.5 * np.linalg.norm(edge[0] - edge[1]) * np.arccos(np.clip(np.dot(nrm[i], nrm[j]), -1, 1))
            pr = C.properties["pt_radius"]
            self.assertAlmostEqual(C.get_surface() / (hull.area + 2 * M * pr + 4 * np.pi * pr**2), 1.0, delta=5e-5)
            self.assertAlmostEqual(C.get_volume() / (hull.volume + hull.area * pr + M * pr**2 + 4 * np.pi * pr**3 / 3), 1.0, delta=5e-5)
            self.assertAlmostEqual(C.ccs(gas=1) / ((hull.area + 2 * M * (pr + 1) + 4 * np.pi * (pr + 1)**2) / 4), 1.0, delta=5e-5)

        # squeezing is rebuilt from the nominal sphere, and preserves its volume
        S1 = bb.Sphere(10, radius=1.9)
        S1.translate(5, 0, 0)
        S1.squeeze(2)
        S2 = bb.Sphere(10, radius=1.9)
        S2.translate(5, 0, 0)
        S2.squeeze(2.0)
        S2.squeeze(2.0)
        np.testing.assert_allclose(S1.points, S2.points, atol=1e-12)
        self.assertEqual([S1.properties[k] for k in ["a", "b", "c"]], [20.0, 10 / np.sqrt(2), 10 / np.sqrt(2)])
        self.assertAlmostEqual(S1.get_volume(), 4 * np.pi * 10**3 / 3, places=8)
        S1.squeeze([1.0, 1.0, 1.0])
        np.testing.assert_allclose(S1.points, bb.Sphere(10, radius=1.9).points + [5, 0, 0], atol=1e-3)
        self.assertEqual(list(S1.check_inclusion(np.array([[5, 0, 9.9], [5, 0, 10.1]]))), [True, False])

        # points too large to touch the surface everywhere
        before = S1.points.copy()
        for build in [lambda: bb.Sphere(2, radius=2), lambda: S1.squeeze([0.3, 1.0, 1.0]), lambda: bb.Ellipsoid(3, 10, 10, radius=1.9),
                      lambda: bb.Cylinder(5, 2), lambda: bb.Cylinder(5, 20, squeeze=0.3), lambda: bb.Prism(2, 20, 3),
                      lambda: bb.Prism(10, 2, 6), lambda: bb.Cone(2, 3), lambda: bb.Cone(10, 20, skew=30, radius=4)]:
            with self.assertRaises(Exception):
                build()
        np.testing.assert_array_equal(S1.points, before)
        self.assertEqual(S1.properties["p1"], 1.0)

    def test_convex_inclusion_and_contact_ratio(self):

        print("\n> testing convex shape inclusion and Assembly contact ratio")
        # the nominal ellipsoid, centred at the current center of geometry, decides inclusion
        E = bb.Ellipsoid(6, 8, 10)
        E.translate(1, 2, 3)
        inside = E.check_inclusion(np.array([[1, 2, 3], [6.9, 2, 3], [7.1, 2, 3], [1, 2, 12.9], [1, 2, 13.1]]))
        self.assertEqual(inside.dtype, bool)
        self.assertEqual(list(inside), [True, True, False, True, False])

        # fraction of the points of the second unit inside the first one
        A = bb.Assembly()
        A.load_list([bb.Ellipsoid(6, 8, 10), bb.Structure(np.array([[0.0, 0, 0], [0, 0, 9], [0, 0, 11], [20, 0, 0]]))], ["E", "P"])
        self.assertEqual(A.contact_ratio("E", "P"), 0.5)
        self.assertIsInstance(A.contact_ratio("E", "P"), float)
        self.assertEqual(A.contact_ratio("E", "E"), 1.0)
        with self.assertRaisesRegex(Exception, "unit P is a Structure"):
            A.contact_ratio("P", "E")

    def test_global_grid_hull(self):

        print("\n> testing that the convex hull of a global grid follows the obstacles")
        from biobox.measures.path import Path
        # the swollen hull is centred on the obstacles, so translating them does not change the grid
        counts = []
        for shift in [0.0, 50.0]:
            P = Path(self.M.points + shift)
            P.setup_global_search(step=1.0, use_hull=True)
            counts.append(int(P.graph.access_grid.sum()))
        self.assertLess(abs(counts[0] - counts[1]), 0.01 * counts[0])


if __name__ == '__main__':
    unittest.main()
