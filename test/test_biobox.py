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
