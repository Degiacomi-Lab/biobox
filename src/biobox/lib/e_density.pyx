# Author: Lucas Rudden, l.s.rudden@durham.ac.uk

import os
from copy import deepcopy
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

cpdef np.ndarray c_get_dipole_map(np.ndarray crd, np.ndarray orig, np.ndarray charges, int time_start = 0, int time_end = 2, float resolution = 1.0, float vox_in_window = 3, bool write_dipole_map = False, str fname = "dipole_map.tcl"):
    '''
    Generate a vector (x, y, z) of instantaneous dipole moments, for every frame from time_start to time_end-1,
    within voxels centred on the grid defined by orig.
    For every voxel centre c and frame, the dipole is the sum of q_i*(r_i - c) over the atoms i whose coordinates
    lie in [c - window_size/2, c + window_size/2) along each axis, where window_size = resolution*vox_in_window.
    Voxels containing no atom have a zero dipole.

    Orig should be built in a separate function that looks at the entirety of the multipdb to account for atomic
    coordinates outside our current investigated bounds (and to keep the number of voxels the same for different
    cartisian sized systems).

    :param crd: coordinates of all frames (numpy array of shape frames x atoms x 3, in A). Given by bb.molecule.coordinates
    :param orig: voxel centres, as three 1D numpy arrays with the x, y and z grid coordinates (in A). Must be constant across timeframes
    :param charges: numpy array with the partial charge of every atom, in units of e (e.g. from a PQR file, see pdb2pqr)
    :param time_start: Start frame for finding the dipole map
    :param time_end: End frame for finding the dipole map, excluded (for just 1 frame, it needs to be one more than time_start)
    :param resolution: voxel size, in A
    :param vox_in_window: width of the window around each voxel centre in which atoms contribute to its dipole, in voxels.
        Should account for electrostatics falling to zero (or close) at the boundaries.
    :param write_dipole_map: Boolean. If true, write a tcl file of VMD "draw cone" commands, to be read in with VMD command: source dipole_map.tcl.
        Each cone goes from a voxel centre to the centre plus its dipole averaged over frames, and is written only for voxels where the magnitude of this averaged dipole exceeds 0.7 e*A.
    :param fname: Name of dipole_map tcl file.
    :returns: float32 numpy array of shape (frames, nx, ny, nz, 3), with nx, ny, nz the number of orig points in x, y and z: dipole vector of every voxel in every frame, in e*A
    '''

    window_size = resolution * vox_in_window
    time_val = np.arange(time_start, time_end) # Create range of frames for us to explore depending on user input. (Default is just the first 2)
    
    if write_dipole_map:
        data_file = open(fname, "w") # open a file for writing to
        #data_file.write("draw material Diffuse\n")
    
    x_range = orig[0] - window_size / 2.   # Create shifted coordinates to account for start of windows
    y_range = orig[1] - window_size / 2.
    z_range = orig[2] - window_size / 2.

    x_fill = np.zeros((len(z_range) * len(y_range), 3)).tolist() # prep empty arrays in case we don't find any atoms in our loops
    y_fill = np.zeros((len(z_range), 3)).tolist()
    z_fill = [[0., 0., 0.]]
 
    dipole_map = []
    for it in time_val:     # it for i in time
        
        D = crd[it]
        dipole_snapshot = []
    
        for ix, x_item in enumerate(x_range):
            x_test1 = x_item <= D[:,0]
            x_test2 = D[:,0] < x_item + window_size
            x_where = np.logical_and(x_test1, x_test2)

            xslice = D[x_where]
        
            if not xslice.size:    # append dipole of 0s along axis if no atoms are found within this slice
                dipole_snapshot.extend(x_fill)   # multiply by these lengths to account for grid for first z
                continue
       
            else:
                chargex_slice = charges[x_where]
                for iy, y_item in enumerate(y_range):
                    y_test1 = y_item <= xslice[:,1]
                    y_test2 = xslice[:,1] < y_item + window_size
                    y_where = np.logical_and(y_test1, y_test2)

                    yslice = xslice[y_where]
            
                    if not yslice.size:

                        dipole_snapshot.extend(y_fill)   # Again, account for what we're about to skip
                        continue
            
                    else:
                        chargey_slice = chargex_slice[y_where]
                        for iz, z_item in enumerate(z_range):

                            z_test1 = z_item <= yslice[:,2]
                            z_test2 = yslice[:,2] < z_item + window_size
                            z_where = np.logical_and(z_test1, z_test2)

                            coord = yslice[z_where] 
                            charge_slice = chargey_slice[z_where]
                            
                            if not coord.size:
                                dipole_snapshot.extend(z_fill)   # Again, account for what we're about to skip
                                continue
                        
                            else:

                                # We take our centre point as the centre of the voxel box
                                x_diff = charge_slice * (coord[:,0] - orig[0][ix]) # calculate the displacements in x,y,z 
                                y_diff = charge_slice * (coord[:,1] - orig[1][iy])
                                z_diff = charge_slice * (coord[:,2] - orig[2][iz])
                                 
                                #now units are in C m
                                dipole = [[np.sum(x_diff, axis=0), np.sum(y_diff, axis=0), np.sum(z_diff, axis=0)]]
                                dipole_snapshot.extend(dipole)
                                   
            
        #print np.shape(dipole_snapshot)
        dipole_snapshot = np.array(dipole_snapshot).astype(np.float32)
        dipole_snapshot = np.reshape(dipole_snapshot, (len(x_range), len(y_range), len(z_range), 3))  # Reshape as necessary size to match coordinate system
        dipole_map.append(dipole_snapshot) # Create dipole_map over time
        #dipole_map.append(np.reshape(np.zeros((len(z_range) * len(y_range) * len(x_range), 3)).tolist(), (len(x_range), len(y_range), len(z_range), 3)))
      
    if write_dipole_map:  
        dip_avg = np.mean(np.array(dipole_map), axis=0)
        for ix in range(np.shape(dip_avg)[0]):
                for iy in range(np.shape(dip_avg)[1]):
                    for iz in range(np.shape(dip_avg)[2]):
                        if np.sqrt(dip_avg[ix][iy][iz][0]**2 + dip_avg[ix][iy][iz][1]**2 + dip_avg[ix][iy][iz][2]**2) > 0.7:
                            dip_x = orig[0][ix] + dip_avg[ix][iy][iz][0]
                            dip_y = orig[1][iy] + dip_avg[ix][iy][iz][1]
                            dip_z = orig[2][iz] + dip_avg[ix][iy][iz][2]
                            data_file.write("draw cone { %f %f %f } { %f %f %f } radius 0.3\n"%(orig[0][ix], orig[1][iy], orig[2][iz], dip_x, dip_y, dip_z))
                        else:
                            continue
        data_file.close() 

    return np.array(dipole_map).astype(np.float32)

def _add_clipped_kernel(pts, kernel, ix, iy, iz, half):
    '''
    Add a kernel centred on voxel (ix, iy, iz) to pts, keeping only the part of the kernel that lies inside the grid.

    :param pts: 3D numpy array the kernel is added to, in place
    :param kernel: 3D numpy array of shape (2*half+1, 2*half+1, 2*half+1)
    :param ix: x index of the voxel the kernel is centred on
    :param iy: y index of the voxel the kernel is centred on
    :param iz: z index of the voxel the kernel is centred on
    :param half: half width of the kernel, in voxels
    '''
    grid_slices = []
    kernel_slices = []
    for i, n in zip((ix, iy, iz), pts.shape):
        lo = max(i - half, 0)
        hi = min(i + half + 1, n)
        grid_slices.append(slice(lo, hi))
        kernel_slices.append(slice(lo - (i - half), hi - (i - half)))
    pts[tuple(grid_slices)] += kernel[tuple(kernel_slices)]

cpdef int c_get_dipole_density(np.ndarray dipole_map, np.ndarray orig, list min_val, float V, str outname, float vox_in_window = 3., str eqn = 'gauss', float T = 310.15, float P = 101 * 1E+3, float epsilonE = 54., float resolution = 1.0):
    '''
    This generates an electron density based on a dipole map obtained with get_dipole_map. It requires the same coordinate system, orig, as
    said dipole map. It is based on a paper by Pitera et al. written in 2001: 
            
    Dielectric properties of proteins from simulation; The effects of solvent, ligands, pH and temperature.
    
    It also requires the approximation that polarisability can be defined using the permitivitty of local space, and subsequently a van der Waal
    object can also be defined in terms of polarisability, this relies on the Clausius-Mossotti relation between molecular
    polarisability and dielectric constant.

    The dipole fluctuations of every voxel give its dielectric permittivity (clamped to a minimum of 1), hence a polarisability
    and a van der Waals radius, which sets the width sigma (r_vdw/(2*sqrt(2*ln 2)), i.e. r_vdw is the FWHM of the Gaussian)
    of a function centred on the voxel. The sum of these functions, normalised
    to a maximum of 1, is written as a dx file. Near the edges of the grid, only the part of each function lying inside the grid is added.

    :param dipole_map: Dimensions of (t, x, y, z, [v_x, v_y, v_z]) where [v_x, v_y, v_z] is the vector dipole values (in e*A) for points x, y, z at time t. At least 2 frames are required.
    :param orig: Coordinate system (x, y, z) we measure our dipole from. MUST be the same as that used in get_dipole_map
    :param min_val: Minimum coordinates (x, y, z) from which to define our origin, used as the origin of the dx file. Wrong choice could cause a shift in real space of the density.
    :param V: The partial specific volume for the protein (worth investigating further). Units of m^3.
    :param outname: Filename for output dx file.
    :param vox_in_window: Width of the sliding window in voxels. Each voxel's function is sampled at whole-voxel offsets within half this width of the voxel centre, i.e. 2*floor(vox_in_window/2)+1 points per axis
    :param eqn: Type of equation used for convolution. Options are 'gauss' (Gaussian) and 'slater' (Slater)
    :param T: Temperature of simulation. Default is body temp (K).
    :param P: Pressure of simulation (Pa). Not used in the calculation.
    :param epsilonE: External relative permittivity outside the protein. Another variable worth investigating. Default is from 2001 paper regarding a salt water solvent.
    :param resolution: voxel size in A, setting the spacing of the sampled functions and of the dx grid. Should be the same as in get_dipole_map
    :returns: 0, once the dx file is written
    :raises ValueError: if eqn is not 'gauss' or 'slater', or if no voxel has a dipole fluctuation, so that the density is zero everywhere
    '''
    if eqn not in ('gauss', 'slater'):
        raise ValueError("eqn must be 'gauss' or 'slater', got %r"%(eqn,))

    window_size = resolution * vox_in_window
    test = dipole_map.shape

    cdef float polar_au = 1.6487772731 * 1E-41 # C^2 m^2 J^-1 - conversion from real to atomic units for polarisability
    cdef float dist_au = 5.29177 * 1E-11 # m - one bohr unit, convert from real to a. u. for distance
    cdef float epsilon0, kB, e, m , Na

    epsilon0 = 8.8542 * 1E-12 # m**-3 kg**-1 s**4 A**2, Permitivitty of free space
    kB = 1.3806 * 1E-23 # m**2 kg s**-2 K-1, Lattice Boltzmann constant
    e = 1.602 * 1E-19 # A s, electronic charge
    m = 1. * 1E-10 # number of m in 1A
    Na = 6.022 * 1E+23 # Avagadros Number
    
    if test[0] < 2:
        raise Exception("ERROR: The number of frames in your dipole map is %i. 2 or more are required for electron density calculations."%(test[0]))
    
    #print("What function would you like to use? Please enter a number.\n1. Gaussian: exp(-(x**2 + y**2 + z**2) / 2 * sigma)\n2. Slater: exp(-(x**2 + y**2 + z**2)**(1./2.) / 2 * sigma)")
    #eqn = input()
    #if eqn != 1 or 2:
    #    print("ERROR: You did not enter a valid number for your choice of equation\n Defaulting to Gaussian.")
    #    eqn = 1
            
    # Depending on the size of the system and user RAM, we need to try two slightly different methods to avoid memory issues.
    try:
        p_M = np.sum(np.mean(np.power(dipole_map, 2.), axis=0) - np.power(np.mean(dipole_map, axis=0), 2), axis=3)
        
        p_M = np.array(p_M).astype(np.float64) * e**2 * m**2 # Unit conversion

        # Now we need to define epsilon_r (the dielectric permitivitty)
        val = p_M / (3. * epsilon0 * V * kB * T)
  
        epsilon_top = 1. + (val * ((2. * epsilonE) / (2. * epsilonE + 1.)))
        epsilon_bot = 1. - (val * (1. / (2. * epsilonE + 1.)))
        epsilon = epsilon_top / epsilon_bot
        epsilon[np.where(epsilon < 1.0)] = 1.0
      
        # Now we want to calculate our van der waals volume based on the Claussius-Moletti relation between polarisability and permitivitty.

        # BASED ON PAPER OUT END OF MARCH 2018 ON LINK VIA QM, Rvdw = 0.24 alpha^(1/7), derived from noble gases - Quantum approximation
        # Find polarisability and convert to atomic units - alpha = 3 eps0 / N * (eps - 1 / eps + 2)
        alpha_au = ((3 * epsilon0 * V) * ((epsilon -1) / (epsilon + 2))) / polar_au

        r_au = 2.54 * alpha_au**(1. / 7.) # convert to vdw radius in a. u. based on quantum paper

        r_vdw = r_au * dist_au # convert back to m

        sigma = r_vdw / (2. * np.sqrt(2. * np.log(2.))) # Setting r_vdw equal to the FWHM of our gaussian / Lorentz / Slater function.
    
        pts = np.zeros((len(orig[0]), len(orig[1]), len(orig[2])))

        sigma = sigma / m  # convert back into A units to match with x, y, z coord used in meshgrid below
    
        # Create 3D function kernal

        # kernel sampled at whole-voxel offsets from the voxel centre, within the window
        # get_dipole_map uses for each voxel, [-window_size/2, window_size/2]
        half = int(np.floor(vox_in_window / 2. + 1e-6))
        mesh = np.arange(-half, half + 1) * resolution

        x, y, z = np.meshgrid(mesh, mesh, mesh, indexing='ij')
        r2 = x * x + y * y + z * z

        sigmanonzero = np.nonzero(sigma) #  Get only contributing sigmas for faster calculations.
        for i in range(np.shape(sigmanonzero)[1]):
            ix, iy, iz = sigmanonzero[0][i], sigmanonzero[1][i], sigmanonzero[2][i]
            if eqn == 'gauss':
                gauss = np.exp(-r2 / (2. * sigma[ix][iy][iz]**2))   # Create gaussian with specific sigma from e density
            elif eqn == 'slater':
                gauss = np.exp(-np.sqrt(r2) / (2. * sigma[ix][iy][iz]**2))   # Create Slater functional with specific sigma from e density
            _add_clipped_kernel(pts, gauss, ix, iy, iz, half)
    
    except MemoryError:
        print("Size of protein is too large for electron density map production. Breaking calculations down into smaller chunks (may take longer, or not work if data structure too big).\n")
        
        # Too much to handle! We'll have to create a loop to slim down the large arrays. Let's make the loop in x (second set of indices).
        p_M = []
        for i in range(np.shape(dipole_map)[1]):
            fluc = np.sum(np.mean(np.power(dipole_map[:,i], 2.), axis=0) - np.power(np.mean(dipole_map[:,i], axis=0), 2), axis=2)
            p_M.append(fluc)
        p_M = np.array(p_M).astype(np.float64) * e**2. * m**2. # convert units
        
        # Now we need to define epsilon_r (the dielectric permitivitty)
        val = p_M / (3. * epsilon0 * V * kB * T)
            
        epsilon_top = 1. + (val * ((2. * epsilonE) / (2. * epsilonE + 1.)))
        epsilon_bot = 1. - (val * (1. / (2. * epsilonE + 1.)))
        epsilon = epsilon_top / epsilon_bot
        epsilon[np.where(epsilon < 1.0)] = 1.0

        # Now we want to calculate our van der waals volume based on the Claussius-Moletti relation between polarisability and permitivitty.
        #Vvdw = (kB * T * epsilon0 * (epsilon - 1.) * 3.) / (4. * np.pi * P * (epsilon + 2.))  # Hard sphere approximation
        #r_vdw = ((3. * Vvdw) / (4. * np.pi))**(1./3.)

        # BASED ON PAPER OUT END OF MARCH 2018 ON LINK VIA QM, Rvdw = 0.24 alpha^(1/7), derived from noble gases - Quantum approximation
        # Find polarisability and convert to atomic units - alpha = 3 eps0 / N * (eps - 1 / eps + 2)
        alpha_au = ((3 * epsilon0 * V) * ((epsilon -1) / (epsilon + 2))) / polar_au

        r_au = 2.54 * alpha_au**(1. / 7.) # convert to vdw radius in a. u. based on quantum paper

        r_vdw = r_au * dist_au # convert back to m
            
        sigma = r_vdw / (2. * np.sqrt(2. * np.log(2.))) # Setting r_vdw equal to the FWHM of our gaussian / Lorentz / Slater function.
    
        pts = np.zeros((len(orig[0]), len(orig[1]), len(orig[2])))

        sigma = sigma / m  # convert back into nm units to match with x, y, z coord used in meshgrid below

        # Create 3D function kernal

        # kernel sampled at whole-voxel offsets from the voxel centre, within the window
        # get_dipole_map uses for each voxel, [-window_size/2, window_size/2]
        half = int(np.floor(vox_in_window / 2. + 1e-6))
        mesh = np.arange(-half, half + 1) * resolution

        x, y, z = np.meshgrid(mesh, mesh, mesh, indexing='ij')
        r2 = x * x + y * y + z * z

        sigmanonzero = np.nonzero(sigma) #  Get only contributing sigmas for faster calculations.
        for i in range(np.shape(sigmanonzero)[1]):
            ix, iy, iz = sigmanonzero[0][i], sigmanonzero[1][i], sigmanonzero[2][i]
            if eqn == 'gauss':
                gauss = np.exp(-r2 / (2. * sigma[ix][iy][iz]**2))   # Create gaussian with specific sigma from e density
            elif eqn == 'slater':
                gauss = np.exp(-np.sqrt(r2) / (2. * sigma[ix][iy][iz]**2))   # Create Slater functional with specific sigma from e density
            _add_clipped_kernel(pts, gauss, ix, iy, iz, half)
    
    # prepare density structure export
  
    if pts.max() <= 0:
        raise ValueError("no voxel of the dipole map fluctuates over the frames, so the density is zero everywhere and no dx file is written")
    pts /= pts.max()

    from biobox.classes.density import Density
        
    D = Density()

    D.properties['density'] = pts #epsilon #pts 
    D.properties['size'] = np.array(pts.shape) #epsilon.shape #np.array(pts.shape) 
    D.properties['origin'] = np.array(min_val)  
    D.properties['delta'] = np.identity(3) * resolution #(step size)
    D.properties['format'] = 'dx'
    D.properties['filename'] = ''
    D.properties['sigma'] = np.std(pts) #np.std(epsilon) #np.std(pts) 
    
    D.write_dx(outname)

    return 0