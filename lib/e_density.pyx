# Author: Lucas Rudden, l.s.rudden@durham.ac.uk

import os
from copy import deepcopy
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

cpdef np.ndarray c_get_dipole_map(np.ndarray crd, np.ndarray orig, np.ndarray charges, int time_start = 0, int time_end = 2, float resolution = 1.0, float vox_in_window = 3, bool write_dipole_map = False, str fname = "dipole_map.tcl"):
    '''
    Generate a vector (x, y, z) of instantaneous dipole moments at time_val within voxels centred at orig. 
    The size of the voxels is governed by the number of orig points and the size of the system. In essence,
    orig contains the inherent desired shift for the sliding window.
       
    Orig should be built in a separate function that looks at the entirety of the multipdb to account for atomic 
    coordinates outside our current investigated bounds (and to keep the number of voxels the same for different
    cartisian sized systems).
    
    :param crd: coordinate system. Given by bb.molecule.coordinates
    :param orig: Origin of voxels from which we'll find our dipole for (this must be constant across timeframes)
    :param pqr: PQR converted file of PDB file. See pdb2pqr for more details.
    :param time_start: Start frame for finding the dipole map
    :param time_end: End frame for finding the dipole map (for just 1 frame, it needs to be one more than time_start)
    :param window_size: Size of the window we're calculating dipole moments for. Should account for electrostatics falling to zero
    (or close) at the boundaries. Shouldn't be too small otherwise the memory demand will be too high. 1 nm is default in +- x, y, z.
    :param write_dipole_map: Boolean. If true, write a dipole map for time_val in the tcl format to be read in with VMD command: source dipole_map.tcl
    :param fname: Name of dipole_map tcl file.
    :returns: Vector map of dipoles in x, y and z in the shape of the no. or orig points in x, y and z.
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
                        if np.sqrt(dip_avg[ix][iy][iz][0]**2 + dip_avg[ix][iy][iz][0]**2 + dip_avg[ix][iy][iz][0]**2) > 0.7:
                            dip_x = orig[0][ix] + dip_avg[ix][iy][iz][0]
                            dip_y = orig[1][iy] + dip_avg[ix][iy][iz][1]
                            dip_z = orig[2][iz] + dip_avg[ix][iy][iz][2]
                            data_file.write("draw cone { %f %f %f } { %f %f %f } radius 0.3\n"%(orig[0][ix], orig[1][iy], orig[2][iz], dip_x, dip_y, dip_z))
                        else:
                            continue
        data_file.close() 

    return np.array(dipole_map).astype(np.float32)

cpdef int c_get_dipole_density(np.ndarray dipole_map, np.ndarray orig, list min_val, float V, str outname, float vox_in_window = 3., str eqn = 'gauss', float T = 310.15, float P = 101 * 1E+3, float epsilonE = 54., float resolution = 1.0):
    '''
    This generates an electron density based on a dipole map obtained with get_dipole_map. It requires the same coordinate system, orig, as
    said dipole map. It is based on a paper by Pitera et al. written in 2001: 
            
    Dielectric properties of proteins from simulation; The effects of solvent, ligands, pH and temperature.
    
    It also requires the approximation that polarisability can be defined using the permitivitty of local space, and subsequently a van der Waal
    object can also be defined in terms of polarisability, this relies on the Clausius-Moletti relation between molecular
    polarisability and dielectric constant. 
    
    :param dipole_map: Dimensions of (t, x, y, z, [v_x, v_y, v_z]) where [v_x, v_y, v_z] is the vector dipole values for points x, y, z at time t.
    :param orig: Coordinate system (x, y, z) we measure our dipole from. MUST be the same as that used in get_dipole_map
    :param min_val: Minimum coorinates (x, y, z) from which to define our origin. Wrong choice could cause a shift in real space of the density.
    :param vox_in_window: Number of voxels to define our gaussian from. Essentially the number of voxels within our sliding window
    :param V: The partial specific volume for the protein (worth investigating further). Units of m^3.
    :param outname: Filename for output dx file.
    :param eqn: Type of equation used for convolution. OPtions are Gaussian, Slater or Lorentzian
    :param T: Temperature of simulation. Default is body temp (K).
    :param P: Pressure of simulation. Default is atmospheric (Pa).
    :param epsilonE: External permitivitty outside the protein. Another variable worth investigating. Default is from 2001 paper regarding a salt water solvent.
    :param window_size: Size of the window we're calculating dipole moments for. Should account for electrostatics falling to zero and be same as get_dipole_map
    '''    
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

        mesh = np.arange(0, window_size+0.0001, window_size/vox_in_window) - window_size / 2.
     
        x, y, z = np.meshgrid(mesh, mesh, mesh)

        # We should have a buffer (default is 2 * window_size) at the edges of our box, so should be able to sum contributing gaussians across entire system.
        sigmanonzero = np.nonzero(sigma) #  Get only contributing sigmas for faster calculations.
        if len(mesh) % 2. == 0.: # Number is even (no point doing this check later - saves time)
            for i in range(np.shape(sigmanonzero)[1]):
                if eqn == 'gauss':
                    gauss = np.exp(-(x * x + y * y + z * z) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create gaussian with specific sigma from e density
                elif eqn == 'slater':
                    gauss = np.exp(-np.sqrt((x * x + y * y + z * z)) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create Slater functional with specific sigma from e density
                pts_range = int(((len(mesh) + 1.) / 2.))
                pts[sigmanonzero[0][i] - pts_range : sigmanonzero[0][i] + pts_range ,
                    sigmanonzero[1][i] - pts_range : sigmanonzero[1][i] + pts_range ,
                    sigmanonzero[2][i] - pts_range : sigmanonzero[2][i] + pts_range ] += gauss # move 1 ahead duye to python numbering
        else:  # number is odd
            for i in range(np.shape(sigmanonzero)[1]):
                if eqn == 'gauss':
                    gauss = np.exp(-(x * x + y * y + z * z) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create gaussian with specific sigma from e density
                elif eqn == 'slater':
                    gauss = np.exp(-np.sqrt((x * x + y * y + z * z)) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create Slater functional with specific sigma from e density
                pts_range = int(((len(mesh)) / 2.))
                pts[sigmanonzero[0][i] - pts_range : sigmanonzero[0][i] + pts_range + 1,
                    sigmanonzero[1][i] - pts_range : sigmanonzero[1][i] + pts_range + 1,
                    sigmanonzero[2][i] - pts_range : sigmanonzero[2][i] + pts_range + 1] += gauss # move 1 ahead duye to python numbering
    
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

        #mesh = np.linspace(0, window_size, vox_in_window) - window_size / 2.
        mesh = np.arange(0, window_size+0.0001, window_size/vox_in_window) - window_size / 2.
            
        x, y, z = np.meshgrid(mesh, mesh, mesh)

        # We should have a buffer (default is 2 * window_size) at the edges of our box, so should be able to sum contributing gaussians across entire system.
        sigmanonzero = np.nonzero(sigma) #  Get only contributing sigmas for faster calculations.
        if len(mesh) % 2. == 0.: # Number is even (no point doing this check later - saves time)
            for i in range(np.shape(sigmanonzero)[1]):
                if eqn == 'gauss':
                    gauss = np.exp(-(x * x + y * y + z * z) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create gaussian with specific sigma from e density
                elif eqn == 'slater':
                    gauss = np.exp(-np.sqrt((x * x + y * y + z * z)) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create Slater functional with specific sigma from e density
                pts_range = int(((len(mesh) + 1.) / 2.))
                pts[sigmanonzero[0][i] - pts_range : sigmanonzero[0][i] + pts_range ,
                    sigmanonzero[1][i] - pts_range : sigmanonzero[1][i] + pts_range ,
                    sigmanonzero[2][i] - pts_range : sigmanonzero[2][i] + pts_range ] += gauss # move 1 ahead duye to python numbering
        else:  # number is odd
            for i in range(np.shape(sigmanonzero)[1]):
                if eqn == 'gauss':
                    gauss = np.exp(-(x * x + y * y + z * z) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create gaussian with specific sigma from e density
                elif eqn == 'slater':
                    gauss = np.exp(-np.sqrt((x * x + y * y + z * z)) / (2. * sigma[sigmanonzero[0][i]][sigmanonzero[1][i]][sigmanonzero[2][i]]**2))   # Create Slater functional with specific sigma from e density
                pts_range = int(((len(mesh)) / 2.))
                pts[sigmanonzero[0][i] - pts_range : sigmanonzero[0][i] + pts_range + 1,
                    sigmanonzero[1][i] - pts_range : sigmanonzero[1][i] + pts_range + 1,
                    sigmanonzero[2][i] - pts_range : sigmanonzero[2][i] + pts_range + 1] += gauss # move 1 ahead duye to python numbering
    
    # prepare density structure export
  
    pts /= pts.max()

    from biobox.classes.density import Density
        
    D = Density()

    D.properties['density'] = pts
    D.properties['size'] = np.array(pts.shape)
    D.properties['origin'] = np.array(min_val)  
    D.properties['delta'] = np.identity(3) * resolution #(step size)
    D.properties['format'] = 'dx'
    D.properties['filename'] = ''
    D.properties['sigma'] = np.std(pts)
    
    D.write_dx(outname)

    return 0


