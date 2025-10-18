from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import os

#---------------------------- Options for fit ----------------------------------

#The input data
data_file = 'isochrone_fitting/isochrones_grid_subgiants.fits'
# Name the fits file where results will be stored 
output_file = 'isochrone_fitting/testing.fits'

# Choose to fit to observational data or isochrone models 
# Set to True to fit ages to Gaia data
Gaia_fit = False
# Set to True to fit ages to isochone models 
model_fit = True

# Set which iron abundance to use for Gaia fit (gspphot, gspspec, cannon_cal)
feh_type = 'gspspec'

# Set uncertainties for isochrone model fit 
mag_err = 0.02
bprp_err = 0.005
feh_err = 0.08

# toggle to plot the chi2 
plot_chi2 = False


# Parameters to make the fit better
min_log_age = 7.5

# Load fits file with isochrone models
ff = fits.open('isochrone_models.fits')

#----------- Change stuff above here ------------------
# Index of minimum log age
min_age_ix = np.where(ff['age_grid'].data >= min_log_age)[0][0]

# Extract the individual data cubes from fits file
fractional_mass_grid = ff['fractional_mass_grid'].data
metallicity_grid = ff['metallicity_grid'].data
age_grid = ff['age_grid'].data


# Extract interpolated parameters from the fits file
log_Teff = ff['log_Teff'].data
log_L = ff['log_L'].data
star_mass_max = ff['star_mass_max'].data
Gaia_G_EDR3 = ff['Gaia_G_EDR3'].data
Gaia_BP_EDR3 = ff['Gaia_BP_EDR3'].data
Gaia_RP_EDR3 = ff['Gaia_RP_EDR3'].data

# Close the fits file
ff.close()

# Create interpolators for the parameters. First, 3D interpolators, use cubic method to avoid preferential fitting to the isochrone ages
Gaia_G_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]),         
                                    Gaia_G_EDR3[:,:,min_age_ix:],bounds_error=False, method = 'cubic')
Gaia_BP_RP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    Gaia_BP_EDR3[:,:,min_age_ix:] - Gaia_RP_EDR3[:,:,min_age_ix:], bounds_error=False, method = 'cubic')

#Now a 2D interpolator for the maximum star mass
star_mass_max_interp = RegularGridInterpolator((metallicity_grid, age_grid[min_age_ix:]), star_mass_max[:,min_age_ix:],bounds_error=False)

# Create funtion that returns the Gaia G, BP and RP magnitudes for a given mass, metallicity and age.
def get_Gaia_magnitudes(mass, metallicity, age):
    # Calculate the maximum star mass for the given metallicity and age
    max_star_mass = star_mass_max_interp((metallicity, age))
    
    # Find the corresponding fractional mass for the given mass by interpolating
    fractional_mass = mass / max_star_mass

    # Interpolate to get the Gaia G, BP and RP magnitudes for the given mass, metallicity and age
    Gaia_G_EDR3 = Gaia_G_EDR3_interp((fractional_mass, metallicity, age))
    Gaia_BP_RP_EDR3 = Gaia_BP_RP_EDR3_interp((fractional_mass, metallicity, age))
    #Gaia_BP_EDR3 = Gaia_BP_EDR3_interp((fractional_mass, metallicity, age))
    #Gaia_RP_EDR3 = Gaia_RP_EDR3_interp((fractional_mass, metallicity, age))
    return float(Gaia_G_EDR3), float(Gaia_BP_RP_EDR3)   #, float(Gaia_BP_EDR3), float(Gaia_RP_EDR3)

# Create funtion to return the normalised difference between the observed and modelled Gaia G, BP and RP magnitudes
def residuals_Gaia(params, observed_Gaia_G_EDR3, observed_Gaia_BP_RP_EDR3, Gaia_G_EDR3_err, Gaia_BP_RP_EDR3_err, metallicity, metallicity_err):
    
    # Get the modelled Gaia G, BP and RP magnitudes for the given mass, metallicity and age
    model_Gaia_G_EDR3, model_Gaia_BP_RP_EDR3 = get_Gaia_magnitudes(*params)

    # Calculate the residuals
    residuals = np.array([(observed_Gaia_G_EDR3 - model_Gaia_G_EDR3)/Gaia_G_EDR3_err, (observed_Gaia_BP_RP_EDR3 - model_Gaia_BP_RP_EDR3)/Gaia_BP_RP_EDR3_err,
                          (params[1] - metallicity)/metallicity_err])
    return residuals.flatten()

#Define function to get initial guess from metallicities
def initial_guess(metallicity_obs, G_mag_obs, BP_RP_mag_obs, G_mag_err, BP_RP_mag_err):

    # Set up mass grid fractional 
    # Only include stars that are roughly in the right part of the HR diagram
    # NB - this will avoid considering young stars.
    frac_mass_grid = np.linspace(0.9, 1, 50)# change back to 200 for proper fitting
    log_age_grid = np.linspace(7.5, max(age_grid), 50)

    # Initialise 2d array to hold chi2 values for the grid search
    mass_age_chi2_array = np.zeros((len(frac_mass_grid), len(log_age_grid)))

    # Run grid search over mass and age for the given observed magnitudes as metallicity
    for j, age in enumerate(log_age_grid):
        
        # Find photometry of this age and mass
        G_mag_model = Gaia_G_EDR3_interp((frac_mass_grid, metallicity_obs, age))
        BP_RP_mag_model = Gaia_BP_RP_EDR3_interp((frac_mass_grid, metallicity_obs, age))

        # Find sum of chi2 values of this model
        chi2 = ((G_mag_obs-G_mag_model)/G_mag_err)**2 + ((BP_RP_mag_obs-BP_RP_mag_model)/BP_RP_mag_err)**2 

        # Add chi2 value to 2d array 
        mass_age_chi2_array[:,j] = chi2
    
    # plot the chi2 array 
    if plot_chi2 == True:
        from matplotlib.colors import LogNorm

        #Helper function for plotting chi2 grids
        def get_unique_filename(base_name, ext):
            """
            If 'base_name.ext' exists, appends _1, _2, etc. until unique.
            Returns full filename as a string.
            """
            filename = f"{base_name}{ext}"
            counter = 1
            while os.path.exists(filename):
                filename = f"{base_name}_{counter}{ext}"
                counter += 1
            return filename

        plt.figure(figsize=(6, 5))
        plt.imshow(mass_age_chi2_array.T, 
                origin='lower', 
                aspect='auto', 
                extent=[min(mass_grid), max(mass_grid), min(log_age_grid), max(log_age_grid)],
                cmap='plasma_r', 
                norm=LogNorm())
        plt.colorbar(label=r'$\chi^2$')
        plt.xlabel('Mass ($M_\\odot$)')
        plt.ylabel('Log age (yr)')
        
        # save graph
        base_name = f"chi2_grids/chi2_M{min(mass_grid):.2f}-{max(mass_grid):.2f}_age{min(log_age_grid):.2f}-{max(log_age_grid):.2f}_{len(mass_grid)}mbins_{len(log_age_grid)}agebins"
        filename = get_unique_filename(base_name, ".png")

        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()

    # Find age and mass of minimum chi2 value
    # Find index
    min_index = np.unravel_index(np.nanargmin(mass_age_chi2_array), mass_age_chi2_array.shape)
    mass_idx, age_idx = min_index
    #chi2_min = mass_age_chi2_array[mass_idx, age_idx]

    # find corresponding mass and age values
    best_frac_mass = frac_mass_grid[mass_idx]
    best_log_age = log_age_grid[age_idx]

    # Convert best fractional mass to physical mass 
    best_mass = best_frac_mass * star_mass_max_interp((metallicity_obs, best_log_age))

    return (best_mass, metallicity_obs, best_log_age)

# Define function to fit age for give metallicity and Gaia magitudes with errors:
def fit_age(Fe_H, Fe_H_err, G, G_err, Bp_Rp, Bp_Rp_err):

    # get the initial guess from grid search 
    guess = initial_guess(Fe_H, G, Bp_Rp, G_err, Bp_Rp_err)

    # Set bounds of least square fit (mass, metallicity, age)
    bounds = (
        [0.09, min(metallicity_grid), min(age_grid)],  # lower bounds
        [20.0, max(metallicity_grid), max(age_grid)]  # upper bounds
    )
    
    
    # Run least square fit
    # Set mass error to half the fitted mass - can refine further
    try:
        fit = least_squares(residuals_Gaia, guess, args=(G, Bp_Rp, G_err, Bp_Rp_err, Fe_H, Fe_H_err), bounds=bounds)
    except Exception as e:
        print(f"Fit failed for index: {e}")
        return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

    # Find errors and covariance of the fit
    cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
    errs = np.sqrt(np.diag(cov))

    # Fitted parameters
    fitted = fit.x

    # Check to see how far the fit is from the initial guess
    #if np.abs(fitted[0] - guess[0]) > 0.1:
        #print(f"Warning: Fitted mass {fitted[0]:.2f} deviates significantly from initial guess {guess[0]:.2f}")
    # Print the fitted mass, metallicity and age with errors
    #print(f"Fitted mass: {fitted[0]:.2f} ± {errs[0]:.2f}, "
          #f"metallicity: {fitted[1]:.2f} ± {errs[1]:.2f}, "
          #f"age: {fitted[2]:.2f} ± {errs[2]:.2f}")          

    return fitted, errs




#----------------------- Fitting with Gaia observations --------------------
if Gaia_fit == True:

    #Load the Gaia data from the fits file   
    sample = fits.open(data_file)
    targets = sample[1].data

    # Close the fits file
    sample.close()

    # extract extinction and it's error in the g band
    Ag = targets['ag_gspphot']
    Ag_err = (targets['ag_gspphot_upper'] - targets['ag_gspphot_lower'])/2

    # propagate error for bp and rp apparent magnitude
    bp_err = 2.5/np.log(10) * 1/targets['phot_bp_mean_flux_over_error']
    rp_err = 2.5/np.log(10) * 1/targets['phot_rp_mean_flux_over_error']

    # extract reddening error
    redd_err = (targets['ebpminrp_gspphot_upper'] - targets['ebpminrp_gspphot_lower'])/2

    # Format Gaia data into absolute magnitude that are corrected for extinction 
    mg = targets['phot_g_mean_mag'] + 5*np.log10(targets['parallax']/100) - Ag 
    # calculate gaia colour corrected for reddenning 
    bp_rp = targets['phot_bp_mean_mag'] - targets['phot_rp_mean_mag'] - targets['ebpminrp_gspphot']

    # Extrapolate flux error to magnitude error using over error in flux, parallax and extinction/redenning
    mg_err = np.sqrt((2.5/np.log(10) * 1/targets['phot_g_mean_flux_over_error'])**2 + (5/np.log(10) * 1/targets['parallax_over_error'])**2 + (Ag_err)**2)
    # propagate error in the colour
    bp_rp_err = bp_err + rp_err + redd_err

    # Extract the iron abundance
    if feh_type == 'gspphot':
        # Set iron abundance and error to the gspphot
        Fe_H = targets['mh_gspphot']
        Fe_H_err = (targets['mh_gspphot_upper'] - targets['mh_gspphot_lower'])/2
    
    elif feh_type == 'gspspec':
        # Set iron abundance and error to the gspspec values
        Fe_H = targets['mh_gspspec']
        Fe_H_err = (targets['mh_gspspec_upper'] - targets['mh_gspspec_lower'])/2
    
    elif feh_type == 'cannon_cal':
        # Set iron abundance and error to the values calibrated with the Cannon (Das et al 2024)
        Fe_H = targets['FeH']
        Fe_H_err = targets['e_FeH']
    else:
        print('Invalid iron abundance chosen. Please select gspphot, gspspec or cannon_cal.')

    # Initialise lists to store fitted parameters
    fitted_mass = []
    fitted_Fe_H = []
    fitted_age = []
    fitted_mass_err = []
    fitted_Fe_H_err = []
    fitted_age_err = []

    # We can now run the fit age function 
    for i in range(len(Fe_H)):
    
        # run least square fit 
        try:
            fitted, errs = fit_age(Fe_H[i], Fe_H_err[i], mg[i], mg_err[i], bp_rp[i], bp_rp_err[i])
        except Exception as e:
            print(f"Fit failed for index {i}: {e}")
            fitted = [np.nan, np.nan, np.nan]
            errs = [np.nan, np.nan, np.nan]

        # Add fitted parameters and errors to the targets table
        fitted_mass.append(fitted[0])
        fitted_Fe_H.append(fitted[1])
        fitted_age.append(fitted[2])
        fitted_mass_err.append(errs[0])
        fitted_Fe_H_err.append(errs[1])
        fitted_age_err.append(errs[2])

        # print progress every 100 iterations
        if i % 100 == 0:
            print(f'Fitted {i} of {len(Fe_H)} targets')

    # Convert targets to an Astropy Table to allow adding new columns
    targets_table = Table(targets)

    # Add the fitted parameters to the targets table
    targets_table['fitted_mass'] = fitted_mass
    targets_table['fitted_Fe_H'] = fitted_Fe_H
    targets_table['fitted_age'] = fitted_age
    targets_table['fitted_mass_err'] = fitted_mass_err
    targets_table['fitted_Fe_H_err'] = fitted_Fe_H_err
    targets_table['fitted_age_err'] = fitted_age_err

    # Save the targets table with fitted parameters to a new fits file
    targets_table.write(output_file, format='fits', overwrite=True)
    print(f'Fitted parameters saved to {output_file}')



#-------------------- Fitting to isochrone models ---------------------

if model_fit == True:
    
    # Open combined isochrone file
    file = fits.open(data_file)
    iso_comb = file[1].data

    # limit to stars with an age of 10 gyr and solar metallicity for testing
    iso_comb = iso_comb[(np.abs(iso_comb['log_age'] - 10) < 0.01) & (np.abs(iso_comb['feh']) < 0.01)]
    # limit to subgiatn stars
    iso_comb = iso_comb[len(iso_comb)//2 : ]

    file.close()

    # Initialise lists to store fitted parameters
    fitted_mass = []
    fitted_feh = []
    fitted_age = []
    fitted_mass_err = []
    fitted_feh_err = []
    fitted_age_err = []

    # Set random seed for reproducible results
    rng = np.random.default_rng(100)

    n_models = len(iso_comb)
   
    # extract rows fundamental parameters
    true_age = iso_comb['log_age']
    true_feh = iso_comb['feh']
    true_mass = iso_comb['star_mass']

    # Extract true gaia magnitudes
    true_g = iso_comb['Gaia_G']
    true_bprp = iso_comb['Gaia_BP'] - iso_comb['Gaia_RP']
    # change only for survey simulations
    #true_bprp = iso_comb['Gaia_BpRp']

    # perturb true values by the noise using the seeded generator, use half to quoted error to simulate the given two sigma uncertainty
    feh_in = rng.normal(loc=true_feh, scale=feh_err)
    g_in = rng.normal(loc=true_g, scale=mag_err)
    bprp_in = rng.normal(loc=true_bprp, scale=bprp_err)

    for i in range(n_models):

        # least squares fit
        fitted, errs = fit_age(feh_in[i], feh_err, g_in[i], mag_err, bprp_in[i], bprp_err)
            
        print(f'true parameters: {true_mass[i]:.4f}, {true_feh[i]:.4f}, {true_age[i]:.4f}')
        print(f'input parameters: Gmag {g_in[i]}, feh {feh_in[i]}, bp-rp {bprp_in[i]}')
        print(f'fitted parameters: {fitted[0]:.4f} pm {errs[0]:.4f} , {fitted[1]:.4f} pm {errs[1]:.4f}, {fitted[2]:.4f} pm {errs[2]:.4f}')

        # Add fitted parameters and errors to the targets table
        fitted_mass.append(fitted[0])
        fitted_feh.append(fitted[1])
        fitted_age.append(fitted[2])
        fitted_mass_err.append(errs[0])
        fitted_feh_err.append(errs[1])
        fitted_age_err.append(errs[2])


    # Convert targets to an Astropy Table to allow adding new columns
    table = Table(iso_comb)

    # Add the fitted parameters to the targets table
    table['fitted_mass'] = fitted_mass
    table['fitted_feh'] = fitted_feh
    table['fitted_age'] = fitted_age
    table['fitted_mass_err'] = fitted_mass_err
    table['fitted_feh_err'] = fitted_feh_err
    table['fitted_age_err'] = fitted_age_err

    # Save fitted parameters to the same fits file 
    table.write(output_file, format='fits', overwrite=True)
    print(f'Fitted parameters saved to {output_file}')
