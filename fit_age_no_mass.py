from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares


# Parameters to make the fit better
min_log_age = 7.5

# Load fits file with isochrone models
ff = fits.open('isochrone_models.fits')

#-------------------------------------------------------
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
log_Teff_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    log_Teff[:,:,min_age_ix:],bounds_error=False, method = 'cubic')
log_L_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    log_L[:,:,min_age_ix:],bounds_error=False, method = 'cubic')
Gaia_G_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]),         
                                    Gaia_G_EDR3[:,:,min_age_ix:],bounds_error=False, method = 'cubic')
Gaia_BP_RP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    Gaia_BP_EDR3[:,:,min_age_ix:] - Gaia_RP_EDR3[:,:,min_age_ix:], bounds_error=False, method = 'cubic')
#Gaia_BP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    #Gaia_BP_EDR3[:,:,min_age_ix:],bounds_error=False, method = 'cubic')
#Gaia_RP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    #Gaia_RP_EDR3[:,:,min_age_ix:],bounds_error=False, method = 'cubic')

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

    # Set up mass grid in physical units of solar mass
    mass_grid = np.linspace(0.1, 8, 100)
    log_age_grid = np.linspace(7.0, max(age_grid), 50)

    # Initialise 2d array to hold chi2 values for the grid search
    mass_age_chi2_array = np.zeros((len(mass_grid), len(log_age_grid)))

    # Run grid search over mass and age for the given observed magnitudes as metallicity
    for j, age in enumerate(log_age_grid):
        # Find the maximum mass correspinding to the given metallicity and age
        max_mass = star_mass_max_interp((metallicity_obs, age))
        # Calculate fractional mass 
        frac_mass = mass_grid/max_mass
        mass_age_chi2_array[frac_mass > 1,j] = np.nan
        
        # Find photometry of this age and mass
        G_mag_model = Gaia_G_EDR3_interp((frac_mass, metallicity_obs, age))
        BP_RP_mag_model = Gaia_BP_RP_EDR3_interp((frac_mass, metallicity_obs, age))
        #RP_mag_model = Gaia_RP_EDR3_interp((frac_mass, metallicity_obs, age))


        # Find sum of chi2 values of this model
        chi2 = ((G_mag_obs-G_mag_model)/G_mag_err)**2 + ((BP_RP_mag_obs-BP_RP_mag_model)/BP_RP_mag_err)**2 

        # Add chi2 value to 2d array 
        mass_age_chi2_array[:,j] = chi2
 
    # Find age and mass of minimum chi2 value
    # Find index
    min_index = np.unravel_index(np.nanargmin(mass_age_chi2_array), mass_age_chi2_array.shape)
    mass_idx, age_idx = min_index
    #chi2_min = mass_age_chi2_array[mass_idx, age_idx]

    # find corresponding mass and age values
    best_mass = mass_grid[mass_idx]
    best_log_age = log_age_grid[age_idx]

    # Find confidence interval for mass at the best age
    # Confidence interval on mass at best-fit age (1 parameter: Δχ² = 1.0 for 68.3%)
    #chi2_column_at_best_age = mass_age_chi2_array[:, age_idx]
    #within_confidence = chi2_column_at_best_age <= chi2_min + 1.0

    #if np.any(within_confidence):
        #mass_lower = np.min(mass_grid[within_confidence])
        #mass_upper = np.max(mass_grid[within_confidence])
        # least square fit only take symmetrical error, so take the average of the lower and upper bounds
        #mass_err = (mass_upper - mass_lower) / 2.0
        # mass error must be greater than the step size of the mass grid
        #if mass_err < (mass_grid[1] - mass_grid[0]):
            #mass_err = (mass_grid[1] - mass_grid[0])
    #else:
        # if the confidence interval is not found set mass error to the step size of the mass grid
        #mass_err = (mass_grid[1] - mass_grid[0])


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

    return fitted, errs

# Load the Gaia data from the fits file
sample = fits.open('gaia_giants_test_sample.fits')
targets = sample[1].data[:200]
#targets = np.where(all['source_id'] == 439766825338229504)[0]
# Close the fits file
sample.close()

# Set extinction coefficients for the Gaia bands
Rbp = 3.374
Rrp = 2.035

# calulate extinction in each observation band
Ag = targets['ag_gspphot']
 
Abp_man = Rbp/(Rbp-Rrp) * targets['ebpminrp_gspphot']
Arp_man = Rrp/(Rbp-Rrp) * targets['ebpminrp_gspphot']

Abp = np.where(np.isnan(targets['abp_gspphot']), Abp_man, targets['abp_gspphot'])
Arp = np.where(np.isnan(targets['arp_gspphot']), Arp_man, targets['arp_gspphot'])


# Extract errors in extinction
Ag_err = np.max([np.abs(targets['ag_gspphot_lower']-Ag), np.abs(targets['ag_gspphot_upper']-Ag)], axis=0)

#Abp_err_dir = np.max([np.abs(targets['abp_gspphot_lower']-Abp), np.abs(targets['abp_gspphot_upper']-Abp)], axis=0)
#Arp_err_dir = np.max([np.abs(targets['arp_gspphot_lower']-Arp), np.abs(targets['arp_gspphot_upper']-Arp)], axis=0)
#Abp_err_man = np.max([np.abs(targets['ebpminrp_gspphot_lower']*Rbp/(Rbp-Rrp) - Abp), np.abs(targets['ebpminrp_gspphot_upper']*Rbp/(Rbp-Rrp) - Abp)], axis=0)
#Arp_err_man = np.max([np.abs(targets['ebpminrp_gspphot_lower']*Rrp/(Rbp-Rrp) - Arp), np.abs(targets['ebpminrp_gspphot_upper']*Rrp/(Rbp-Rrp) - Arp)], axis=0)

#Abp_err = np.where(np.isnan(targets['abp_gspphot']), Abp_err_man, Abp_err_dir)
#Arp_err = np.where(np.isnan(targets['arp_gspphot']), Arp_err_man, Arp_err_dir)

bp_err = 2.5/np.log(10) * 1/targets['phot_bp_mean_flux_over_error']
rp_err = 2.5/np.log(10) * 1/targets['phot_rp_mean_flux_over_error']
redd_err = np.maximum(targets['ebpminrp_gspphot'] - targets['ebpminrp_gspphot_lower'], targets['ebpminrp_gspphot_upper']-targets['ebpminrp_gspphot'])

# Format Gaia data into absolute magnitudes that are corrected for extinction
mg = targets['phot_g_mean_mag'] + 5*np.log10(targets['parallax']/100) - Ag 
#mbp = targets['phot_bp_mean_mag'] + 5*np.log10(targets['parallax']/100) - Abp
#mrp = targets['phot_rp_mean_mag'] + 5*np.log10(targets['parallax']/100) - Arp
bp_rp = targets['phot_bp_mean_mag'] - targets['phot_rp_mean_mag'] - targets['ebpminrp_gspphot']

# Extrapolate flux error to magnitude error using over error in flux, parallax and extinction/redenning
mg_err = np.sqrt((2.5/np.log(10) * 1/targets['phot_g_mean_flux_over_error'])**2 + (5/np.log(10) * targets['parallax_error']/targets['parallax'])**2 + (Ag_err)**2)
#mbp_err = np.sqrt((2.5/np.log(10) * 1/targets['phot_bp_mean_flux_over_error'])**2 + (5/np.log(10) * 1/targets['parallax_over_error'])**2 + (Abp_err)**2)
#mrp_err = np.sqrt((2.5/np.log(10) * 1/targets['phot_rp_mean_flux_over_error'])**2 + (5/np.log(10) * 1/targets['parallax_over_error'])**2 + (Arp_err)**2)
bp_rp_err = bp_err + rp_err + redd_err

# Extract the iron abundance,
Fe_H = targets['mh_gspspec']
Fe_H_err = np.max([np.abs(targets['mh_gspspec_lower']-Fe_H), np.abs(targets['mh_gspspec_upper']-Fe_H)], axis=0)


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

# Add identifying information to check stars that did not fit properly
targets_table['g'] = mg
targets_table['g_err'] = mg_err
targets_table['bprp'] = bp_rp
targets_table['bprp_err'] = bp_rp_err

# Save the targets table with fitted parameters to a new fits file
targets_table.write('test.fits', format='fits', overwrite=True)
print('Fitted parameters saved to bad_point_test .fits')

# Open combined isochrone file
#file = fits.open('isochrones_grid_bprp.fits')
#iso_comb = file[1].data
#file.close()

# Set error for magntiudes and metallicity 
#mag_err = 0.01
#feh_err = 0.05

# Initialise lists to store fitted parameters
#fitted_mass = []
#fitted_feh = []
#fitted_age = []
#fitted_mass_err = []
#fitted_feh_err = []
#fitted_age_err = []

#for i in range(len(iso_comb['feh'])):
    # extract rows fundemental parameters
    #true_age = iso_comb['log_age'][i]
    #true_feh = iso_comb['feh'][i]
    #true_mass = iso_comb['star_mass'][i]

    # Extract true gaia magnitudes
    #true_g = iso_comb['Gaia_G'][i]
    #true_bp = iso_comb['Gaia_BP'][i]
    #true_rp = iso_comb['Gaia_RP'][i]

    # perturb true values by the noise
    #feh_in = np.random.normal(loc=true_feh, scale=feh_err)
    #g_in = np.random.normal(loc=true_g, scale=mag_err)
    #bp_in = np.random.normal(loc=true_bp, scale=mag_err)
    #rp_in = np.random.normal(loc=true_rp, scale=mag_err)

    # Run fit with no mass
    #try:
        #fitted, errs = fit_age(feh_in, feh_err, g_in, mag_err, bp_in, mag_err, rp_in, mag_err)
    #except Exception as e:
        #print(f"Fit failed for index {i}: {e}")
        #fitted = [np.nan, np.nan, np.nan]
        #errs = [np.nan, np.nan, np.nan]

    # Add fitted parameters and errors to the targets table
    #fitted_mass.append(fitted[0])
    #fitted_feh.append(fitted[1])
    #fitted_age.append(fitted[2])
    #fitted_mass_err.append(errs[0])
    #fitted_feh_err.append(errs[1])
    #fitted_age_err.append(errs[2])


    # print progress every 100 iterations
    #if i % 100 == 0:
        #print(f'Fitted {i} of {len(iso_comb['feh'])} targets')

# Convert targets to an Astropy Table to allow adding new columns
#table = Table(iso_comb)

# Add the fitted parameters to the targets table
#table['fitted_mass'] = fitted_mass
#table['fitted_feh'] = fitted_feh
#table['fitted_age'] = fitted_age
#table['fitted_mass_err'] = fitted_mass_err
#table['fitted_feh_err'] = fitted_feh_err
#table['fitted_age_err'] = fitted_age_err

# Also append true parameters for comparision
#table['true_mass'] = iso_comb['star_mass']
#table['true_age'] = iso_comb['log_age']
#table['true_feh'] = iso_comb['feh']

# Save fitted parameters to the same fits file 
#table.write('isochrones_grid_bprp_colour_fitted.fits', format='fits', overwrite=True)
