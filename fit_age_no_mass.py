from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import time

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

# Create interpolators for the parameters. First, 3D interpolators
log_Teff_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    log_Teff[:,:,min_age_ix:],bounds_error=False)
log_L_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    log_L[:,:,min_age_ix:],bounds_error=False)
Gaia_G_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]),         
                                    Gaia_G_EDR3[:,:,min_age_ix:],bounds_error=False)
Gaia_BP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    Gaia_BP_EDR3[:,:,min_age_ix:],bounds_error=False)
Gaia_RP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid[min_age_ix:]), 
                                    Gaia_RP_EDR3[:,:,min_age_ix:],bounds_error=False)

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
    Gaia_BP_EDR3 = Gaia_BP_EDR3_interp((fractional_mass, metallicity, age))
    Gaia_RP_EDR3 = Gaia_RP_EDR3_interp((fractional_mass, metallicity, age))
    return Gaia_G_EDR3, Gaia_BP_EDR3, Gaia_RP_EDR3

# Create funtion to return the normalised difference between the observed and modelled Gaia G, BP and RP magnitudes
def residuals_Gaia(params, observed_Gaia_G_EDR3, observed_Gaia_BP_EDR3, observed_Gaia_RP_EDR3, Gaia_G_EDR3_err, Gaia_BP_EDR3_err, Gaia_RP_EDR3_err, mass, mass_err, metallicity, metallicity_err):
    
    # Get the modelled Gaia G, BP and RP magnitudes for the given mass, metallicity and age
    model_Gaia_G_EDR3, model_Gaia_BP_EDR3, model_Gaia_RP_EDR3 = get_Gaia_magnitudes(*params)

    # Calculate the residuals
    residuals = np.array([(observed_Gaia_G_EDR3 - model_Gaia_G_EDR3)/Gaia_G_EDR3_err, (observed_Gaia_BP_EDR3 - model_Gaia_BP_EDR3)/Gaia_BP_EDR3_err, (observed_Gaia_RP_EDR3 - model_Gaia_RP_EDR3)/Gaia_RP_EDR3_err,
                          (params[1] - metallicity)/metallicity_err, (params[0] - mass)/mass_err])
    return residuals.flatten()

#Define function to get initial guess from metallicities
def initial_guess(metallicity_obs, G_mag_obs, BP_mag_obs, RP_mag_obs, G_mag_err, BP_mag_err, RP_mag_err):

    # Set up mass grid in physical units of solar mass
    mass_grid = np.linspace(0.1, 5, 100)
    log_age_grid = np.linspace(7.5, max(age_grid), 50)

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
        BP_mag_model = Gaia_BP_EDR3_interp((frac_mass, metallicity_obs, age))
        RP_mag_model = Gaia_RP_EDR3_interp((frac_mass, metallicity_obs, age))

        # Find sum of chi2 values of this model
        chi2 = ((G_mag_obs-G_mag_model)/G_mag_err)**2 + ((BP_mag_obs-BP_mag_model)**2/BP_mag_err)**2 + ((RP_mag_obs-RP_mag_model)**2/RP_mag_err)**2

        # Add chi2 value to 2d array 
        mass_age_chi2_array[:,j] = chi2
 
    # Find age and mass of minimum chi2 value
    # Find index
    min_index = np.unravel_index(np.nanargmin(mass_age_chi2_array), mass_age_chi2_array.shape)
    mass_idx, age_idx = min_index

    # find corresponding mass and age values
    best_mass = mass_grid[mass_idx]
    best_log_age = log_age_grid[age_idx]

    return [best_mass, metallicity_obs, best_log_age]

# Define function to fit age for give metallicity and Gaia magitudes with errors:
def fit_age(Fe_H, Fe_H_err, G, G_err, Bp, Bp_err, Rp, Rp_err):

    # get the initial guess from grid search 
    guess = initial_guess(Fe_H, G, Bp, Rp, G_err, Bp_err, Rp_err)

    # Set bounds of least square fit (mass, metallicity, age)
    bounds = (
        [0.09, min(metallicity_grid), min(age_grid)],  # lower bounds
        [20.0, max(metallicity_grid), max(age_grid)]  # upper bounds
    )

    # Run least square fit
    # Set mass error to half the fitted mass - can refine further
    fit = least_squares(residuals_Gaia, guess, args=(G, Bp, Rp, G_err, Bp_err, Rp_err, guess[0], 0.5*guess[0], Fe_H, Fe_H_err), bounds=bounds)

    # Find errors and covariance of the fit
    cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
    errs = np.sqrt(np.diag(cov))

    # Fitted parameters
    fitted = fit.x

    return fitted, errs

# Take parameters from isochrone mode
true_age = 9.55
true_mass = 1.34259395186597

Fe_H = 0.00 # dex
Fe_H_err = 0.1

G_mag = 2.805213
Bp_mag = 3.165582
Rp_mag = 2.279105

G_err = 0.1
Bp_err = 0.1
Rp_err = 0.1

# Add noise to inputs to make sure fitting routine works properly
Fe_H_obs = Fe_H + np.random.normal(scale=Fe_H_err)
G_obs = G_mag + np.random.normal(scale=G_err)
Bp_obs = Bp_mag + np.random.normal(scale=Bp_err)
Rp_obs = Rp_mag + np.random.normal(scale=Rp_err)

# Run age fitting function on isochrone data 
fitted, errs = fit_age(Fe_H, Fe_H_err, G_mag, G_err, Bp_mag, Bp_err, Rp_mag, Rp_err)

print(f'Intrinsic mass (solar masses), [Fe/H] and age (Gyr): {true_mass:.3f}, {Fe_H:.3f}, {true_age:.3f}')
print(f'Fitted mass (solar masses), [Fe/H] and age (Gyr): {fitted[0]:.3f}, {fitted[1]:.3f}, {fitted[2]:.3f}')
print(f'Errors in mass (solar masses), [Fe/H] and age (Gyr): {errs[0]:.3f}, {errs[1]:.3f}, {errs[2]:.3f}')