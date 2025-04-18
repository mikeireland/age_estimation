from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# Load fits file with isochrone models
ff = fits.open('isochrone_models.fits')
# Extract the individual data cubes from fits file
fractional_mass_grid = ff['fractional_mass_grid'].data
metallicity_grid = ff['metallicity_grid'].data
age_grid = ff['age_grid'].data
# Extract interpolated parameters from the fits file
log_Teff = ff['log_Teff'].data
log_L = ff['log_L'].data
star_mass = ff['star_mass'].data
Gaia_G_EDR3 = ff['Gaia_G_EDR3'].data
Gaia_BP_EDR3 = ff['Gaia_BP_EDR3'].data
Gaia_RP_EDR3 = ff['Gaia_RP_EDR3'].data
# Close the fits file

# Cut off stars younger than a few million years 
ff.close()

# Create interpolators for the parameters
log_Teff_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), log_Teff,bounds_error=False)
log_L_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), log_L,bounds_error=False)
star_mass_max_interp = RegularGridInterpolator((metallicity_grid, age_grid), star_mass)
Gaia_G_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), Gaia_G_EDR3,bounds_error=False)
Gaia_BP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), Gaia_BP_EDR3,bounds_error=False)
Gaia_RP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), Gaia_RP_EDR3,bounds_error=False)


# Create function that returns log_Teff and log_L for a given mass, metallicity and age.
def get_log_Teff_and_log_L(mass, metallicity, age):
    # Calculate the maximum star mass for the given metallicity and age
    max_star_mass = star_mass_max_interp((metallicity, age))
    # Check if the mass is greater than the maximum star mass
    if mass > max_star_mass:
        raise ValueError(f"Mass {mass} is greater than the maximum star mass {max_star_mass} for metallicity {metallicity} and age {age}.")
    fractional_mass = mass / max_star_mass
 
    # Interpolate to get the log_Teff and log_L for the given mass, metallicity and age
    log_Teff = log_Teff_interp((fractional_mass, metallicity, age))
    log_L = log_L_interp((fractional_mass, metallicity, age))
    return log_Teff, log_L

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


# Create function to return the normalised difference betwen the observed and modelled log_Teff and log_L
def residuals_phys(params, observed_log_Teff, observed_log_L, log_L_err, log_Teff_err, mass, mass_err, metallicity, metallicity_err):

    # Get the modelled log_Teff and log_L for the given mass, metallicity and age
    model_log_Teff, model_log_L = get_log_Teff_and_log_L(*params)


    # Calculate the normalised residuals
    residuals = np.array([(observed_log_Teff - model_log_Teff)/log_Teff_err, (observed_log_L - model_log_L)/log_L_err,
                           (params[1] - metallicity)/metallicity_err, (params[0] - mass)/mass_err])
    return residuals.flatten()

# Create funtion to return the normalised difference between the observed and modelled Gaia G, BP and RP magnitudes
def residuals_Gaia(params, observed_Gaia_G_EDR3, observed_Gaia_BP_EDR3, observed_Gaia_RP_EDR3, Gaia_G_EDR3_err, Gaia_BP_EDR3_err, Gaia_RP_EDR3_err, mass, mass_err, metallicity, metallicity_err):
    
    # Get the modelled Gaia G, BP and RP magnitudes for the given mass, metallicity and age
    model_Gaia_G_EDR3, model_Gaia_BP_EDR3, model_Gaia_RP_EDR3 = get_Gaia_magnitudes(*params)

    # Calculate the residuals
    residuals = np.array([(observed_Gaia_G_EDR3 - model_Gaia_G_EDR3)/Gaia_G_EDR3_err, (observed_Gaia_BP_EDR3 - model_Gaia_BP_EDR3)/Gaia_BP_EDR3_err, (observed_Gaia_RP_EDR3 - model_Gaia_RP_EDR3)/Gaia_RP_EDR3_err,
                          (params[1] - metallicity)/metallicity_err, (params[0] - mass)/mass_err])
    return residuals.flatten()

# Set initial guess

# Intrinsic parameters
mass = 1.0 # Solar mass 
metallicity = 0 # dex mh_gspphot
age = 8.0

log_Teff_obs, log_L_obs = get_log_Teff_and_log_L(mass, metallicity, age)
observed_Gaia_G_EDR3, observed_Gaia_BP_EDR3, observed_Gaia_RP_EDR3 = get_Gaia_magnitudes(mass, metallicity, age)

log_Teff_err = 0.005
log_L_err = 0.01

# Gaia magnitude errors, will be propagated from parallax and apparent magnitude asymmetric errors in magnitude space. 
Gaia_G_EDR3_err = 0.05
Gaia_BP_EDR3_err = 0.05
Gaia_RP_EDR3_err = 0.05

metallicity_err = 0.07
mass_err = 0.02 #Change this to a big number to have fitting without mass.
initial_guess = [mass + np.random.normal(scale=0.01), metallicity + np.random.normal(scale=0.01), age+ np.random.normal(scale=0.0005)]

# Run least squares fit.
# Fit with physical parameters
#fit = least_squares(residuals_phys, initial_guess, args=(log_Teff_obs, log_L_obs, log_L_err, log_Teff_err, mass, mass_err, metallicity, metallicity_err), x_scale=[0.01, 0.01, 0.0005])
# Fit with observable Gaia magnitudes
fit = least_squares(residuals_Gaia, initial_guess, args=(observed_Gaia_G_EDR3, observed_Gaia_BP_EDR3, observed_Gaia_RP_EDR3, Gaia_G_EDR3_err, Gaia_BP_EDR3_err, Gaia_RP_EDR3_err, mass, mass_err, metallicity, metallicity_err))

print(fit)

# Find errors, covariance and correlation of the fit
cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
errs = np.sqrt(np.diag(cov))
correlation = cov / np.outer(errs, errs)

# prints the results of the fit
print(f'Input mass (solar masses), [Fe/H] and age (Gyr): {mass:.3f}, {metallicity:.3f}, {age:.3f}')
print(f'Observed log Teff: {log_Teff_obs:.3f}')
print(f'Observed log luminosity: {log_L_obs:.3f}')
print(f'Errors in mass (solar masses), [Fe/H] and age (Gyr): {errs[0]:.3f}, {errs[1]:.3f}, {errs[2]:.3f}')
print(f'Fitted mass (solar masses), [Fe/H] and age (Gyr): {fit.x[0]:.3f}, {fit.x[1]:.3f}, {fit.x[2]:.3f}')
print(f'Covariance matrix:\n{cov}')
print(f'Correlation matrix:\n{correlation}')



