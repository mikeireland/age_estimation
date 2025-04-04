from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares


# Load fits file with isochrone models
ff = fits.open('isochrone_models.fits')
# Extract the individual data cubes from fits file
fractional_mass_grid = ff['fractional_mass_grid'].data
metallicity_grid = ff['metallicity_grid'].data
age_grid = ff['age_grid'].data
# Extract interpolated parameters from the fits file
star_mass = ff['star_mass'].data
log_Teff = ff['log_Teff'].data
log_L = ff['log_L'].data
Gaia_G_EDR3 = ff['Gaia_G_EDR3'].data
Gaia_BP_EDR3 = ff['Gaia_BP_EDR3'].data
Gaia_RP_EDR3 = ff['Gaia_RP_EDR3'].data
# Close the fits file
ff.close()

#look at how fractional mass changes for a given initial mass and age 

# Create interpolators for the parameters
star_mass_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), star_mass)
log_Teff_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), log_Teff)
log_L_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), log_L)
Gaia_G_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), Gaia_G_EDR3)
Gaia_BP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), Gaia_BP_EDR3)
Gaia_RP_EDR3_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), Gaia_RP_EDR3)

# Create function that returns log_Teff and log_L for a given mass, metallicity and age.
def get_log_Teff_and_log_L(mass, metallicity, age):
    # Find the fractional mass by finding the best match in the star_mass data cube with the given age and metallicity
    # Find the array of star masses for the given age and metallicity as a function of fractional mass
    star_mass_array = star_mass_interp((fractional_mass_grid, [metallicity], [age]))
    # Find the corresponding fractional mass for the given mass by interpolating
    fractional_mass = np.interp(mass, star_mass_array.flatten(), fractional_mass_grid)

    # Interpolate to get the log_Teff and log_L for the given mass, metallicity and age
    log_Teff = log_Teff_interp((fractional_mass, metallicity, age))
    log_L = log_L_interp((fractional_mass, metallicity, age))
    return log_Teff, log_L

# Create funtion that returns the Gaia G, BP and RP magnitudes for a given mass, metallicity and age.
def get_Gaia_magnitudes(mass, metallicity, age):
    # Find the fractional mass from the observed mass using the star_mass data cube
    # Find the array of star masses for the given age and metallicity as a function of fractional mass
    star_mass_array = star_mass_interp((fractional_mass_grid, [metallicity], [age]))
    # Find the corresponding fractional mass for the given mass by interpolating
    fractional_mass = np.interp(mass, star_mass_array.flatten(), fractional_mass_grid)


    # Interpolate to get the Gaia G, BP and RP magnitudes for the given mass, metallicity and age
    Gaia_G_EDR3 = Gaia_G_EDR3_interp((fractional_mass, metallicity, age))
    Gaia_BP_EDR3 = Gaia_BP_EDR3_interp((fractional_mass, metallicity, age))
    Gaia_RP_EDR3 = Gaia_RP_EDR3_interp((fractional_mass, metallicity, age))
    return Gaia_G_EDR3, Gaia_BP_EDR3, Gaia_RP_EDR3


# Create function to return the normalised difference betwen the observed and modelled log_Teff and log_L
def residuals(params, observed_log_Teff, observed_log_L, observed_log_L_err, observed_log_Teff_err):
    # Unpack the parameters
    mass, metallicity, age = params

    # Get the modelled log_Teff and log_L for the given mass, metallicity and age
    model_log_Teff, model_log_L = get_log_Teff_and_log_L(mass, metallicity, age)

    # Calculate the residuals
    residuals = np.array([(observed_log_Teff - model_log_Teff)/observed_log_Teff_err, (observed_log_L - model_log_L)/observed_log_L_err])
    return residuals.flatten()

# Set initial guess

# Run least squares fit.
