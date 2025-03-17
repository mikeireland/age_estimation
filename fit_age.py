"""
This script is used to show how to interpolate between models based on initial mass to 
fit to an age. Note that it assumes no mass loss. I'm not sure how to account for mass loss...
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
plt.ion()

# Load the fits file, converting yr to Gyr as we go for simplicity
ff = fits.open('MIST_v1.2.fits')
# Get the fractional star age grid
fractional_star_age = ff['fractional_star_age'].data
# Get the star mass grid
mass_grid = ff['mass_grid'].data
# Get the metallicity grid
metallicity_grid = ff['metallicity_grid'].data
# Get the log_Teff, log_L and star_mass grids
log_Teff = ff['log_Teff'].data
log_L = ff['log_L'].data
star_mass = ff['star_mass'].data
# Get the max_age grid
max_age = ff['max_age'].data/1e9
# Close the fits file
ff.close()  

# Make a HR diagram plot (log_L vs log_Teff) for metalicities of -1 and 0 for stars of mass
# 0.6, 1.0, 1.4 solar masses. Probably this should be a Gaia density plot!
plt.figure(1)
plt.clf()
for i, (metallicity, linestyle) in enumerate(zip([-1, -0.5, 0], [':', '--', '-'])):
    for j, mass in enumerate([0.6, 1.0, 1.4]):
        # Find the index of the mass and metallicity
        mass_index = np.where(mass_grid == mass)[0][0]
        metallicity_index = np.where(metallicity_grid == metallicity)[0][0]
        # Plot the HR diagram
        max_ix = None
        if j==0:
            max_ix = -5
        plt.plot(log_Teff[mass_index, metallicity_index][7:max_ix], log_L[mass_index, metallicity_index][7:max_ix], \
            label=f'M={mass}, [Fe/H]={metallicity}', color=f'C{j}', linestyle=linestyle)
plt.xlabel('log(Teff)')
plt.ylabel('log(L)')
plt.axis([4.0, 3.58, -1, 2.5])
plt.legend()


log_Teff_interpolator = RegularGridInterpolator((mass_grid, metallicity_grid, fractional_star_age), log_Teff, bounds_error=False, fill_value=None)
log_L_interpolator = RegularGridInterpolator((mass_grid, metallicity_grid, fractional_star_age), log_L, bounds_error=False, fill_value=None)
max_age_interpolator = RegularGridInterpolator((mass_grid, metallicity_grid), max_age, bounds_error=False, fill_value=None)

# Let's create a function that returns the log_Teff and log_L for a given initial mass, metallicity and age,
# by interpolating between the models in all 3 axes on our regular grid.
def get_log_Teff_log_L(mass, metallicity, age):
    this_fractional_age = age / max_age_interpolator((mass, metallicity))
    return log_Teff_interpolator((mass, metallicity, this_fractional_age)), log_L_interpolator((mass, metallicity, this_fractional_age))

# For fitting using least_squares, we need to define a function that returns the 
# normalised difference between the observed and model log_Teff and log_L.
def residuals(params, log_Teff_obs, log_L_obs, FeH, mass, log_Teff_err, log_L_err, FeH_err, mass_err):
    """
    params: array-like, [mass, metallicity, age]
    log_Teff_obs: float, observed log_Teff
    log_L_obs: float, observed log_L
    FeH: float, observed metallicity in dex
    log_Teff_err: float, error on log_Teff
    log_L_err: float, error on log_L
    FeH_err: float, error on metallicity in dex
    """
    log_Teff_model, log_L_model = get_log_Teff_log_L(*params)
    return np.array([(log_Teff_model - log_Teff_obs) / log_Teff_err, (log_L_model - log_L_obs) / log_L_err, (FeH - params[1]) / FeH_err, (mass - params[0]) / mass_err])

# Let's test the residuals function by fitting to a star of mass 1.0 solar masses, metallicity 0, and age 8 Gyr.
# We'll use the log_Teff and log_L from the model, with some added noise on the starting point.
mass = 1.0
FeH = -0.5

#age = 7.7   #This is a lower giant branch star, where metalicity and age are degenerate. Great ages with mass!
#age = 7.36    #This is a subgiant star. Metalicity and age are still degenerate!
#age = 7.18    #This is another subgiant star, but closer to the turnoff. The lowest possible age error without mass.
#age = 6.55    #This is a turnoff star - the edge of useable (10%) ages without mass.
age = 5.5   #This is an upper main sequence star - still has a useable mass with age.
ages = [7.7, 7.36, 7.18, 6.55, 5.5]
symbols = ['rs', 'rx', 'ro', 'rx', 'rs']
for age, symbol in zip(ages, symbols):
    log_Teff_obs, log_L_obs = get_log_Teff_log_L(mass, FeH, age)
    plt.plot(log_Teff_obs, log_L_obs, symbol)
plt.tight_layout()

log_Teff_obs, log_L_obs = get_log_Teff_log_L(mass, FeH, age)
log_Teff_err = 0.005
log_L_err = 0.01
Fe_H_err = 0.07
mass_err = 0.02
initial_guess = [mass + np.random.normal(scale=0.01), FeH+ np.random.normal(scale=0.01), age+ np.random.normal(scale=0.0005)]
fit = least_squares(residuals, initial_guess, args=(log_Teff_obs, log_L_obs, FeH, mass, log_Teff_err, log_L_err, Fe_H_err, mass_err), x_scale=[0.01, 0.01, 0.0005])
print(fit)

cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
errs = np.sqrt(np.diag(cov))
correlation = cov / np.outer(errs, errs)
print(log_L_obs)
print(log_Teff_obs)
print(errs)
print(correlation)

