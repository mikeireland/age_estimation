"""
This script is used to show how to interpolate between models based on initial mass to 
fit to an age. Note that it assumes no mass loss. I'm not sure how to account for mass loss...

See sample.py for the Gaia DR3 SQL query to get the data, and see
extract_and_package.py for the code to extract the MIST models from the MIST website.
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import bcutil
plt.ion()

# Set this to True to plot the Gaia colors, or False to plot the theoretical HR diagram.
gaia_plot = False

# Make a HR diagram plot  for metalicities of -1 and 0 for stars of mass
# 0.8, 1.0, 1.2 solar masses. Probably this should be a Gaia density plot!

plt.figure(1)
plt.clf()
dd = fits.getdata('1742325284998O-result.fits',1) #All Orbits

# A little messy, but this is the code copied from "sample.py" that
# extracts the sample.
abs_g_mag = dd['phot_g_mean_mag'] + 5*np.log10(dd['parallax']/100)
bp_rp = dd['phot_bp_mean_mag'] - dd['phot_rp_mean_mag']
sep = (dd['period']/365)**(2/3)*dd['parallax']
ww= np.where((dd['phot_g_mean_mag']<10) & (abs_g_mag < 3.1) & (bp_rp>0.65) & \
	(abs_g_mag>2*bp_rp-1.1) & (abs_g_mag<3*bp_rp + 0.7) & \
	( sep > 4) & (dd['ra']<240) & (dd['dec']<20) &
	( dd['period_error']/dd['period']<0.01) &
	( dd['parallax_error']/dd['parallax']<0.007) )[0]

plt.clf()
plt.hist2d(bp_rp, abs_g_mag, bins=40,range=[[0,2],[-0.5,5.5]], cmap='Greys', cmin=5, norm='log')
plt.plot(bp_rp[ww], abs_g_mag[ww], 'm.')

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

for i, (metallicity, linestyle) in enumerate(zip([-1, -0.5, 0], [':', '--', '-'])):
    for j, mass in enumerate([0.8, 1.0, 1.2]):
        # Find the index of the mass and metallicity
        mass_index = np.where(mass_grid == mass)[0][0]
        metallicity_index = np.where(metallicity_grid == metallicity)[0][0]
        # Plot the HR diagram
        max_ix = -1
        if j==0:
            max_ix = -4
        Teff = 10**log_Teff[mass_index, metallicity_index][7:max_ix]
        #log(g) is M/R^2, and T^4 = L/R^2, i.e. R^(-2) = T^4/L
        logg = 4.44 + np.log10(mass) + 4*(log_Teff[mass_index, metallicity_index][7:max_ix] - np.log10(5770)) - log_L[mass_index, metallicity_index][7:max_ix]
        vec = np.ones(len(Teff))
        #Do we plot Gaia colours?
        if gaia_plot:
            bcs = bcutil.bcstar(np.zeros(len(Teff), dtype=str),Teff,logg,metallicity*vec,0.00*vec,filters='G3 BP3 RP3',retarr=True)
            plt.plot(bcs[:,2] - bcs[:,1], -2.5*log_L[mass_index, metallicity_index][7:max_ix]-bcs[:,0]+4.74, \
                label=f'M={mass}, [Fe/H]={metallicity}', color=f'C{j}', linestyle=linestyle)
        else:
            plt.plot(Teff, log_L[mass_index, metallicity_index][7:max_ix], \
                label=f'M={mass}, [Fe/H]={metallicity}', color=f'C{j}', linestyle=linestyle)
if gaia_plot:
    plt.xlabel('Bp-Rp')
    plt.ylabel('G')
    plt.axis([0.3,2,5.5,-0.5])
else:
    plt.xlabel('log(Teff)')
    plt.ylabel('log(L)')
    plt.axis([4.0, 3.58, -1, 2.5])
plt.legend()

# Now we can create the interpolators for log_Teff, log_L and max_age
log_Teff_interpolator = RegularGridInterpolator((mass_grid, metallicity_grid, fractional_star_age), log_Teff, bounds_error=False, fill_value=None)
log_L_interpolator = RegularGridInterpolator((mass_grid, metallicity_grid, fractional_star_age), log_L, bounds_error=False, fill_value=None)
max_age_interpolator = RegularGridInterpolator((mass_grid, metallicity_grid), max_age, bounds_error=False, fill_value=None)

# Let's create a function that returns the log_Teff and log_L for a given initial mass, metallicity and age,
# by interpolating between the models in all 3 axes on our regular grid. Note that this is 
# by default a linear interpolation, but we can change this to a cubic interpolation by setting the method parameter
# to 'linear' or 'cubic' : For Catherine.
def get_log_Teff_log_L(mass, metallicity, age):
    this_fractional_age = age / max_age_interpolator((mass, metallicity))
    return log_Teff_interpolator((mass, metallicity, this_fractional_age)), \
        log_L_interpolator((mass, metallicity, this_fractional_age))

# For fitting using least_squares, we need to define a function that returns the 
# normalised difference between the observed and model log_Teff and log_L.
def residuals(params, log_Teff_obs, log_L_obs, FeH, mass, log_Teff_err, log_L_err, FeH_err, mass_err):
    """
    Compute the residuals for the least_squares fit. The model parameters "params" is
    what we'll have as the output of the fit.
    
    params: array-like, [mass, metallicity, age]
    log_Teff_obs: float, observed log_Teff
    log_L_obs: float, observed log_L
    FeH: float, observed metallicity in dex
    log_Teff_err: float, error on log_Teff
    log_L_err: float, error on log_L
    FeH_err: float, error on metallicity in dex
    """
    log_Teff_model, log_L_model = get_log_Teff_log_L(*params)
    return np.array([(log_Teff_model - log_Teff_obs) / log_Teff_err, \
        (log_L_model - log_L_obs) / log_L_err, \
        (FeH - params[1]) / FeH_err, \
        (mass - params[0]) / mass_err])

# Let's test the residuals function by fitting to a star of mass 1.0 solar masses, metallicity 0, and age 8 Gyr.
# We'll use the log_Teff and log_L from the model, with some added noise on the starting point.
mass = 1.0
FeH = -0.5

ages = [7.7, 7.36, 7.18, 6.55, 5.5]
symbols = ['rs', 'rx', 'ro', 'rx', 'rs']
for age, symbol in zip(ages, symbols):
    log_Teff_obs, log_L_obs = get_log_Teff_log_L(mass, FeH, age)
    logg = 4.44 + np.log10(mass) + 4*(log_Teff_obs - np.log10(5770)) - log_L_obs
    if gaia_plot:
        bcs = bcutil.bcstar('',10**log_Teff_obs,logg,FeH,0.00*vec,filters='G3 BP3 RP3',retarr=True)
        plt.plot(bcs[:,2] - bcs[:,1], -2.5*log_L_obs - bcs[:,0] + 4.74, symbol)
    else:
        plt.plot(log_Teff_obs, log_L_obs, symbol)
plt.colorbar()
plt.tight_layout()

#Comment out the line you want to test for printed output.
age = 7.7   #This is a lower giant branch star, where metalicity and age are degenerate. Great ages with mass!
#age = 7.36    #This is a subgiant star. Metalicity and age are still degenerate!
#age = 7.18    #This is another subgiant star, but closer to the turnoff. The lowest possible age error without mass. 4.6% without mass, 3.2% with mass.
#age = 6.55    #This is a turnoff star - the edge of useable (10%) ages without mass.
#age = 5.5   #This is an upper main sequence star - still has a useable mass with age.

#For a test, lets imagine that we have an observed star with Teff and L that corresponds 
# a model with the above mass, FeH and age.
log_Teff_obs, log_L_obs = get_log_Teff_log_L(mass, FeH, age)
log_Teff_err = 0.005
log_L_err = 0.01
Fe_H_err = 0.07
mass_err = 0.02 #Change this to a big number to have fitting without mass.
initial_guess = [mass + np.random.normal(scale=0.01), FeH+ np.random.normal(scale=0.01), age+ np.random.normal(scale=0.0005)]
fit = least_squares(residuals, initial_guess, args=(log_Teff_obs, log_L_obs, FeH, mass, log_Teff_err, log_L_err, Fe_H_err, mass_err), x_scale=[0.01, 0.01, 0.0005])
print(fit)

cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
errs = np.sqrt(np.diag(cov))
correlation = cov / np.outer(errs, errs)

print(f'Input mass (solar masses), [Fe/H] and age (Gyr): {mass:.3f}, {FeH:.3f}, {age:.3f}')
print(f'Observed Teff: {10**log_Teff_obs:.3f}')
print(f'Observed log luminosity: {log_L_obs:.3f}')
print(f'Errors in mass (solar masses), [Fe/H] and age (Gyr): {errs[0]:.3f}, {errs[1]:.3f}, {errs[2]:.3f}')
print(f'Fitted mass (solar masses), [Fe/H] and age (Gyr): {fit.x[0]:.3f}, {fit.x[1]:.3f}, {fit.x[2]:.3f}')
print(f'Covariance matrix:\n{cov}')
print(f'Correlation matrix:\n{correlation}')
print(correlation)

