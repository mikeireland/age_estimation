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
star_mass_max = ff['star_mass_max'].data
Gaia_G_EDR3 = ff['Gaia_G_EDR3'].data
Gaia_BP_EDR3 = ff['Gaia_BP_EDR3'].data
Gaia_RP_EDR3 = ff['Gaia_RP_EDR3'].data

# Close the fits file
ff.close()

# Create interpolators for the parameters
log_Teff_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), log_Teff,bounds_error=False)
log_L_interp = RegularGridInterpolator((fractional_mass_grid, metallicity_grid, age_grid), log_L,bounds_error=False)
star_mass_max_interp = RegularGridInterpolator((metallicity_grid, age_grid), star_mass_max)
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

# Define function to run grid search over just age for observations with mass and metallicity 
def initial_guess_mass(metallicity_obs, mass_obs, G_mag_obs, BP_mag_obs, RP_mag_obs, G_mag_err, BP_mag_err, RP_mag_err):
    
    # Set up age grid to check models for (don't use lowest age in the isohrone because those stars are still in star forming regions and surrounded by dust, meaning their observed magnitudes are not close to the intrinsic once as modelled by the isochrone)
    log_age_grid = np.linspace(7.5, max(age_grid), 100)

    # Initiliase 1d array to hold chi2 values for the grid search
    age_chi2_array = np.zeros(len(log_age_grid))

    for i, age in enumerate(log_age_grid):
        # Calculate the fractional mass for the given mass and age
        max_mass = star_mass_max_interp((metallicity_obs, age))
        frac_mass = mass_obs/max_mass

        # If the fractional mass is greater than one, the model will not be interpolated well
        if frac_mass > 1:
            # set chi2 to Nan and move to next grid spot
            chi2 = np.nan

            # Add chi2 value to 2d array 
            age_chi2_array[i] = chi2

            # Skip to next age-mass combination
            continue
        # Find the photometry of this model 
        G_mag_model = Gaia_G_EDR3_interp((frac_mass, metallicity_obs, age))
        BP_mag_model = Gaia_BP_EDR3_interp((frac_mass, metallicity_obs, age))
        RP_mag_model = Gaia_RP_EDR3_interp((frac_mass, metallicity_obs, age))

        # Find sum of chi2 values of this model
        chi2 = ((G_mag_obs-G_mag_model)/G_mag_err)**2 + ((BP_mag_obs-BP_mag_model)/BP_mag_err)**2 + ((RP_mag_obs-RP_mag_model)/RP_mag_err)**2

        # Add to chi2 array 
        age_chi2_array[i] = chi2

    # Find the age corresponding to the minimum chi2 value
    min_index = np.nanargmin(age_chi2_array)
    best_log_age = log_age_grid[min_index]

    return[mass_obs, metallicity_obs, best_log_age]

# define function to fit with the least square fit given the mass, metallcity and gaia magnitudes with corresponding errors
def fit_age(feh, feh_err, mass, mass_err, G, G_err, Bp, Bp_err, Rp, Rp_err):

    # get the initial guess from grid search 
    guess = initial_guess_mass(feh, mass, G, Bp, Rp, G_err, Bp_err, Rp_err)

    # Set bounds of least square fit (mass, metallicity, age)
    bounds = (
        [0.09, min(metallicity_grid), min(age_grid)],  # lower bounds
        [20.0, max(metallicity_grid), max(age_grid)]  # upper bounds
    )

    # Run least square fit
    # Set mass error to half the fitted mass - can refine further
    try:
        fit = least_squares(residuals_Gaia, guess, args=(G, Bp, Rp, G_err, Bp_err, Rp_err, mass, mass_err, feh, feh_err), 
                            bounds=bounds)
    except:
        print(f'Fit failed for Fe_H={feh}, G={G}, Bp={Bp}, Rp={Rp}')
        return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

    # Find errors and covariance of the fit
    cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
    errs = np.sqrt(np.diag(cov))

    # Fitted parameters
    fitted = fit.x

    return fitted, errs

# import fits file with isochrones 
file = fits.open('isochrones_combined.fits')
iso_comb = file[1].data

# Set error for magntiudes and metallicity 
mag_err = 0.1
feh_err = 0.1

# Initialise lists to store fitted parameters
mass_fitted_mass = []
mass_fitted_feh = []
mass_fitted_age = []
mass_fitted_mass_err = []
mass_fitted_feh_err = []
mass_fitted_age_err = []

for i in range(len(iso_comb['feh'])):
    # extract rows fundemental parameters
    true_age = iso_comb['log_age'][i]
    true_feh = iso_comb['feh'][i]
    true_mass = iso_comb['star_mass'][i]

    # Extract true gaia magnitudes
    true_g = iso_comb['Gaia_G'][i]
    true_bp = iso_comb['Gaia_BP'][i]
    true_rp = iso_comb['Gaia_RP'][i]
    
    # perturb true values by the noise
    feh_in = np.random.normal(loc=true_feh, scale=feh_err)
    mass_in =np.random.normal(loc=true_mass, scale=true_mass*0.01)
    g_in = np.random.normal(loc=true_g, scale=mag_err)
    bp_in = np.random.normal(loc=true_bp, scale=mag_err)
    rp_in = np.random.normal(loc=true_rp, scale=mag_err)

    # Run fit with no mass
    try:
        fitted, errs = fit_age(feh_in, feh_err, mass_in, true_mass*0.01, g_in, mag_err, bp_in, mag_err, rp_in, mag_err)
    except Exception as e:
        print(f"Fit failed for index {i}: {e}")
        fitted = [np.nan, np.nan, np.nan]
        errs = [np.nan, np.nan, np.nan]

    # Add fitted parameters and errors to the targets table
    mass_fitted_mass.append(fitted[0])
    mass_fitted_feh.append(fitted[1])
    mass_fitted_age.append(fitted[2])
    mass_fitted_mass_err.append(errs[0])
    mass_fitted_feh_err.append(errs[1])
    mass_fitted_age_err.append(errs[2])

    # print progress every 100 iterations
    if i % 100 == 0:
        print(f'Fitted {i} of {len(iso_comb['feh'])} targets')

# Convert targets to an Astropy Table to allow adding new columns
from astropy.table import Table
table = Table(iso_comb)

# Add the fitted parameters to the targets table
table['mass_fitted_mass'] = mass_fitted_mass
table['mass_fitted_feh'] = mass_fitted_feh
table['mass_fitted_age'] = mass_fitted_age
table['mass_fitted_mass_err'] = mass_fitted_mass_err
table['mass_fitted_feh_err'] = mass_fitted_feh_err
table['mass_fitted_age_err'] = mass_fitted_age_err

# Save fitted parameters to the same fits file 
table.write('isochrones_combined_mass.fits', format='fits', overwrite=True)