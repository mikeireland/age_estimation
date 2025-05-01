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

# Set initial guess (without mass estimate)
# Intrinsic parameters taken from isochrone 
mass_intrin = 1.8091925608779658
metallicity_intrin = -0.5
age_intrin = 8.9000000000000004

# Observed parameters
mass_err = 0.01
metallicity_err = 0.02

mass_obs = mass_intrin + np.random.normal(scale=mass_err)
metallicity_obs = metallicity_intrin + np.random.normal(scale=metallicity_err)

# Average over errors for the Gaia nss 
par_over_err = 85.78064
g_flux_over_err = 6674.98225
bp_flux_over_err = 3045.04454
rp_flux_over_err = 3592.30083

# Propagated errors for Gaia absolute magnitudes
G_mag_err = 5/np.log(10) * np.sqrt((1/par_over_err)**2 + (1/(2*g_flux_over_err))**2)
RP_mag_err = 5/np.log(10) * np.sqrt((1/par_over_err)**2 + (1/(2*bp_flux_over_err))**2)
BP_mag_err = 5/np.log(10) * np.sqrt((1/par_over_err)**2 + (1/(2*rp_flux_over_err))**2)


G_mag_obs = 1.436750 + np.random.normal(scale=G_mag_err)
BP_mag_obs = 1.442280 + np.random.normal(scale=BP_mag_err)
RP_mag_obs = 1.427353 + np.random.normal(scale=RP_mag_err)

# Set up mass grid in physical units of solar mass
mass_grid = np.linspace(1.1*10**-1, 20, 200)
log_age_grid = np.linspace(min(age_grid), max(age_grid), 170)

# Initialise 2d array to hold chi2 values for the grid search
mass_age_chi2_array = np.zeros((len(mass_grid), len(log_age_grid)))

# Run grid search over mass and age for the given observed magnitudes as metallicity
for i, mass in enumerate(mass_grid): 
    for j, age in enumerate(log_age_grid):

        # Find the maximum mass correspinding to the given metallicity and age
        max_mass = star_mass_max_interp((metallicity_obs, age))

        # Calculate fractional mass 
        frac_mass = mass/max_mass

        # If the fractional mass is greater than one, the model will not be interpolated well
        if frac_mass > 1:
            # set chi2 to Nan and move to next grid spot
            chi2 = np.nan

            # Add chi2 value to 2d array 
            mass_age_chi2_array[i][j] = chi2

            # Skip to next age-mass combination
            continue
        
        # Find photometry of this age and mass
        G_mag_model = Gaia_G_EDR3_interp((frac_mass, metallicity_obs, age))
        BP_mag_model = Gaia_BP_EDR3_interp((frac_mass, metallicity_obs, age))
        RP_mag_model = Gaia_RP_EDR3_interp((frac_mass, metallicity_obs, age))

        # Find sum of chi2 values of this model
        chi2 = ((G_mag_obs-G_mag_model)/G_mag_model)**2 + ((BP_mag_obs-BP_mag_model)**2/BP_mag_model)**2 + ((RP_mag_obs-RP_mag_model)**2/RP_mag_model)**2

        # Add chi2 value to 2d array 
        mass_age_chi2_array[i][j] = chi2

# Find age and mass of minimum chi2 value
# Find index
min_index = np.unravel_index(np.nanargmin(mass_age_chi2_array), mass_age_chi2_array.shape)
mass_idx, age_idx = min_index

# find corresponding mass and age values
best_mass = mass_grid[mass_idx]
best_log_age = log_age_grid[age_idx]

print(f"Minimum chi2: {mass_age_chi2_array[mass_idx, age_idx]}")
print(f"Best-fit mass: {best_mass:.4f} Msol")
print(f"Best-fit log(age): {best_log_age:.4f}")


# Set initial guess
initial_guess = [best_mass, metallicity_obs, best_log_age]

# Run fit
fit = least_squares(residuals_Gaia, initial_guess, args=(G_mag_obs, BP_mag_obs, RP_mag_obs, G_mag_err, BP_mag_err, RP_mag_err, best_mass, mass_err, metallicity_obs, metallicity_err))

# Find errors, covariance and correlation of the fit
cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
errs = np.sqrt(np.diag(cov))
correlation = cov / np.outer(errs, errs)

print(f'True mass (solar masses), [Fe/H] and age (Gyr): {mass_intrin:.3f}, {metallicity_intrin:.3f}, {age_intrin:.3f}')
print(f'Errors in mass (solar masses), [Fe/H] and age (Gyr): {errs[0]:.3f}, {errs[1]:.3f}, {errs[2]:.3f}')
print(f'Fitted mass (solar masses), [Fe/H] and age (Gyr): {fit.x[0]:.3f}, {fit.x[1]:.3f}, {fit.x[2]:.3f}')


# Run grid search over just age for observations with mass and metallicity 
#initialise 1d array to hold chi2 values for the grid search
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
    chi2 = ((G_mag_obs-G_mag_model)/G_mag_model)**2 + ((BP_mag_obs-BP_mag_model)**2/BP_mag_model)**2 + ((RP_mag_obs-RP_mag_model)**2/RP_mag_model)**2

    # Add to chi2 array 
    age_chi2_array[i] = chi2

# Find the age corresponding to the minimum chi2 value
min_index = np.nanargmin(age_chi2_array)
best_log_age = log_age_grid[min_index]
print(f'fitted age estimate from mass and metallicity: {best_log_age:.4f}')
# plot the chi2 values as a function of age
plt.plot(log_age_grid, age_chi2_array)
plt.yscale('log')
plt.xlabel('log(age)')
plt.ylabel('chi2')
plt.title(f'Mass = {mass_obs:.2f} Msol, [Fe/H] = {metallicity_obs:.2f}')
plt.show()

# Run fit with this initial guess
initial_guess = [mass_obs, metallicity_obs, best_log_age]

# Run fit
fit = least_squares(residuals_Gaia, initial_guess, args=(G_mag_obs, BP_mag_obs, RP_mag_obs, G_mag_err, BP_mag_err, RP_mag_err, mass_obs, mass_err, metallicity_obs, metallicity_err))

# Find errors, covariance and correlation of the fit
cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
errs = np.sqrt(np.diag(cov))
correlation = cov / np.outer(errs, errs)

print(f'True mass (solar masses), [Fe/H] and age (Gyr): {mass_intrin:.3f}, {metallicity_intrin:.3f}, {age_intrin:.3f}')
print(f'Errors in mass (solar masses), [Fe/H] and age (Gyr): {errs[0]:.3f}, {errs[1]:.3f}, {errs[2]:.3f}')
print(f'Fitted mass (solar masses), [Fe/H] and age (Gyr) with mass and ageS: {fit.x[0]:.3f}, {fit.x[1]:.3f}, {fit.x[2]:.3f}')




