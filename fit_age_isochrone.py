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
    log_age_grid = np.linspace(6.3, max(age_grid), 100)

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

# Plot age precision as a function of a single isochrone with and with out mass estimate
def plot_age_precision(age, file, num_data):
    from astropy.table import Table
    import os

    iso = Table.read(os.path.join(os.getcwd(), 'data', file), format='ascii.commented_header', header_start=-1)
    metallicity = 0 
    metallicity_err = 0.02
    
    #Extract the portion of this isochrone for the same age
    true_age = age # between 5.0 and 10.3
    age_idx = np.where(np.abs(iso['log10_isochrone_age_yr'] - true_age) < 0.001)[0]

    # Extract fundemental stellar parameters for this isochrone
    #m_star = iso['star_mass'][age_idx]
    #plt.plot(m_star)

    # Find index where mass starts to decrease
    #increasing_mask = np.diff(m_star) > 0.0
    #print(increasing_mask)
    #cutoff_index = np.argmax(~increasing_mask)

    # Truncate to only the strictly increasing part !NEED TO MAKE THIS MORE ROBUST!
    #age_idx = age_idx[:cutoff_index]
    age_idx = age_idx[:num_data]
    #Also only take every 8th point to speed up the fit and increase vsibility on the plot
    age_idx = age_idx[::8]
    
    # Reextract the mass values for the truncated isochrone
    m_star = iso['star_mass'][age_idx]

    # Extract Gaia G, BP and RP magnitudes for this isochrone
    G_mag = iso['Gaia_G_EDR3'][age_idx]
    Bp_mag = iso['Gaia_BP_EDR3'][age_idx]
    Rp_mag = iso['Gaia_RP_EDR3'][age_idx]

    # Set mass and magnitude error to 1 percent
    mass_err = 0.01 * m_star
    G_mag_err = 0.1 
    Bp_mag_err = 0.1 
    Rp_mag_err = 0.1

    # Observed parameters are the intrinsic ones with some noise
    metallicity_obs = metallicity + np.random.normal(scale=metallicity_err)
    mass_obs = m_star + np.random.normal(scale=mass_err, size=len(age_idx))
    G_mag_obs = G_mag + np.random.normal(scale=G_mag_err, size=len(age_idx))
    Bp_mag_obs = Bp_mag + np.random.normal(scale=Bp_mag_err, size=len(age_idx))
    Rp_mag_obs = Rp_mag + np.random.normal(scale=Rp_mag_err, size=len(age_idx))

    # Set bounds for the fit
    bounds = (
        [0.09, min(metallicity_grid), min(age_grid)],  # lower bounds
        [20.0, max(metallicity_grid), max(age_grid)]  # upper bounds
    )

    ages_mass = []
    age_errs_mass = []


    # Run fit with mass in the guess
    for i in range(len(age_idx)):

        # Set intial guess for the fit
        get_initial_guess = initial_guess_mass(metallicity_obs, mass_obs[i], G_mag_obs[i], Bp_mag_obs[i], Rp_mag_obs[i], G_mag_err, Bp_mag_err, Rp_mag_err)
        print(f"Initial guess with mass: {get_initial_guess}")
        # Make sure the initial guess for age is within the range of the isochrone
        if get_initial_guess[2] < min(age_grid) or get_initial_guess[2] > max(age_grid):
            # If the initial guess is outside the range, set it to the closest value in the grid
            get_initial_guess[2] = np.clip(get_initial_guess[2], min(age_grid), max(age_grid))

        # Run least square fit
        fit = least_squares(residuals_Gaia, get_initial_guess, args=(G_mag_obs[i], Bp_mag_obs[i], Rp_mag_obs[i], G_mag_err, Bp_mag_err, Rp_mag_err, mass_obs[i], mass_err[i], metallicity_obs, metallicity_err), 
                            bounds=bounds)

        # Find errors and covariance of the fit
        cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
        errs = np.sqrt(np.diag(cov))

        # Append age error to list
        age_errs_mass.append(errs[2])
        #Append age to list
        ages_mass.append(fit.x[2])

    # Convert the lists to numpy arrays
    ages_mass = np.array(ages_mass)
    age_errs_mass = np.array(age_errs_mass)

    #Only plot the points where the fitted age is correct within the error and error is less than 0.5
    good_fit = (np.abs(ages_mass - true_age) < 2 * age_errs_mass) 
    #& (age_errs_mass < 0.5)
    return G_mag[good_fit], Bp_mag[good_fit] - Rp_mag[good_fit], age_errs_mass[good_fit]
    # Plot the sigma difference from the fitted age and the true age on a colour-magnitude diagram
    cmd = plt.figure(figsize=(5,5))

    ax2 = cmd.add_subplot(1,1,1)
    sc2 = ax2.scatter(G_mag[good_fit], Bp_mag[good_fit] - Rp_mag[good_fit], c=age_errs_mass[good_fit], s=2/age_errs_mass[good_fit], alpha=0.5, edgecolors='black', linewidths=0.5, 
                      cmap='viridis', 
                      vmin=0, vmax=max(age_errs_mass[good_fit]))
    ax2.set_xlabel('Bp - Rp')
    ax2.set_ylabel('G')
    ax2.invert_yaxis()
    ax2.set_title('Age estimate with mass')
    cb2 = cmd.colorbar(sc2, ax=ax2)
    cb2.set_label('sigma age')


    plt.show()

#plot_age_precision(10.3, 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd', 280)

# Get age and age error for input data of multiple isochrones
def get_age_precision(age, file):
    from astropy.table import Table
    import os


    iso = Table.read(os.path.join(os.getcwd(), 'data', file), format='ascii.commented_header', header_start=-1)
    metallicity = 0 
    metallicity_err = 0.02

    #Extract the portion of this isochrone for the same age
    true_age = age # between 5.0 and 10.3
    age_idx = np.where(iso['log10_isochrone_age_yr'] == true_age)[0]

    split = int(len(age_idx)/4)
    # Remove stars beyond the max_star_mass_index for the current age
    #max_star_mass_idx = np.argmax(iso['star_mass'][age_idx])

    #age_idx = age_idx[:max_star_mass_idx]

    age_idx=age_idx[:split]

    m_star = iso['star_mass'][age_idx]
    #Only take every 8th point to speed up the fit and increase visibility on the plot
    #age_idx = age_idx[::2]
    
    # Reextract the mass values for the truncated isochrone
    m_star = iso['star_mass'][age_idx]

    # Extract Gaia G, BP and RP magnitudes for this isochrone
    G_mag = iso['Gaia_G_EDR3'][age_idx]
    Bp_mag = iso['Gaia_BP_EDR3'][age_idx]
    Rp_mag = iso['Gaia_RP_EDR3'][age_idx]

    # Set mass and magnitude error to 1 percent
    mass_err = 0.01 * m_star
    G_mag_err = 0.1 
    Bp_mag_err = 0.1 
    Rp_mag_err = 0.1

    # Observed parameters are the intrinsic ones with some noise
    metallicity_obs = metallicity + np.random.normal(scale=metallicity_err)
    mass_obs = m_star + np.random.normal(scale=mass_err, size=len(age_idx))
    G_mag_obs = G_mag + np.random.normal(scale=G_mag_err, size=len(age_idx))
    Bp_mag_obs = Bp_mag + np.random.normal(scale=Bp_mag_err, size=len(age_idx))
    Rp_mag_obs = Rp_mag + np.random.normal(scale=Rp_mag_err, size=len(age_idx))

    # Set bounds for the fit
    bounds = (
        [0.09, min(metallicity_grid), min(age_grid)],  # lower bounds
        [20.0, max(metallicity_grid), max(age_grid)]  # upper bounds
    )

    # Initialise lists to hold fitted ages and errors
    ages_mass = []
    age_errs_mass = []

    # Run fit with mass in the guess
    for i in range(len(age_idx)):

        # Set intial guess for the fit
        get_initial_guess = initial_guess_mass(metallicity_obs, mass_obs[i], G_mag_obs[i], Bp_mag_obs[i], Rp_mag_obs[i], G_mag_err, Bp_mag_err, Rp_mag_err)
        #print(f"Initial guess with mass: {get_initial_guess}")
        # Make sure the initial guess for age is within the range of the isochrone
        if get_initial_guess[2] < 6.3 or get_initial_guess[2] > max(age_grid):
            # If the initial guess is outside the range, do not run the fit
            get_initial_guess[2] = np.clip(get_initial_guess[2], min(age_grid), max(age_grid))
            #continue

        # Run least square fit
        fit = least_squares(residuals_Gaia, get_initial_guess, args=(G_mag_obs[i], Bp_mag_obs[i], Rp_mag_obs[i], G_mag_err, Bp_mag_err, Rp_mag_err, mass_obs[i], mass_err[i], metallicity_obs, metallicity_err), 
                                bounds=bounds)

        # Find errors and covariance of the fit
        cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
        errs = np.sqrt(np.diag(cov))

        # Append age error to list
        age_errs_mass.append(errs[2])
        #Append age to list
        ages_mass.append(fit.x[2])

    # Convert the lists to numpy arrays
    ages_mass = np.array(ages_mass)
    age_errs_mass = np.array(age_errs_mass)

    #Only plot the points where the fitted age is correct within the error and error is less than 0.5
    good_fit = (np.abs(ages_mass - true_age) < 2 * age_errs_mass) & (age_errs_mass < 0.5)

    return G_mag[good_fit], Bp_mag[good_fit], Rp_mag[good_fit], age_errs_mass[good_fit]

#ages_to_fit = np.linspace(6.5, 9.0, 6)
#fig, ax = plt.subplots()
# Choose a colormap with enough distinct colors
#cmap = plt.get_cmap('gist_rainbow', len(ages_to_fit))
#norm = plt.Normalize(vmin=min(ages_to_fit), vmax=max(ages_to_fit))

#for age in ages_to_fit:

    G, BP, RP, age_err = get_age_precision(age, 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd')

    sc = ax.scatter(
        BP - RP,
        G,
        c=[age] * len(G),  # Color by age
        s=2/age_err,       # Size by inverse error
        cmap=cmap,
        norm=norm,
        edgecolors='black',
        linewidths=0.5, 
        alpha=0.5,
        label=f'log(age) = {age}'
    )

# Label and invert axis
#ax.set_xlabel('Bp - Rp')
#ax.set_ylabel('G')
#ax.invert_yaxis()
#plt.legend()
#plt.show()

# Ages to fit on plot
ages_to_fit = np.arange(9.5, 10.2, 0.1)
num_data = [330,310,300,280,290,280,270,270]

fig, ax = plt.subplots()

# Choose a colormap with enough distinct colors
cmap = plt.get_cmap('gist_rainbow', len(ages_to_fit))
norm = plt.Normalize(vmin=min(ages_to_fit), vmax=max(ages_to_fit))

# Store handles for legends
color_legend_handles = []

for i, ages in enumerate(ages_to_fit):
    g_mag, bp_rp, age_err = plot_age_precision(ages, 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd', num_data[i]-30)

    sc = ax.scatter(
        bp_rp,
        g_mag,
        c=[ages] * len(g_mag),
        s=4 / age_err,
        cmap=cmap,
        norm=norm,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.5,
        label=None  # Prevent automatic legend
    )

    # One handle per age
    color_legend_handles.append(
        ax.scatter([], [], color=cmap(norm(ages)), label=f"{10**(ages-9):.1f}", s=40)
    )

# Size legend
example_errors = [0.02, 0.05, 0.1, 0.2]
size_legend_handles = [
    ax.scatter([], [], s=4/err, edgecolors='black', facecolors='gray', label=f'±{err:.2f}')
    for err in example_errors
]

# Create legends
size_legend = ax.legend(
    handles=size_legend_handles,
    title="Age Error (dex)",
    loc='upper left',
    bbox_to_anchor=(1, 1),  # Adjust horizontal and vertical offset
    borderaxespad=0.
)

color_legend = ax.legend(
    handles=color_legend_handles,
    title="Age (Gyr)",
    loc='upper left',
    bbox_to_anchor=(1, 0.65),  # Lower than size legend
    borderaxespad=0.
)

# Add both legends to the plot
ax.add_artist(size_legend)
ax.add_artist(color_legend)

# Axis labels and style
ax.set_xlabel('Bp - Rp')
ax.set_ylabel('G')
ax.invert_yaxis()

# Adjust to leave space for legends
fig.tight_layout()
plt.subplots_adjust(right=0.6)  # Increase margin to the right for both legends

plt.show()