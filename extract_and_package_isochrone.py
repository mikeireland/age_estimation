# Download and extract isochrone model from zip file and package into a single FITS file
import astropy.io.fits as fits
from astropy.table import Table
import numpy as np
import os
import glob
from urllib.request import urlretrieve

# Toggle for whether the files are already downloaded
already_downloaded = True

# Download the isochrone models from the MIST website with urlretrieve if not already downloaded (couldn't get this to work)
if not already_downloaded:
    # Create a "data" subdirectory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Download files directly into the "data" directory
    metstrs = ['m4.00', 'm3.50', 'm3.00', 'm2.50', 'm2.00', 'm1.75', 'm1.50', 'm1.25', 'm1.00', 'm0.75', 'm0.50', 'p0.00', 'p0.25', 'p0.50']
    for metstr in metstrs:
        fname = f'MIST_v1.2_feh_{metstr}_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmb'
        url = f'https://waps.cfa.harvard.edu/MIST/model_grids/isochrones/{fname}'
        dest_path = os.path.join(data_dir, fname)
        print(f'Downloading {url}')
        urlretrieve(url, dest_path)

# If data is already downloaded
if already_downloaded:
    # Define the data directory as a subdirectory named "data" in the current working directory
    data_path = os.path.join(os.getcwd(), 'data')
    
    # Verify the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory not found: {data_path}")
    
    # List all .iso.cmb files in the directory
    isochrone_files = os.listdir(data_path)
    

# Define fractional stellar mass
fractional_mass_grid = np.arange(0.001, 1.0, 0.001)

# Build metallicity grid from the filenemes of the isochrone files
metallicity_strings = [string.split('MIST_v1.2_feh_')[1][:5] for string in isochrone_files]

# Define the metallicity grid, where the first character "m" or "p" is the sign of the metallicity
# and the next 3 characters are the metallicity in dex.
metallicity_signs = np.array([1 if metallicity_string[0] == 'p' else -1 for metallicity_string in metallicity_strings])
metallicity_grid = np.array([float(metallicity_string[1:]) for metallicity_string in metallicity_strings]) * metallicity_signs

# Sort the metallicity grid, so that it is in increasing order, and the metallicity_strings are in the same order.
sort_indices = np.argsort(metallicity_grid)
metallicity_grid = metallicity_grid[sort_indices]
metallicity_strings = np.array(metallicity_strings)[sort_indices]
isochrone_files = np.array(isochrone_files)[sort_indices]

# From the first isocrone file find the age grid
# Read in the first isochrone file
iso = Table.read(os.path.join(data_path, isochrone_files[0]), format='ascii.commented_header', header_start=-1)
# Extract the unique values of iso['log10_isochrone_age_yr'] 
age_log_grid = np.array(np.unique(iso['log10_isochrone_age_yr']))


# Define function to calculate the fractional mass at the given age and metallicity
def calculate_fractional_mass(iso):
    for age in age_log_grid:
        # Find indcies of models at the same age
        age_index = np.where(np.isclose(iso['log10_isochrone_age_yr'], age, rtol=0.01))[0]

        # Maximum star mass 
        max_star_mass = np.max(iso['star_mass'][age_index])

        # Calculate the fractional mass
        fractional_mass = iso['star_mass'][age_index] / max_star_mass


# Define 3-D arrays to hold in input physical parameters of the models
log_Teff = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
log_L = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
star_mass = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
# Define the 3-D arrays to hold the observations parameters from Gaia
Gaia_G_EDR3 = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
Gaia_BP_EDR3 = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
Gaia_RP_EDR3 = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))

# Loop over metallicity strings to find the right directory
for i, metallicity_string in enumerate(isochrone_files):
    # Open the isochrone file
    iso = Table.read(os.path.join(data_path, metallicity_string), format='ascii.commented_header', header_start=-1)

    # Initialise fisr_age_index to 0
    first_age_index = 0

    # Create an empty fractional_mass column with default values (e.g., NaN)
    iso['fractional_mass'] = np.full(len(iso), np.nan)

    # Loop over the ages
    for j, age in enumerate(age_log_grid):
        # Find the index of the age in the isochrone file, array of all ages that match, 
        age_index = np.where(iso['log10_isochrone_age_yr'] == age)[0]
        
        # Remove models where mass loss becomes significant.
        # This is where the initial mass is greater than the initial mass associated with the greatest star_mass at a given age
        
        # Maximum star mass 
        max_star_mass = np.max(iso['star_mass'][age_index])

        # Index of maximum star mass, this is the index in the array for the specific age. 
        max_star_mass_index = np.argmax(iso['star_mass'][age_index])

        # Find index of the the first apperence of the specific age in the isochrone file
        age_first_index = np.where(iso['log10_isochrone_age_yr'] == age)[0][0]

        # Find the index of the maximum star mass in the isochrone file
        max_star_mass_index_in_file = age_first_index + max_star_mass_index
        
        # Calculate the fractional mass and add to file
        fractional_mass = iso['star_mass'][age_index] / max_star_mass
        iso['fractional_mass'][age_index] = fractional_mass
        
        # Remove rows after the maximum star mass index with the same age
        iso.remove_rows(np.arange(max_star_mass_index_in_file+1, age_first_index+len(age_index)-1, 1))

        # Recompute age_index after row removal
        age_index = np.where(iso['log10_isochrone_age_yr'] == age)[0]

        # Interpolate to populate the defined fractional mass grid
        log_Teff[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['log_Teff'][age_index])
        log_L[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['log_L'][age_index])
        star_mass[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['star_mass'][age_index])
        Gaia_G_EDR3[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['Gaia_G_EDR3'][age_index])    
        Gaia_BP_EDR3[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['Gaia_BP_EDR3'][age_index])
        Gaia_RP_EDR3[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['Gaia_RP_EDR3'][age_index])



# Package into FITS file format
hdus = fits.HDUList()
hdus.append(fits.PrimaryHDU())
hdus.append(fits.ImageHDU(log_Teff, name='log_Teff'))
hdus.append(fits.ImageHDU(log_L, name='log_L'))
hdus.append(fits.ImageHDU(star_mass, name='star_mass'))
hdus.append(fits.ImageHDU(Gaia_G_EDR3, name='Gaia_G_EDR3'))
hdus.append(fits.ImageHDU(Gaia_BP_EDR3, name='Gaia_BP_EDR3'))
hdus.append(fits.ImageHDU(Gaia_RP_EDR3, name='Gaia_RP_EDR3'))
# Add the age and metallicity and fractional mass as one dimensional arrays
hdus.append(fits.ImageHDU(age_log_grid, name='age_grid'))
hdus.append(fits.ImageHDU(metallicity_grid, name='metallicity_grid'))
hdus.append(fits.ImageHDU(fractional_mass_grid, name='fractional_mass_grid'))
# Write to fits file
hdus.writeto('isochrone_models.fits', overwrite=True)