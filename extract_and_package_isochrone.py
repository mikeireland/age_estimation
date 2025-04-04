# Download and extract isochrone model from zip file and package into a single FITS file
import astropy.io.fits as fits
from astropy.table import Table
import numpy as np
import os
import glob
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

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
    # Define the data directory (update this path to match your actual directory!)
    data_path = r'C:\Users\cathe\ANU-Honours\MIST model fitting\data'
    
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
# Somes of these only appear ones - maybe an floating point error. 
age_log_grid = np.array(np.unique(iso['log10_isochrone_age_yr']))

# Find array of all initial masses, find initial masses that are unique within 0.1 percent to avoid floating point errors
initial_mass_grid = np.array(np.unique(iso['initial_mass']))


# For each row find the initial_mass, 
# find all other rows with the same initial_mass value, 
# find the maxmimum star_mass for that initial_mass value, 
# and divide the star_mass in the working by that maximum star_mass value to get the fractional_mass.

# Define function to find the maximum star_mass 
def find_maximum_star_mass(iso):

    # Initialise list to store maximum star_mass
    star_max_mass_list = []

    for index, initial_mass in enumerate(iso['initial_mass']):
        # Find indicies of other apperences of the same initial mass values in the array within 0.1 percent to avoid floating point errors
        initial_mass_index = np.where(np.isclose(iso['initial_mass'], initial_mass, rtol=0.001))[0]
        # Find the maximum star_mass for that initial mass value    
        star_max_mass = max(iso['star_mass'][initial_mass_index])
        # Append to list
        star_max_mass_list.append(star_max_mass)

    return np.array(star_max_mass_list, dtype=np.float64)

# Define function to calculate fractional mass for each row. This retuens an array of fractional mass values for each row in the isochrone file.
def calculate_fractional_mass(iso):

    # Initialise list to store fractional_mass
    fractional_mass_list = []

    for index, initial_mass in enumerate(iso['initial_mass']):
        # Find indicies of other apperences of the same initial mass values in the array within 0.1 percent to avoid floating point errors
        initial_mass_index = np.where(np.isclose(iso['initial_mass'], initial_mass, rtol=0.001))[0]
        # Find the maximum star_mass for that initial mass value    
        star_max_mass = max(iso['star_mass'][initial_mass_index])
        # Divide the star_mass in the working by that maximum star_mass value to get the fractional_mass   
        fractional_mass = iso['star_mass'][index] / star_max_mass
        # Append to list 
        fractional_mass_list.append(fractional_mass)

    return np.array(fractional_mass_list, dtype=np.float64)

# Calculate the fractional mass for the first isochrone file
fractional_mass = calculate_fractional_mass(iso)
# Add the fractional mass to the isochrone file
iso['fractional_mass'] = fractional_mass

# Look at how fractional mass changes for a given initial mass and age

    
# Find all rows with the same initial mass value
initial_mass = initial_mass_grid[90000]
initial_mass_index = np.where(np.isclose(iso['initial_mass'], initial_mass, rtol=0.001))[0]
# Find correspinding fractional mass values for that initial mass value
fractional_mass = iso['fractional_mass'][initial_mass_index]
# Find correspinding ages for that initial mass value
age = iso['log10_isochrone_age_yr'][initial_mass_index]

# Plot the fractional mass against age
plt.scatter(10**age, fractional_mass, label=f'Initial mass: {initial_mass:.2f} M_sun')
plt.xlabel('Age (yr)')
plt.ylabel('Fractional mass')
plt.legend()

plt.show()

# The fractional mass in general goes down with age, but when mass loss becomes significant there is a larger drop in fractional mass.
# Identify when mass loss becomes important by finding the point when there is a large drop in fractional mass with age.

# Define 3-D arrays to hold in input physical parameters of the models
#star_mass = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
#log_Teff = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
#log_L = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
# Define the 3-D arrays to hold the observations parameters from Gaia
#Gaia_G_EDR3 = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
#Gaia_BP_EDR3 = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))
#Gaia_RP_EDR3 = np.zeros((len(fractional_mass_grid), len(metallicity_grid), len(age_log_grid)))

# Loop over metallicity strings to find the right directory
#for i, metallicity_string in enumerate(isochrone_files):
    # Open the isochrone file
    #iso = Table.read(os.path.join(data_path, metallicity_string), format='ascii.commented_header', header_start=-1)

    # Add new column for the fractional mass
    #fractional_mass = calculate_fractional_mass(iso)
    #iso['fractional_mass'] = fractional_mass

    # Cut out the models that are after mass loss becoming significant


    # Loop over the ages
    #for j, age in enumerate(age_log_grid):
        # Find the index of the age in the isochrone file, array of all ages that match, also within 1 percent to avoid floating point errors
        #age_index = np.where(np.isclose(iso['log10_isochrone_age_yr'], age, rtol=0.01))[0]
        
        # Cut off those models at the specific age where the star_mass stars decreasing. 

        # Interpolate to populate the defined fractional mass grid
        #log_Teff[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['log_Teff'][age_index])
        #log_L[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['log_L'][age_index])
        #star_mass[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['star_mass'][age_index])
        #Gaia_G_EDR3[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['Gaia_G_EDR3'][age_index])    
        #Gaia_BP_EDR3[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['Gaia_BP_EDR3'][age_index])
        #Gaia_RP_EDR3[:,i,j] = np.interp(fractional_mass_grid, iso['fractional_mass'][age_index], iso['Gaia_RP_EDR3'][age_index])



# Package into FITS file format
#hdus = fits.HDUList()
#hdus.append(fits.PrimaryHDU())
#hdus.append(fits.ImageHDU(star_mass, name='star_mass'))
#hdus.append(fits.ImageHDU(log_Teff, name='log_Teff'))
#hdus.append(fits.ImageHDU(log_L, name='log_L'))
#hdus.append(fits.ImageHDU(Gaia_G_EDR3, name='Gaia_G_EDR3'))
#hdus.append(fits.ImageHDU(Gaia_BP_EDR3, name='Gaia_BP_EDR3'))
#hdus.append(fits.ImageHDU(Gaia_RP_EDR3, name='Gaia_RP_EDR3'))
# Add the age and metallicity and fractional mass as one dimensional arrays
#hdus.append(fits.ImageHDU(age_log_grid, name='age_grid'))
#hdus.append(fits.ImageHDU(metallicity_grid, name='metallicity_grid'))
#hdus.append(fits.ImageHDU(fractional_mass_grid, name='fractional_mass_grid'))
# Write to fits file
#hdus.writeto('isochrone_models.fits', overwrite=True)