"""
We will extract the data from the zip file and package it into a single fits file.
"""
import astropy.io.fits as fits
from astropy.table import Table
import numpy as np
import os
import glob
from urllib.request import urlretrieve

already_downloaded = True

# Make a downloads directory if it doesn't exist
if not os.path.exists('downloads'):
    os.mkdir('downloads')

# Work in the downloads directory, unless we are already there
if os.getcwd().split('/')[-1] != 'downloads':
    os.chdir('downloads')

if not already_downloaded:
    # Download the data from the MIST website, using urlretrieve
    metstrs = ['m1.50', 'm1.25', 'm1.00', 'm0.75', 'm0.50', 'm0.25', 'p0.00', 'p0.25', 'p0.50']
    for metstr in metstrs:
        fname = 'MIST_v1.2_feh_' + metstr + '_afe_p0.0_vvcrit0.0_EEPS.txz'
        url = 'https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/' + fname
        print('Downloading ' + url)
        urlretrieve(url, fname)  
    # Extract the data from the gzipped tar files in the downloads directory. 
    txz_files = glob.glob('*.txz')
    for txz_file in txz_files:
        os.system('tar -xvf ' + txz_file)  # Extract the tar file
    
# Package the data into a single fits file, with a 2D array for fractional_star_age, and
# 3D arrays for star_mass, log_Teff, log_L. The axes of the 3D arrays are initial_star_mass, metallicity,
# and fractional star age. There is also a 2D array for max_age, which is the maximum star age for each
# star mass and metallicity.

# Lets start by defining the grid in fractional star age. For a solar-mass star, resolving
# the upper giant branch needs 1 part in 1000.
fractional_star_age = np.arange(0.001, 1, 0.0005)
dirs = glob.glob('MIST_v1.2_feh_*EEPS')

# Find the metallicity grid strings from the filenames
metallicity_strings = [dir.split('MIST_v1.2_feh_')[1][:5] for dir in dirs]

# Define the metallicity grid, where the first character "m" or "p" is the sign of the metallicity
# and the next 3 characters are the metallicity in dex.
metallicity_signs = np.array([1 if metallicity_string[0] == 'p' else -1 for metallicity_string in metallicity_strings])
metallicity_grid = np.array([float(metallicity_string[1:]) for metallicity_string in metallicity_strings]) * metallicity_signs

# Sort the metallicity grid, so that it is in increasing order, and the metallicity_strings are in the same order.
sort_indices = np.argsort(metallicity_grid)
metallicity_grid = metallicity_grid[sort_indices]
metallicity_strings = np.array(metallicity_strings)[sort_indices]
dirs = np.array(dirs)[sort_indices]

# From the first directory, find the star mass grid.
model_files = glob.glob(dirs[0] + '/*eep')
# The star mass grid is in the file names, in units of 0.01 solar masses (or 10MJ)
mass_grid_10MJ = np.array([int(model_file.split('EEPS/')[1][:5]) for model_file in model_files])
# Restrict to masses between 0.5 and 5 solar masses
mass_grid_10MJ = mass_grid_10MJ[(mass_grid_10MJ >= 50) & (mass_grid_10MJ <= 500)]
# Sort the star mass grid
mass_grid_10MJ = np.sort(mass_grid_10MJ)
# Now we can define the star mass grid in solar masses
mass_grid = mass_grid_10MJ / 100

# First we define the 2D array for max_age
max_age = np.zeros((len(mass_grid), len(metallicity_grid)))

# Now we can define the 3D arrays for star_mass, log_Teff, log_L
star_mass = np.zeros((len(mass_grid), len(metallicity_grid), len(fractional_star_age)))
log_Teff = np.zeros((len(mass_grid), len(metallicity_grid), len(fractional_star_age)))
log_L = np.zeros((len(mass_grid), len(metallicity_grid), len(fractional_star_age)))

# Loop over the metalicity strings which define the directories
for i, metallicity_string in enumerate(metallicity_strings):
    this_dir = f'MIST_v1.2_feh_{metallicity_string}_afe_p0.0_vvcrit0.0_EEPS/'
    # Loop over the star masses
    for j, mass_10MJ in enumerate(mass_grid_10MJ):
        # Load the data from the files
        track_file = this_dir + f'{mass_10MJ:05d}M.track.eep'
        track_table = Table.read(track_file, format='ascii.commented_header', header_start=-1)
        # Find the maximum star age
        max_age[j, i] = track_table['star_age'][-1]
        # Interpolate the log_Teff, log_L and star_mass to the fractional_star_age grid
        log_Teff[j, i] = np.interp(fractional_star_age, track_table['star_age'] / max_age[j, i], track_table['log_Teff'])
        log_L[j, i] = np.interp(fractional_star_age, track_table['star_age'] / max_age[j, i], track_table['log_L'])
        star_mass[j, i] = np.interp(fractional_star_age, track_table['star_age'] / max_age[j, i], track_table['star_mass'])

# Change back to the parent directory
os.chdir('..')

# Now we can package the data into a fits file
hdus = fits.HDUList()
hdus.append(fits.PrimaryHDU())
hdus.append(fits.ImageHDU(max_age, name='max_age'))
hdus.append(fits.ImageHDU(star_mass, name='star_mass'))
hdus.append(fits.ImageHDU(log_Teff, name='log_Teff'))
hdus.append(fits.ImageHDU(log_L, name='log_L'))
# Include the initial star mass and metallicity grids as one dimensional arrays
hdus.append(fits.ImageHDU(mass_grid, name='mass_grid'))
hdus.append(fits.ImageHDU(metallicity_grid, name='metallicity_grid'))
# Include the fractional star age grid as a one dimensional array
hdus.append(fits.ImageHDU(fractional_star_age, name='fractional_star_age'))
# Write the fits file
hdus.writeto('MIST_v1.2.fits', overwrite=True)

