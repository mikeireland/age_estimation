import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

# open fits file with Gaia data

uncut = fits.open('gaia_uncut.fits')
uncut_data = uncut[1].data

# Remove unresolved binary sources 
uncut_data = uncut_data[(uncut_data['ruwe'] < 1.2) & (uncut_data['non_single_star'] == 0)]

# 1 percent precision on parallax, and fluxes
uncut_data = uncut_data[(uncut_data['phot_g_mean_flux_over_error'] > 100) 
                        & (uncut_data['phot_bp_mean_flux_over_error'] > 100) 
                        & (uncut_data['phot_rp_mean_flux_over_error'] > 100) 
                        & (uncut_data['parallax_over_error'] > 100)]

# limit to specific [Fe/H] range
uncut_data = uncut_data[(uncut_data['mh_gspphot'] > -0.2) & (uncut_data['mh_gspphot'] < 0.0)]
# check number of stars
print(f'Number of stars after cuts: {len(uncut_data)}')

# extract the absolute G magnitude and effective termperature 
Gmag = uncut_data['phot_g_mean_mag'] + 5*np.log10(uncut_data['parallax']/100) 
teff = uncut_data['teff_gspphot']

# Convert G mag to K mag
Kmag = Gmag-1.81 + 7.23*np.log10(teff/5000)

# select the giants (as a function of [Fe/H])

# list slope1 and zpt1 from XR
slope1 = [-0.005, -0.005, -0.005, -0.0045, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004]
zpt1 = [26, 26.25, 26.5, 24.25, 22, 22.2, 22.6, 23, 23.2, 23.4, 23.6, 23.8, 24, 24.2, 24.2]

# list slope2 and zpt2 from XR
slope2 = [-0.0014, -0.0014, -0.0014, -0.0014, -0.0014, -0.0014, -0.00125, -0.00125, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001]
zpt2 = [10, 10.1, 10.2, 10.3, 10.5, 10.7, 9.9, 10, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95]

# limit to giants between -0.2 and 0.0 [Fe/H]
giants = (Kmag>slope1[2]*teff + zpt1[2]) & (Kmag<slope2[2]*teff + zpt2[2]) 

# Check number of giants
giant_data = uncut_data[giants]
print(f'Number of giants: {len(giant_data)}')

# Build fits file with only the giants
giant_data = uncut_data[giants]
giant_fits = fits.BinTableHDU(data=giant_data)
giant_fits.writeto('gaia_giants.fits', overwrite=True)


