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
                        & (uncut_data['parallax_over_error'] < 100) & (uncut_data['parallax_over_error'] > 50)
                        ]

# limit to specific [Fe/H] range
#uncut_data = uncut_data[(uncut_data['mh_gspphot'] > -0.2) & (uncut_data['mh_gspphot'] < 0.0)]
# check number of stars
print(f'Number of stars after binary removal: {len(uncut_data)}')

# Plot historgrapm of FE/H for all gaia sources before and after the precision cuts


# extract the absolute G magnitude and effective termperature 
Gmag = uncut_data['phot_g_mean_mag'] + 5*np.log10(uncut_data['parallax']/100) - uncut_data['ag_gspphot']
teff = uncut_data['teff_gspphot']

# Convert G mag to K mag - need to write up how this function is derived
Kmag = Gmag-1.81 + 7.23*np.log10(teff/5000)

# discard stars with an absolute K magnitue brighter than 0.5
HB_star = Kmag<0.5
uncut_data = uncut_data[~HB_star]

Gmag = Gmag[~HB_star]
Kmag = Kmag[~HB_star]
teff = teff[~HB_star]

print(f'Number of star after Horizontal branch star cleaning: {len(Kmag)}')

# Cut sources that are dimmer than the 20 Gyr subgiant branch 
# Create the mask for high metallicity
high_metallicity = uncut_data['mh_gspphot'] > -1.5

# Initialize G_cut as all 3.8
G_cut = np.full(len(uncut_data), 3.8)

# Apply the linear relation for higher metallicity stars (no .loc needed)
G_cut[high_metallicity] = uncut_data['mh_gspphot'][high_metallicity] * 0.45 + 4.5

# Now, select sources that are dimmer than the subgiant branch cut
dim_G = Gmag > G_cut

# Apply the mask to filter out dim sources
uncut_data = uncut_data[~dim_G]
Kmag = Kmag[~dim_G]
teff = teff[~dim_G]

# select the giants (as a function of [Fe/H])

# metallicity Fe/H points for the slopes and zero points
feh = np.arange(-2.5, 0.4, 0.2)

# list slope1 and zpt1 from XR
slope1 = [-0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.005, -0.005, -0.005]
zpt1 = [24.2, 24.2, 24, 23.8, 23.6, 23.4, 23.2, 23, 22.6, 22.2, 22, 24.25, 26.5, 26.25, 26]

# list slope2 and zpt2 from XR
slope2 = [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.00125, -0.00125, -0.0014, -0.0014, -0.0014, -0.0014, -0.0014]
zpt2 = [8.95, 8.9, 8.85, 8.8, 8.75, 8.7, 8.65, 10, 9.9, 10.7, 10.5, 10.3, 10.2, 10.1, 10]

# Make interpolation function for slopes and zero points as function of [Fe/H]
slope1_interp = np.interp(uncut_data['mh_gspphot'], feh, slope1)
slope2_interp = np.interp(uncut_data['mh_gspphot'], feh, slope2)
zpt1_interp = np.interp(uncut_data['mh_gspphot'], feh, zpt1)
zpt2_interp = np.interp(uncut_data['mh_gspphot'], feh, zpt2)

# limit to giants between -0.2 and 0.0 [Fe/H]
giants = (Kmag>slope1_interp*teff + zpt1_interp) & (Kmag<slope2_interp*teff + zpt2_interp) 

# Check number of giants
giant_data = uncut_data[giants]
print(f'Number of giants: {len(giant_data)}')

# Build fits file with only the giants
giant_data = uncut_data[giants]
giant_fits = fits.BinTableHDU(data=giant_data)
giant_fits.writeto('gaia_giants_test.fits', overwrite=True)


