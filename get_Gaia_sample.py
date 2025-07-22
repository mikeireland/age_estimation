import numpy as np
import astropy.io.fits as fits

# Open fits file with Gaia data 
uncut = fits.open('new_sample_all-result.fits')
uncut_data = uncut[1].data
print(f'Number of sources in the uncut data: {len(uncut_data)}')


#--------- Set precision limits of flux and parallax----------
flux_precise = (uncut_data['phot_g_mean_flux_over_error'] > 100) & (uncut_data['phot_bp_mean_flux_over_error'] > 100) & (uncut_data['phot_rp_mean_flux_over_error'] > 100) & (uncut_data['parallax']/uncut_data['parallax_error'] > 100)
flux_precise_cut = uncut_data[flux_precise]
print(f'Number of sources after flux precision limits: {len(flux_precise_cut)}')



# ------- Retain sources that have a spectroscopic metallicity and effective temperature---------
mh_spec_mask = np.isnan(flux_precise_cut['mh_gspspec']) | np.isnan(flux_precise_cut['teff_gspspec'])
mh_spec_cut = flux_precise_cut[~mh_spec_mask]
print(f'number of sources with a spectroscopic metallicity and effective temperature: {len(mh_spec_cut)}')




#--------- Remove unresolved binary sources ----------
#binary_mask = (uncut_data['ruwe'] < 1.2) & (uncut_data['non_single_star'] == 0)
#binary_cut = uncut_data[binary_mask]
binary_mask = (mh_spec_cut['ruwe'] < 1.2) & (mh_spec_cut['non_single_star'] == 0)
binary_cut = mh_spec_cut[binary_mask]
print(f'Number of sources after binary removal: {len(binary_cut)}')


#--------- Remove horizontal branch stars that are brighter than 0.5 in K mag ------------
# extract the absolute G magnitude and effective termperature 
Gmag = binary_cut['phot_g_mean_mag'] + 5*np.log10(binary_cut['parallax']/100) - binary_cut['ag_gspphot']
teff = binary_cut['teff_gspspec']

# Convert G mag to K mag - need to write up how this function is derived
Kmag = Gmag-1.81 + 7.23*np.log10(teff/5000)

hb_mask = Kmag < 0.5
hb_cut = binary_cut[~hb_mask]
print(f'Number of sources after removing horizontal branch stars: {len(hb_cut)}')
Kmag = Kmag[~hb_mask]




# --------- Remove variable stars ---------- 
# Build function to test variability
delta_g = np.sqrt(hb_cut['phot_g_n_obs']) / hb_cut['phot_g_mean_flux_over_error']

# within bins of 0.2 apparent magnitude calculate the median, and standard deviation (dispersion) of delta_g
# centres of g magnitude bins
g_mag_bins = np.arange(min(hb_cut['phot_g_mean_mag']) + 0.1, max(hb_cut['phot_g_mean_mag']), 0.2)

# initialise variability mask 
vary_mask = np.zeros_like(hb_cut['phot_g_mean_mag'], dtype=bool)

for bin_center in g_mag_bins:
    bin_filter = np.abs(hb_cut['phot_g_mean_mag'] - bin_center) < 0.1
    bin_delta_g = delta_g[bin_filter]

    if len(bin_delta_g) > 5:  # Only process bins with enough data
        median = np.median(bin_delta_g)
        dispersion = np.std(bin_delta_g)
        
        vary_stat = (bin_delta_g - median) / dispersion
        bin_variable = vary_stat > 10

        # Update arrays
        vary_mask[bin_filter] = bin_variable

# Apply the variability mask to the data
vary_cut = hb_cut[~vary_mask]
Kmag = Kmag[~vary_mask]
print(f'Number of sources after variability cut: {len(vary_cut)}')

# Extract the G band magnitude and colour
Gmag = vary_cut['phot_g_mean_mag'] + 5 * np.log10(vary_cut['parallax'] / 100) - vary_cut['ag_gspphot']
bp_rp = vary_cut['phot_bp_mean_mag'] - vary_cut['phot_rp_mean_mag'] - vary_cut['ebpminrp_gspphot']




#------------ Remove sources less luminous than the oldest isochrone ------------
# Create the mask for high metallicity
high_metallicity = vary_cut['mh_gspspec'] > -1.5

# Initialize G_cut as all 3.8
G_cut = np.full(len(vary_cut), 3.8)

# Apply the linear relation for higher metallicity stars 
G_cut[high_metallicity] = vary_cut['mh_gspspec'][high_metallicity] * 0.3 + 4.5

# Now, select sources that are dimmer than the subgiant branch cut
dim_G = Gmag > G_cut

# Apply the mask to filter out dim sources
vary_cut = vary_cut[~dim_G]
Kmag = Kmag[~dim_G]





#---------- Make giant selection from XR sections in kmag and teff space -----------
# metallicity Fe/H points for the slopes and zero points
feh = np.arange(-2.2, 0.5, 0.2)
# Add -2.5 to the beginning of the feh array to match the XR table
feh = np.insert(feh, 0, -2.5)

# list slope1 and zpt1 from XR
slope1 = [-0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.005, -0.005, -0.005]
zpt1 = [24.2, 24.2, 24, 23.8, 23.6, 23.4, 23.2, 23, 22.6, 22.2, 22, 24.25, 26.5, 26.25, 26]

# list slope2 and zpt2 from XR
slope2 = [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.00125, -0.00125, -0.0014, -0.0014, -0.0014, -0.0014, -0.0014]
zpt2 = [8.95, 8.9, 8.85, 8.8, 8.75, 8.7, 8.65, 10, 9.9, 10.7, 10.5, 10.3, 10.2, 10.1, 10]

# Make interpolation function for slopes and zero points as function of [Fe/H]
slope1_interp = np.interp(vary_cut['mh_gspspec'], feh, slope1)
slope2_interp = np.interp(vary_cut['mh_gspspec'], feh, slope2)
zpt1_interp = np.interp(vary_cut['mh_gspspec'], feh, zpt1)
zpt2_interp = np.interp(vary_cut['mh_gspspec'], feh, zpt2)

# make giant cut between the linear functions
giants = (Kmag>slope1_interp*vary_cut['teff_gspspec'] + zpt1_interp) & (Kmag<slope2_interp*vary_cut['teff_gspspec'] + zpt2_interp) 
giant_cut = vary_cut[giants]
print(f'Number of giants: {len(giant_cut)}')

import random
idx = np.array(random.sample(range(len(giant_cut)), 10000))
sample = giant_cut[idx]

# Build fits file with only the giants
giant_fits = fits.BinTableHDU(data=sample)
giant_fits.writeto('gaia_giants_test_sample.fits', overwrite=True)


# toggle to make plot of selected giants on colour magnitude diagram 
make_plots = False

if make_plots == True:
    import matplotlib.pyplot as plt
    
    # Make masks for metallicity bins
    p00_mask = (np.abs(vary_cut['mh_gspspec']) < 0.05)
    m05_mask = (np.abs(vary_cut['mh_gspspec'] + 0.5) < 0.05)
    m10_mask = (np.abs(vary_cut['mh_gspspec'] + 1.0) < 0.05)
    m15_mask = (np.abs(vary_cut['mh_gspspec'] + 1.5) < 0.05)
    
    # Set up figure 
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Define the parameter to colour the giants
    ag_err = np.maximum(np.abs(vary_cut['ag_gspphot'] - vary_cut['ag_gspphot_lower']), np.abs(vary_cut['ag_gspphot_upper'] - vary_cut['ag_gspphot']))
    mg_err = np.sqrt((5/(2*np.log(10)) * 1/vary_cut['phot_g_mean_flux_over_error'])**2 + (5/np.log(10) * vary_cut['parallax_error']/vary_cut['parallax'])**2 + (ag_err)**2)
    mg_over_error  = mg_err/Gmag

    redd_error = np.maximum(vary_cut['ebpminrp_gspphot'] - vary_cut['ebpminrp_gspphot_lower'], vary_cut['ebpminrp_gspphot_upper'] - vary_cut['ebpminrp_gspphot'])

    temp_err = np.maximum(vary_cut['teff_gspphot'] - vary_cut['teff_gspphot_lower'], vary_cut['teff_gspphot_upper'] - vary_cut['teff_gspphot'])
    temp_over_err = np.log10(vary_cut['teff_gspphot']/temp_err)

    # Set colour scale of graphs
    colour = temp_over_err
    cb_label = 'Photometric effective temperature over error (log)'

    ax1.scatter(bp_rp[p00_mask & ~giants], Gmag[p00_mask & ~giants], c='silver', s=1, alpha=0.25)
    p00_giants = ax1.scatter(bp_rp[p00_mask & giants], Gmag[p00_mask & giants], 
                            c=colour[p00_mask & giants], cmap='jet',
                            #c='indigo', 
                            s=1)
    fig.colorbar(p00_giants, ax=ax1, label=cb_label)

    ax2.scatter(bp_rp[m05_mask & ~giants], Gmag[m05_mask & ~giants], c='silver', s=1, alpha=0.25)
    m05_giants = ax2.scatter(bp_rp[m05_mask & giants], Gmag[m05_mask & giants],
                            c=colour[m05_mask & giants], cmap='jet',
                            #c='indigo', 
                            s=1)
    fig.colorbar(m05_giants, ax=ax2, label=cb_label)


    ax3.scatter(bp_rp[m10_mask & ~giants], Gmag[m10_mask & ~giants], c='silver', s=1, alpha=0.25)
    m10_giants = ax3.scatter(bp_rp[m10_mask & giants], Gmag[m10_mask & giants],
                            c=colour[m10_mask & giants], cmap='jet',
                            #c='indigo', 
                            s=1)
    fig.colorbar(m10_giants, ax=ax3, label=cb_label)


    ax4.scatter(bp_rp[m15_mask & ~giants], Gmag[m15_mask & ~giants], c='silver', s=1, alpha=0.25)
    m15_giants = ax4.scatter(bp_rp[m15_mask & giants], Gmag[m15_mask & giants],
                            c=colour[m15_mask & giants], cmap='jet',
                            #c='indigo', 
                            s=1)
    fig.colorbar(m15_giants, ax=ax4, label=cb_label)


    # Set limits 
    for ax in axs.flatten():
        ax.set_ylim(0.5,6)

    ax1.set_xlim(0.2, 1.3)
    ax2.set_xlim(0.0, 1.2)
    ax3.set_xlim(-0.1, 1.2)
    ax4.set_xlim(-0.1, 1.1)

    from astropy.table import Table
    import os

    # Add iochrones to plot 
    iso_p00 = Table.read(os.path.join(os.getcwd(), 'data', 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd'), format='ascii.commented_header', header_start=-1)
    iso_m05 = Table.read(os.path.join(os.getcwd(), 'data', 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd'), format='ascii.commented_header', header_start=-1)
    iso_m10 = Table.read(os.path.join(os.getcwd(), 'data', 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd'), format='ascii.commented_header', header_start=-1)
    iso_m15 = Table.read(os.path.join(os.getcwd(), 'data', 'MIST_v1.2_feh_m1.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd'), format='ascii.commented_header', header_start=-1)

    # Set ages of isochrones to plot
    ages = np.arange(9.1, 10.3, 0.2)

    for age in ages:
    
        # make filter based on age 
        age_indices_p00 = np.where(np.abs(iso_p00['log10_isochrone_age_yr'] - age) < 0.001)[0]
        age_indices_m05 = np.where(np.abs(iso_m05['log10_isochrone_age_yr'] - age) < 0.001)[0]
        age_indices_m10 = np.where(np.abs(iso_m10['log10_isochrone_age_yr'] - age) < 0.001)[0]
        age_indices_m15 = np.where(np.abs(iso_m15['log10_isochrone_age_yr'] - age) < 0.001)[0]

    
        # get isochrone for this age
        single_iso_p00 = iso_p00[age_indices_p00]
        single_iso_m05 = iso_m05[age_indices_m05]
        single_iso_m10 = iso_m10[age_indices_m10]
        single_iso_m15 = iso_m15[age_indices_m15]

        # remove rows in isochrone with larger starting mass than the maximum stellar mass
        max_mass_index_p00 = np.argmax(single_iso_p00['star_mass'])
        max_mass_index_m05 = np.argmax(single_iso_m05['star_mass'])
        max_mass_index_m10 = np.argmax(single_iso_m10['star_mass'])
        max_mass_index_m15 = np.argmax(single_iso_m15['star_mass'])
    

        # cut out rows after the maximum star_mass
        cut_iso_p00 = single_iso_p00[:max_mass_index_p00 + 1]
        cut_iso_m05 = single_iso_m05[:max_mass_index_m05 + 1]
        cut_iso_m10 = single_iso_m10[:max_mass_index_m10 + 1]
        cut_iso_m15 = single_iso_m15[:max_mass_index_m15 + 1]

        # plot in colour space
        ax1.plot(cut_iso_p00['Gaia_BP_EDR3'] - cut_iso_p00['Gaia_RP_EDR3'], cut_iso_p00['Gaia_G_EDR3'], c='black')
        ax2.plot(cut_iso_m05['Gaia_BP_EDR3'] - cut_iso_m05['Gaia_RP_EDR3'], cut_iso_m05['Gaia_G_EDR3'], c='black')
        ax3.plot(cut_iso_m10['Gaia_BP_EDR3'] - cut_iso_m10['Gaia_RP_EDR3'], cut_iso_m10['Gaia_G_EDR3'], c='black')
        ax4.plot(cut_iso_m15['Gaia_BP_EDR3'] - cut_iso_m15['Gaia_RP_EDR3'], cut_iso_m15['Gaia_G_EDR3'], c='black')

    for ax in axs.flatten():
        # Invert y axis
        ax.invert_yaxis()

        # Add axis titles
        ax.set_ylabel('G Absolute Magnitude')
        ax.set_xlabel('BP - RP Colour')


    # Set titles
    ax1.set_title('Metallicity [Fe/H] ~ 0.00')
    ax2.set_title('Metallicity [Fe/H] ~ -0.50')
    ax3.set_title('Metallicity [Fe/H] ~ -1.00')
    ax4.set_title('Metallicity [Fe/H] ~ -1.50')


    # Show the plot
    plt.tight_layout()
    plt.savefig('spec_mh_.png')






