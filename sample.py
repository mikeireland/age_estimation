"""
SELECT g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.phot_g_mean_mag,tb.*
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.nss_two_body_orbit AS tb ON tb.source_id = g.source_id
WHERE tb.period_error/tb.period < 0.02 AND g.phot_g_mean_mag < 13 AND 
(tb.nss_solution_type='Orbital' OR tb.nss_solution_type='OrbitalTargetedSearch' OR tb.nss_solution_type='OrbitalTargetedSearchValidated' OR tb.nss_solution_type='AstroSpectroSB1')

'Orbital' only gives 103 sources
Everything above gives 187 sources now!
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
from astropy.coordinates import Angle
import astropy.units as u

#dd = fits.getdata('1742191636324O-result.fits',1) #Orbital Only
dd = fits.getdata('1742325284998O-result.fits',1) #All Orbits

abs_g_mag = dd['phot_g_mean_mag'] + 5*np.log10(dd['parallax']/100)
bp_rp = dd['phot_bp_mean_mag'] - dd['phot_rp_mean_mag']
sep = (dd['period']/365)**(2/3)*dd['parallax']
ww= np.where((dd['phot_g_mean_mag']<10) & (abs_g_mag < 3.1) & (bp_rp>0.65) & \
	(abs_g_mag>2*bp_rp-1.1) & (abs_g_mag<3*bp_rp + 0.7) & \
	( sep > 4) & (dd['ra']<240) & (dd['dec']<20) &
	( dd['period_error']/dd['period']<0.01) &
	( dd['parallax_error']/dd['parallax']<0.007) )[0]

plt.clf()
plt.hist2d(bp_rp, abs_g_mag, bins=40,range=[[0,2],[-0.5,5.5]], cmap='viridis', cmin=5, norm='log')
plt.plot(bp_rp[ww], abs_g_mag[ww], 'r.')
plt.axis([0,2,5.5,-0.5])
plt.xlabel('Bp-Rp')
plt.ylabel('G')
plt.tight_layout()

outfile = open('targets.csv', 'w')
outfile.write('Name, RA, Dec, Mag\n')
ss = np.argsort(dd[ww]['ra'])
last_name = ''
for i in ww[ss]:
	name = Simbad.query_object("Gaia DR3" + str(dd[i]['source_id']))['MAIN_ID'][0]
	if name == last_name:
		continue
	last_name = name
	ras = Angle(dd[i]['ra']*u.deg).to_string(unit=u.hour, sep=':', pad=True, precision=3)
	decs = Angle(dd[i]['dec']*u.deg).to_string(unit=u.deg, sep=':', pad=True, precision=2, alwayssign=True)
	Gmag = dd[i]['phot_g_mean_mag']
	outfile.write(f'{name}, {ras}, {decs}, G={Gmag:.2f}\n')
outfile.close()