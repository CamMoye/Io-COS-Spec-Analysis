# -*- coding: utf-8 -*-
'''
Written Summer 2020 by Cameron Moye, University of Maryland, College Park

This code is broken up into four sections. The first is all the custom
functions used throughout the code. The second coadds and produces a plot of
the full spectrum with and without doppler correction and with the SUMER solar
atlas overlayed. The third produces the spectrum broken into three segments
with each peak identified. The fourth coadds the isolated neutral species lines
and plots them against the off-disk observations.

IMPORTANT NOTES: The coadd is not correct so absolute brightness will be wrong
but the line shape should be accurate. Make sure each spectrum x1d file is
within your current working directory before running the program.
'''

# Necessary libraries: Matplotlib, Astropy, Scipy, Numpy, Math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from astropy.io import fits
import scipy.constants as scicon
import numpy as np
from math import sqrt
import scipy.io as sio

c = 2.9979E+5
c1 = 2.9979E+8

###Section 1: Custom functions for use in the code

#readfile is a function that takes the .fits spectrum file, extracts the
#wavelength and flux from both detector channels, and returns them as arrays
def readfile(filename):
    data = fits.getdata(filename)
    
    fuvA = data[data['segment'] == 'FUVB']
    wlA = fuvA['wavelength'].flatten()
    flxA = fuvA['flux'].flatten()
    
    fuvB = data[data['segment'] == 'FUVA']
    wlB = fuvB['wavelength'].flatten()
    flxB = fuvB['flux'].flatten()
    return wlA, flxA, wlB, flxB

#dscorr takes wavelength and relative velocity and corrects for the
#heliocentric doppler shift of Io. Returns array of corrected wavelengths
def dscorr(wl, v):
    del_wl = (wl*(v/c))
    corr_wl = wl-del_wl
    return corr_wl

#fluxconn takes flux and wavelength and converts from flux units to Rayleighs/A
#Utilizes the Aperture Filling method for conversion
def fluxcon(flx,wl):
    conversion = ((wl*1E-10)/(scicon.h*c1))*(1/1E+7)
    conversion_AF = (1/1E+6)*((4*scicon.pi*(206265**2))/(scicon.pi*(2.5/2)**2))
    flx = flx*conversion*conversion_AF
    return flx

#smooth takes the intensity and pixel width and smooths the intensity with a
#running average
def smooth(intensity, px):
    box = np.ones(px)/px
    intens_smooth = np.convolve(intensity, box, mode='same')
    return intens_smooth

#spec_combiner combines the A and B detector channels into one array
def spec_combiner(wlA, intsA, wlB, intsB):
    specA = np.zeros((16384, 2))
    specA[:,0] = wlA
    specA[:,1] = intsA
    
    specB = np.zeros((16384, 2))
    specB[:,0] = wlB
    specB[:,1] = intsB

    spec = np.append(specA, specB, axis = 0)
    spec = spec[spec[:,0].argsort()]
    
    return spec

#spec_coadd is the current method for coadding the spectra. This is what needs
#more work. Currently takes each spec and weights the intensity by the square
#root of the exposure time, then combines each spec into one array, sorts by
#wavelength, and divides by the sum of the weights
def spec_coadd(wl1, ints1, wl2, ints2, wl3, ints3, wl6, ints6, wl7, ints7,\
               wl8, ints8, wl9, ints9):
    
    spec1 = np.zeros((32768,2))
    spec1[:,0] = wl1
    spec1[:,1] = ints1*sqrt(2600.16)
    
    spec2 = np.zeros((32768,2))
    spec2[:,0] = wl2
    spec2[:,1] = ints2*sqrt(2740.19)
    
    spec3 = np.zeros((32768, 2))
    spec3[:,0] = wl3
    spec3[:,1] = ints3*sqrt(248.992)
    
    spec6 = np.zeros((32768, 2))
    spec6[:,0] = wl6
    spec6[:,1] = ints6*sqrt(249.984)
    
    spec7 = np.zeros((32768,2))
    spec7[:,0] = wl7
    spec7[:,1] = ints7*sqrt(2543.17)
    
    spec8 = np.zeros((32768, 2))
    spec8[:,0] = wl8
    spec8[:,1] = ints8*sqrt(1753.18)
    
    spec9 = np.zeros((32768, 2))
    spec9[:,0] = wl9
    spec9[:,1] = ints9*sqrt(2543.17)
    
    spec = np.append(spec1, spec2, axis = 0)
    spec = np.append(spec, spec3, axis = 0)
    spec = np.append(spec, spec6, axis = 0)
    spec = np.append(spec, spec7, axis = 0)
    spec = np.append(spec, spec8, axis = 0)
    spec = np.append(spec, spec9, axis = 0)
    spec = spec[spec[:,0].argsort()]
    
    spec[:,1] = spec[:,1]/(sqrt(2600.16)+sqrt(2740.19)+sqrt(248.992)+\
                sqrt(249.984)+sqrt(2543.17)+sqrt(1753.18)+sqrt(2543.17))
    
    return spec

#readfile_sav is to read the SUMER solar atlas .sav file and extract the data
def readfile_sav(filename):
    data = sio.readsav(filename)
    return data


###Section 2: Processing and Coadding each Spectrum

'''
For each spectrum, the data is read in, then the doppler shift is corrected and
the flux is converted to intensity. Then the two detector channels are combined
and smoothed with a running average 7 pixels wide. Doppler velocities are
relative to the Sun due to Hubble pipeline corrections and were determined
using JPL Horizons. Both Doppler corrected and uncorrected spectra are produced
for line shape/height comparisons.
'''


# Spectrum 1 - From 2012
wlA1, flxA1, wlB1, flxB1 = readfile('lb6a03s1q_x1d.fits')
wlA1_corr = dscorr(wlA1, -3.3784512)
intsA1 = fluxcon(flxA1, wlA1)

wlB1_corr = dscorr(wlB1, -3.3784512)
intsB1 = fluxcon(flxB1, wlB1)

spec1 = spec_combiner(wlA1, intsA1, wlB1, intsB1)
spec1[:,1] = smooth(spec1[:,1], 7)

spec1_corr = spec_combiner(wlA1_corr, intsA1, wlB1_corr, intsB1)
spec1_corr[:,1] = smooth(spec1_corr[:,1], 7)


# Spectrum 2 - From 2012
wlA2, flxA2, wlB2, flxB2 = readfile('lb6a04svq_x1d.fits')
wlA2_corr = dscorr(wlA2, -4.88027575)
intsA2 = fluxcon(flxA2, wlA2)

wlB2_corr = dscorr(wlB2, -4.88027575)
intsB2 = fluxcon(flxB2, wlB2)

spec2 = spec_combiner(wlA2, intsA2, wlB2, intsB2)
spec2[:,1] = smooth(spec2[:,1], 7)

spec2_corr = spec_combiner(wlA2_corr, intsA2, wlB2_corr, intsB2)
spec2_corr[:,1] = smooth(spec2_corr[:,1], 7)


# Spectrum 3 - From 2018
wlA3, flxA3, wlB3, flxB3 = readfile('ldk201oeq_x1d.fits')
wlA3_corr = dscorr(wlA3, -17.22026405)
intsA3 = fluxcon(flxA3, wlA3)

wlB3_corr = dscorr(wlB3, -17.22026405)
intsB3 = fluxcon(flxB3, wlB3)

spec3 = spec_combiner(wlA3, intsA3, wlB3, intsB3)
spec3[:,1] = smooth(spec3[:,1], 7)

spec3_corr = spec_combiner(wlA3_corr, intsA3, wlB3_corr, intsB3)
spec3_corr[:,1] = smooth(spec3_corr[:,1], 7)


# Spectrum 4 - Off-disk Leading Spectrum from 2018
wlA4, flxA4, wlB4, flxB4 = readfile('ldk2a1ogq_x1d.fits')
wlA4 = dscorr(wlA4, -17.36526935)
intsA4 = fluxcon(flxA4, wlA4)

wlB4 = dscorr(wlB4, -17.36526935)
intsB4 = fluxcon(flxB4, wlB4)

spec4 = spec_combiner(wlA4, intsA4, wlB4, intsB4)
spec4[:,1] = smooth(spec4[:,1], 7)


# Spectrum 5 - Off-disk Trailing Spectrum from 2018
wlA5, flxA5, wlB5, flxB5 = readfile('ldk2a2v3q_x1d.fits')
wlA5 = dscorr(wlA5, 16.724891)
intsA5 = fluxcon(flxA5, wlA5)

wlB5 = dscorr(wlB5, 16.724891)
intsB5 = fluxcon(flxB5, wlB5)

spec5 = spec_combiner(wlA5, intsA5, wlB5, intsB5)
spec5[:,1] = smooth(spec5[:,1], 7)


# Spectrum 6 - From 2018
wlA6, flxA6, wlB6, flxB6 = readfile('ldk202v1q_x1d.fits')
wlA6_corr = dscorr(wlA6, 16.5563013)
intsA6 = fluxcon(flxA6, wlA6)

wlB6_corr = dscorr(wlB6, 16.5563013)
intsB6 = fluxcon(flxB6, wlB6)

spec6 = spec_combiner(wlA6, intsA6, wlB6, intsB6)
spec6[:,1] = smooth(spec6[:,1], 7)

spec6_corr = spec_combiner(wlA6_corr, intsA6, wlB6_corr, intsB6)
spec6_corr[:,1] = smooth(spec6_corr[:,1], 7)


# Spectrum 7 - From 2019
wlA7, flxA7, wlB7, flxB7 = readfile('ldxc52djq_x1d.fits')
wlA7_corr = dscorr(wlA7, 3.0180835)
intsA7 = fluxcon(flxA7, wlA7)

wlB7_corr = dscorr(wlB7, 3.0180835)
intsB7 = fluxcon(flxB7, wlB7)

spec7 = spec_combiner(wlA7, intsA7, wlB7, intsB7)
spec7[:,1] = smooth(spec7[:,1], 7)

spec7_corr = spec_combiner(wlA7_corr, intsA7, wlB7_corr, intsB7)
spec7_corr[:,1] = smooth(spec7_corr[:,1], 7)


# Spectrum 8 - From 2019
wlA8, flxA8, wlB8, flxB8 = readfile('ldxc53glq_x1d.fits')
wlA8_corr = dscorr(wlA8, 3.27246285)
intsA8 = fluxcon(flxA8, wlA8)

wlB8_corr = dscorr(wlB8, 3.27246285)
intsB8 = fluxcon(flxB8, wlB8)

spec8 = spec_combiner(wlA8, intsA8, wlB8, intsB8)
spec8[:,1] = smooth(spec8[:,1], 7)

spec8_corr = spec_combiner(wlA8_corr, intsA8, wlB8_corr, intsB8)
spec8_corr[:,1] = smooth(spec8_corr[:,1], 7)


# Spectrum 9 - From 2019
wlA9, flxA9, wlB9, flxB9 = readfile('ldxc53gnq_x1d.fits')
wlA9_corr = dscorr(wlA9, 3.27246285)
intsA9 = fluxcon(flxA9, wlA9)

wlB9_corr = dscorr(wlB9, 3.27246285)
intsB9 = fluxcon(flxB9, wlB9)

spec9 = spec_combiner(wlA9, intsA9, wlB9, intsB9)
spec9[:,1] = smooth(spec9[:,1], 7)

spec9_corr = spec_combiner(wlA9_corr, intsA9, wlB9_corr, intsB9)
spec9_corr[:,1] = smooth(spec9_corr[:,1], 7)


#Coadding the uncorrected spectra and smoothing the output w/7px running avg
uncorr_spec = spec_coadd(spec1[:,0], spec1[:,1], spec2[:,0], spec2[:,1],\
                         spec3[:,0], spec3[:,1], spec6[:,0], spec6[:,1],\
                         spec7[:,0], spec7[:,1], spec8[:,0], spec8[:,1],\
                         spec9[:,0], spec9[:,1])

uncorr_spec[:,1] = smooth(uncorr_spec[:,1], 7)

#Coadding the corrected spectra and smoothing output w/7px running avg
corr_spec = spec_coadd(spec1_corr[:,0], spec1_corr[:,1], spec2_corr[:,0],\
                       spec2_corr[:,1], spec3_corr[:,0], spec3_corr[:,1],\
                       spec6_corr[:,0], spec6_corr[:,1], spec7_corr[:,0],\
                       spec7_corr[:,1], spec8_corr[:,0], spec8_corr[:,1],\
                       spec9_corr[:,0], spec9_corr[:,1])

corr_spec[:,1] = smooth(corr_spec[:,1], 7)


#Reading in the SUMER solar atlas and scaling it for overplotting/comparison
#with the coadded Io spectra
sunspec = readfile_sav('SUMER_Quiet_Sun_Atlas.sav')
wavelength = sunspec['qs_lambda']
radiance = sunspec['qs_profil']*2E-4

#Generating a plot with Corrected/Uncorrected coadded spectra and solar atlas
plt.figure(figsize = (15,10))
plt.ylim(top = 30)
plt.xlim(left = 1124.62)
plt.xlim(right = 1471)
plt.plot(corr_spec[:,0], corr_spec[:,1], label = \
         'Hubble COS Spectrum with Correction')
plt.plot(uncorr_spec[:,0], uncorr_spec[:,1], label = \
         'Hubble COS Spectrum w/out Correction')
plt.plot(wavelength[10366:18377], radiance[10366:18377], 'g', label = \
         'Solar Atlas')

plt.title('Coadded Spectra', fontsize = 22)
plt.ylabel(r'Intensity $(Rayleighs/\AA)$', fontsize = 18)
plt.xlabel(r'Wavelength $(\AA)$', fontsize = 18)
plt.legend(fontsize = 14)
plt.tight_layout()



###Section 3: Making the three plots with line identifiers

'''
This section generates three plots. The full coadded, corrected spectrum is
broken into three segments. The weaker signals in the spectrum are scaled and
offset vertically for easier viewing of fine details in the spectrum. Lists for
vertical line identifiers were generated from wavelengths for each species
listed in the NIST database. Dashed lines indicated uncertainty. This could be
from speculating what a line is (like the O III line) or if the oscillator
energy is not listed in the NIST database, making not possible to 100% confirm
the detection of a line. This is common for S I lines.

In the future, solar spectrum should be subtracted out of these. Any edge
effects from the edge of detector segments.
'''

#Splitting the corrected spectrum into 3 segments and applying the scaling
#and offset.
spec1_wl = corr_spec[0:76459,0]
spec1_ints = corr_spec[0:76459,1]
spec1 = np.zeros((76459,2))
spec1[:,0] = spec1_wl
spec1[:,1] = (spec1_ints*100)+60

spec2_wl = corr_spec[76459:152918,0]
spec2_ints = corr_spec[76459:152918,1]
spec2 = np.zeros((76459,2))
spec2[:,0] = spec2_wl
spec2[:,1] = (spec2_ints*10)+4

spec3_wl = corr_spec[152918:229376, 0]
spec3_ints = corr_spec[152918:229376, 1]
spec3 = np.zeros((76458, 2))
spec3[:,0] = spec3_wl
spec3[:,1] = (spec3_ints*10)+5


##Lists of wavelengths for vertical line markers identifying each species

#Markers for Segment 1
ClI_lines1 = np.array([1179.293, 1188.774, 1201.353])
OI_lines1 = np.array([1152.1512])
OIII_lines1 = np.array([1197.331])
SI_lines1 = np.array([1208.85, 1211.212, 1211.38, 1212.795, 1224.424,\
                      1224.479, 1224.544, 1227.089, 1233.922, 1220.162,\
                      1218.51, 1218.571, 1218.595])
OtherSI1 = np.array([1241.905])
SII_lines1 = np.array([1204.29, 1204.335, 1226.706, 1233.922, 1234.157])
SIII_lines1 = np.array([1190.206, 1194.061, 1194.457, 1200.97, 1201.73])
Solar_lines1 = np.array([1206.5])
Tell_lines1 = np.array([1167.8, 1199.55, 1215.67])

#Markers for Segment 2
ClI_lines2 = np.array([1335.726])
OI_lines2 = np.array([1302.168, 1304.858, 1306.029])
SI_lines2 = np.array([1247.16, 1248.05, 1250.81, 1253.325, 1256.09,\
                      1262.86, 1269.2086, 1270.7821, 1295.6526, 1296.1738, \
                      1302.2270, 1302.8633, 1303.1105, 1303.4295, 1310.194, \
                      1313.2493, 1316.5423, 1323.5153, 1323.522, 1326.6432, \
                      1277.2122, 1277.1985, 1316.6183])
SII_lines2 = np.array([1250.578, 1253.805, 1259.518])
Solar_lines2 = np.array([1264.7, 1334.532, 1335.708])
Tell_lines2 = np.array([1302.168, 1306.029])
Unknown_lines2 = np.array([1261.5])

#Markers for Segment 3
ClI_lines3 = np.array([1347.24, 1351.657, 1363.447, 1379.528, 1389.693,\
                       1389.957, 1396.527])
OI_lines3 = np.array([1355.598, 1358.512])
SI_lines3 = np.array([1388.4347, 1392.5878, 1396.1122, 1401.5136, 1472.530,\
                      1381.5521, 1385.51, 1389.1538, 1409.3369, 1412.8726,\
                      1433.28, 1433.3105, 1436.9675, 1448.229, 1472.9706])
OtherSI3 = np.array([1363.033, 1425.0301, 1425.2190])
SII_lines3 = np.array([1363.384])
SIV_lines3 = np.array([1404.808, 1406.009, 1416.887, 1423.845])
Solar_lines3 = np.array([1393.8, 1402.76])


#Generating the plots for each segment
plt.figure(figsize = (15,10))
plt.xlim(left = 1132.9967)
plt.xlim(right = 1244.5087)
plt.plot(corr_spec[4131:76459,0], corr_spec[4131:76459,1],label = '_nolegend_')
plt.plot(spec1[4131:55384,0], spec1[4131:55384,1], 'g', label = '_nolegend_')
plt.plot(spec1[56998:76459,0], spec1[56998:76459,1], 'g', label = '_nolegend_')
plt.text(1140, 105, 'x100', fontsize = 20)
plt.vlines(ClI_lines1, -25, 1000, colors = 'springgreen', label = 'Cl I',\
           linewidths = 1)
plt.vlines(OI_lines1, -25, 1000, colors = 'dodgerblue', label = 'O I',\
           linewidths = 1)
plt.vlines(OIII_lines1, -25, 1000, colors = 'orangered', label = 'O III',\
           linewidth = 1, linestyle = '--')
plt.vlines(SI_lines1, -25, 1000, colors = 'firebrick', label = '_nolegend_',\
           linewidths = 1, linestyle = '--')
plt.vlines(OtherSI1, -25, 1000, colors = 'firebrick', label = 'S I',\
           linewidths = 1)
plt.vlines(SII_lines1, -25, 1000, colors = 'blueviolet', label = 'S II',\
           linewidths = 1)
plt.vlines(SIII_lines1, -25, 1000, colors = 'darkorange', label = 'S III',\
           linewidths = 1)
plt.vlines(Solar_lines1, -25, 1000, colors = 'magenta', label = 'Solar',\
           linewidths = 1)
plt.vlines(Tell_lines1, -25, 1000, colors = 'forestgreen', label = 'Telluric',\
           linewidths = 1)

plt.title('FUV Spectrum Segment 1', fontsize = 24)
plt.ylabel(r'Intensity $(Rayleighs/\AA)$', fontsize = 20)
plt.xlabel(r'Wavelength $(\AA)$', fontsize = 20)
plt.legend(loc = 2, fontsize = 18)
plt.tight_layout()
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(2))
grph.yaxis.set_minor_locator(MultipleLocator(50))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)


plt.figure(figsize = (15,10))
plt.xlim(left = 1244.5087)
plt.xlim(right = 1343.3642)
plt.plot(corr_spec[76459:152918,0], corr_spec[76459:152918,1],\
         label = '_nolegend_')
plt.plot(spec2[0:5751,0], spec2[0:5751,1], 'g', label = '_nolegend_')
plt.plot(spec2[7366:9825,0], spec2[7366:9825,1], 'g', label = '_nolegend_')
plt.plot(spec2[11228:44847,0], spec2[11228:44847,1], 'g', label = '_nolegend_')
plt.plot(spec2[48847:76459,0], spec2[48847:76459,1], 'g', label = '_nolegend_')
plt.text(1338, 7, 'x10', fontsize = 20)
plt.vlines(ClI_lines2, -2, 36, colors = 'springgreen', label = 'Cl I',\
           linewidths = 1)
plt.vlines(OI_lines2, -2, 36, colors = 'dodgerblue', label = 'O I',\
           linewidths = 1)
plt.vlines(SI_lines2, -2, 36, colors = 'firebrick', label = 'S I',\
           linewidths = 1)
plt.vlines(SII_lines2, -2, 36, colors = 'blueviolet', label = 'S II',\
           linewidths = 1)
plt.vlines(Solar_lines2, -2, 36, colors = 'magenta', label = 'Solar',\
           linewidths = 1)
plt.vlines(Tell_lines2, -2, 36, colors = 'forestgreen', label = 'Telluric',\
           linewidths = 1)
plt.vlines(Unknown_lines2, -2, 36, colors = 'slategray',\
           label = 'Unidentified', linewidth = 1)

plt.title('FUV Spectrum Segment 2', fontsize = 24)
plt.ylabel(r'Intensity $(Rayleighs/\AA)$', fontsize = 20)
plt.xlabel(r'Wavelength $(\AA)$', fontsize = 20)
plt.legend(loc = 1, fontsize = 18)
plt.tight_layout()
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(2))
grph.yaxis.set_minor_locator(MultipleLocator(1))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)


plt.figure(figsize = (15,10))
plt.xlim(left = 1343.3642)
plt.xlim(right = 1471)
plt.plot(corr_spec[152918:228260,0], corr_spec[152918:228260,1],\
         label = '_nolegend_')
plt.plot(spec3[0:7814,0], spec3[0:7814,1], 'g', label = '_nolegend_')
plt.plot(spec3[11255:56820,0], spec3[11255:56820,1], 'g', label = '_nolegend_')
plt.plot(spec3[58014:74961,0], spec3[58014:74961,1], 'g', label = '_nolegend_')
plt.plot(spec3[75241:75342,0], spec3[75241:75342,1], 'g', label = '_nolegend_')
plt.text(1456, 14, 'x10', fontsize = 20)
plt.vlines(ClI_lines3, -2, 62, colors = 'springgreen', label = 'Cl I',\
           linewidths = 1)
plt.vlines(OI_lines3, -2, 62, colors = 'dodgerblue', label = 'O I',\
           linewidths = 1)
plt.vlines(SI_lines3, -2, 62, colors = 'firebrick', label = 'S I',\
           linewidths = 1)
plt.vlines(OtherSI3, -2, 62, colors = 'firebrick', label = '_nolegend_',\
           linewidths = 1, linestyle = '--')
plt.vlines(SII_lines3, -2, 62, colors = 'blueviolet', label = 'S II',
           linewidths = 1)
plt.vlines(SIV_lines3, -2, 62, colors = 'teal', label = 'S IV', linewidths = 1)
plt.vlines(Solar_lines3, -2, 62, colors = 'magenta', label = 'Solar',\
           linewidths = 1)

plt.title('FUV Spectrum Segment 3', fontsize = 24)
plt.ylabel(r'Intensity $(Rayleighs/\AA)$', fontsize = 20)
plt.xlabel(r'Wavelength $(\AA)$', fontsize = 20)
plt.legend(loc = 1, fontsize = 18)
plt.tight_layout()
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(2))
grph.yaxis.set_minor_locator(MultipleLocator(2))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)



###Section 4: Coadding Neutral Species Lines and Comparing to Off-Disk Lines

'''
This section takes the isolated, unblended lines for each neutral species
and coadds them to produce a LSF for each. The wavelength along the horizontal
axis is also converted to km/s by choosing a center wavelength to base the
conversion for each species off of. The wavelength closest to ~1350A was
chosen for each species since that was close to the median wavelength for each.

Certain lines are included in the code but are not used in the final coadd due
to blending on one side or they exhibited some kind of self-absorption or
irregular line shape. These lines are in the code in case of future comparisons
'''

##Isolating each coadded spectrum neutral lines

#O I line at 1355.598A
OI1 = np.zeros((1444,2))
OI1[:,0] = corr_spec[160784:162228,0]
OI1[:,1] = corr_spec[160784:162228,1]

#O I line at 1358.512A
OI2 = np.zeros((1444,2))
OI2[:,0] = corr_spec[162830:164274,0]
OI2[:,1] = corr_spec[162830:164274,1]

#O I optically thick line at 1152.1512A
OI3 = np.zeros((1444,2))
OI3[:,0] = corr_spec[13019:14463,0]
OI3[:,1] = corr_spec[13019:14463,1]

#Total coadd used for plot
OI_total = OI1[:,1]+OI2[:,1]
#Coadd that includes the optically thick line
OI_total1 = OI1[:,1]+OI2[:,1]+OI3[:,1]

#Converting the wavelength to km/s using the 1355A line wavelengths
OI_center = 1355.598
for i in np.linspace(0,1443,1444, dtype = np.int64):
    del_wl = OI1[i,0]-OI_center
    wl_ratio = del_wl/OI1[i,0]
    OI1[i,0] = wl_ratio*c


#Cl I line at 1179.293A
ClI1 = np.zeros((1026,2))
ClI1[:,0] = corr_spec[30148:31174,0]
ClI1[:,1] = corr_spec[30148:31174,1]

#Cl I line at 1351.657A
ClI2 = np.zeros((1026,2))
ClI2[:,0] = corr_spec[158226:159252,0]
ClI2[:,1] = corr_spec[158226:159252,1]

#Cl I line at 1379.528A
ClI3 = np.zeros((1026,2))
ClI3[:,0] = corr_spec[177795:178821,0]
ClI3[:,1] = corr_spec[177795:178821,1]

#Optically thick line at 1347.24A
ClI4 = np.zeros((1026,2))
ClI4[:,0] = corr_spec[155122:156148,0]
ClI4[:,1] = corr_spec[155122:156148,1]

#Line at 1188.774A blended on red side
ClI5 = np.zeros((1026,2))
ClI5[:,0] = corr_spec[36800:37826,0]
ClI5[:,1] = corr_spec[36800:37826,1]

#Coadd used in plot
ClI_total = ClI1[:,1]+ClI2[:,1]+ClI3[:,1]
#Coadd including optically thick line
ClI_total1 = ClI1[:,1]+ClI2[:,1]+ClI3[:,1]+ClI4[:,1]
#Coadd including red side blended line
ClI_total2 = ClI1[:,1]+ClI2[:,1]+ClI3[:,1]+ClI5[:,1]

#Converting wavelength to km/s using wavelengths from the 1351A line
ClI_center = 1351.657
for i in np.linspace(0,1025,1026, dtype = np.int64):
    del_wl = ClI2[i,0]-ClI_center
    wl_ratio = del_wl/ClI2[i,0]
    ClI2[i,0] = wl_ratio*c


#S I line at 1270.7821A
SI1 = np.zeros((1000,2))
SI1[:,0] = corr_spec[94410:95410,0]
SI1[:,1] = corr_spec[94410:95410,1]

#Line at 1269.2086A
SI2 = np.zeros((1000,2))
SI2[:,0] = corr_spec[93307:94307,0]
SI2[:,1] = corr_spec[93307:94307,1]

#Line at 1310.194A
SI3 = np.zeros((1000,2))
SI3[:,0] = corr_spec[127122:128122,0]
SI3[:,1] = corr_spec[127122:128122,1]

#Line at 1326.6432A
SI4 = np.zeros((1000,2))
SI4[:,0] = corr_spec[140677:141677,0]
SI4[:,1] = corr_spec[140677:141677,1]

#Line at 1381.5521A
SI5 = np.zeros((1000,2))
SI5[:,0] = corr_spec[179228:180228,0]
SI5[:,1] = corr_spec[179228:180228,1]

#Line at 1385.51A
SI6 = np.zeros((1000,2))
SI6[:,0] = corr_spec[182007:183007,0]
SI6[:,1] = corr_spec[182007:183007,1]

#Line at 1409.3369A
SI7 = np.zeros((1000,2))
SI7[:,0] = corr_spec[198735:199735,0]
SI7[:,1] = corr_spec[198735:199735,1]

#Line at 1436.9675A - Not included bc out of range of off-disk data
SI8 = np.zeros((1000,2))
SI8[:,0] = corr_spec[218135:219135,0]
SI8[:,1] = corr_spec[218135:219135,1]

#Line at 1392.5878A
SI9 = np.zeros((1000,2))
SI9[:,0] = corr_spec[186975:187975,0]
SI9[:,1] = corr_spec[186975:187975,1]

#Line at 1401.5136A
SI10 = np.zeros((1000,2))
SI10[:,0] = corr_spec[193245:194245,0]
SI10[:,1] = corr_spec[193245:194245,1]

#Odd shaped peak at 1448.229A
SI11 = np.zeros((1000,2))
SI11[:,0] = corr_spec[222591:223591,0]
SI11[:,1] = corr_spec[222591:223591,1]

#Line at 1425A - Some uncertainty on if blended or not
SI12 = np.zeros((1000,2))
SI12[:,0] = corr_spec[209754:210754,0]
SI12[:,1] = corr_spec[209754:210754,1]

#Coadd used in plot
SI_total1 = SI1[:,1]+SI2[:,1]+SI3[:,1]+SI4[:,1]+SI5[:,1]+SI6[:,1]+SI7[:,1]+\
            SI9[:,1]+SI10[:,1]
#Coadd that includes the 1425A line
SI_total2 = SI1[:,1]+SI2[:,1]+SI3[:,1]+SI4[:,1]+SI5[:,1]+SI6[:,1]+SI7[:,1]+\
            SI9[:,1]+SI10[:,1]+SI12[:,1]

#Converting wavelength to km/s using wavelengths from 1381A line
SI_center = 1381.5521
for i in np.linspace(0,999,1000, dtype = np.int64):
    del_wl = SI5[i,0]-SI_center
    wl_ratio = del_wl/SI5[i,0]
    SI5[i,0] = wl_ratio*c



##Coadding leading spec lines

leadingspec = spec4

#O I line at 1355.598A
OI_lead1 = np.zeros((258,2))
OI_lead1[:,0] = leadingspec[24033:24291,0]
OI_lead1[:,1] = leadingspec[24033:24291,1]

#O I line at 1358.512A
OI_lead2 = np.zeros((258,2))
OI_lead2[:,0] = leadingspec[24326:24584,0]
OI_lead2[:,1] = leadingspec[24326:24584,1]

OI_lead_total = np.zeros((258,2))
OI_lead_total[:,1] = OI_lead1[:,1]+OI_lead2[:,1]

#Converting wavelength to km/s using same center wl as before
for i in np.linspace(0,257,258, dtype = np.int64):
    del_wl = OI_lead1[i,0]-OI_center
    wl_ratio = del_wl/OI_lead1[i,0]
    OI_lead_total[i,0] = wl_ratio*c

#Scaling the coadded line - will be different once full spec coadd is fixed
OI_lead_total[:,1] = OI_lead_total[:,1]/2

#Cl I line at 1179.293A
ClI_lead1 = np.zeros((144,2))
ClI_lead1[:,0] = leadingspec[5395:5539,0]
ClI_lead1[:,1] = leadingspec[5395:5539,1]

#Cl I line at 1351.657A
ClI_lead2 = np.zeros((144,2))
ClI_lead2[:,0] = leadingspec[23695:23839,0]
ClI_lead2[:,1] = leadingspec[23695:23839,1]

#Cl I line at 1379.528A
ClI_lead3 = np.zeros((144,2))
ClI_lead3[:,0] = leadingspec[26490:26634,0]
ClI_lead3[:,1] = leadingspec[26490:26634,1]

ClI_lead_total = np.zeros((144,2))
ClI_lead_total[:,1] = ClI_lead1[:,1]+ClI_lead2[:,1]+ClI_lead3[:,1]

#Converting wl to km/s using same Cl center wavelength as before
for i in np.linspace(0,143,144, dtype = np.int64):
    del_wl = ClI_lead2[i,0]-ClI_center
    wl_ratio = del_wl/ClI_lead2[i,0]
    ClI_lead_total[i,0] = wl_ratio*c

#S I line at 1270.7821A
SI_lead1 = np.zeros((236,2))
SI_lead1[:,0] = leadingspec[14529:14765,0]
SI_lead1[:,1] = leadingspec[14529:14765,1]

#Line at 1269.2086A
SI_lead2 = np.zeros((236,2))
SI_lead2[:,0] = leadingspec[14371:14607,0]
SI_lead2[:,1] = leadingspec[14371:14607,1]

#Line at 1310.194A
SI_lead3 = np.zeros((236,2))
SI_lead3[:,0] = leadingspec[19491:19727,0]
SI_lead3[:,1] = leadingspec[19491:19727,1]

#Line at 1326.6432A
SI_lead4 = np.zeros((236,2))
SI_lead4[:,0] = leadingspec[21140:21376,0]
SI_lead4[:,1] = leadingspec[21140:21376,1]

#Line at 1381.5521A
SI_lead5 = np.zeros((236,2))
SI_lead5[:,0] = leadingspec[26647:26883,0]
SI_lead5[:,1] = leadingspec[26647:26883,1]

#Line at 1385.51A
SI_lead6 = np.zeros((236,2))
SI_lead6[:,0] = leadingspec[27045:27281,0]
SI_lead6[:,1] = leadingspec[27045:27281,1]

#Line at 1392.5878A
SI_lead7 = np.zeros((236,2))
SI_lead7[:,0] = leadingspec[27754:27990,0]
SI_lead7[:,1] = leadingspec[27754:27990,1]

#Line at 1401.5136A
SI_lead8 = np.zeros((236,2))
SI_lead8[:,0] = leadingspec[28649:28885,0]
SI_lead8[:,1] = leadingspec[28649:28885,1]

#Line at 1409.3369A
SI_lead9 = np.zeros((236,2))
SI_lead9[:,0] = leadingspec[29434:29670,0]
SI_lead9[:,1] = leadingspec[29434:29670,1]

#Line at 1425A - End of dataset so might not be reliable
SI_lead10 = np.zeros((236,2))
SI_lead10[:,0] = leadingspec[31008:31244,0]
SI_lead10[:,1] = leadingspec[31008:31244,1]

#Coadd used in plot w/ 1425A line commented out in csae it needs to be used
SI_lead_total = np.zeros((236,2))
SI_lead_total[:,1] = SI_lead1[:,1]+SI_lead2[:,1]+SI_lead3[:,1]+SI_lead4[:,1]+\
                   SI_lead5[:,1]+SI_lead6[:,1]+SI_lead7[:,1]+SI_lead8[:,1]+\
                   SI_lead9[:,1] #+SI_lead10[:,1]

#Converting the wl to km/s using same SI center wl as before
for i in np.linspace(0,235,236, dtype = np.int64):
    del_wl = SI_lead5[i,0]-SI_center
    wl_ratio = del_wl/SI_lead5[i,0]
    SI_lead_total[i,0] = wl_ratio*c

#Scaling the coadd - will change once full spec coadd is fixed
SI_lead_total[:,1] = SI_lead_total[:,1]/2.5


##Trailing spec lines

trailspec = spec5

#O I line at 1355.598A
OI_trail1 = np.zeros((258,2))
OI_trail1[:,0] = trailspec[24054:24312,0]
OI_trail1[:,1] = trailspec[24054:24312,1]

#O I line at 1358.512A
OI_trail2 = np.zeros((258,2))
OI_trail2[:,0] = trailspec[24347:24605,0]
OI_trail2[:,1] = trailspec[24347:24605,1]

OI_trail_total = np.zeros((258,2))
OI_trail_total[:,1] = OI_trail1[:,1]+OI_trail2[:,1]

#Converting wavelength to km/s using same center wl as before
for i in np.linspace(0,257,258, dtype = np.int64):
    del_wl = OI_trail1[i,0]-OI_center
    wl_ratio = del_wl/OI_trail1[i,0]
    OI_trail_total[i,0] = wl_ratio*c

#Scaling the coadd - will change once full spec coadd is fixed
OI_trail_total[:,1] = OI_trail_total[:,1]/2

#Cl I line at 1179.293A
ClI_trail1 = np.zeros((144,2))
ClI_trail1[:,0] = trailspec[5414:5558,0]
ClI_trail1[:,1] = trailspec[5414:5558,1]

#Cl I line at 1351.657A
ClI_trail2 = np.zeros((144,2))
ClI_trail2[:,0] = trailspec[23716:23860,0]
ClI_trail2[:,1] = trailspec[23716:23860,1]

#Cl I line at 1379.528A
ClI_trail3 = np.zeros((144,2))
ClI_trail3[:,0] = trailspec[26512:26656,0]
ClI_trail3[:,1] = trailspec[26512:26656,1]

ClI_trail_total = np.zeros((144,2))
ClI_trail_total[:,1] = ClI_trail1[:,1]+ClI_trail2[:,1]+ClI_trail3[:,1]

#Converting wl to km/s using same Cl center wavelength as before
for i in np.linspace(0,143,144, dtype = np.int64):
    del_wl = ClI_trail2[i,0]-ClI_center
    wl_ratio = del_wl/ClI_trail2[i,0]
    ClI_trail_total[i,0] = wl_ratio*c

#S I line at 1270.7821A
SI_trail1 = np.zeros((236,2))
SI_trail1[:,0] = trailspec[14549:14785,0]
SI_trail1[:,1] = trailspec[14549:14785,1]

#Line at 1269.2086A
SI_trail2 = np.zeros((236,2))
SI_trail2[:,0] = trailspec[14391:14627,0]
SI_trail2[:,1] = trailspec[14391:14627,1]

#Line at 1310.194A
SI_trail3 = np.zeros((236,2))
SI_trail3[:,0] = trailspec[19511:19747,0]
SI_trail3[:,1] = trailspec[19511:19747,1]

#Line at 1326.6432A
SI_trail4 = np.zeros((236,2))
SI_trail4[:,0] = trailspec[21161:21397,0]
SI_trail4[:,1] = trailspec[21161:21397,1]

#Line at 1381.5521A
SI_trail5 = np.zeros((236,2))
SI_trail5[:,0] = trailspec[26669:26905,0]
SI_trail5[:,1] = trailspec[26669:26905,1]

#Line at 1385.51A
SI_trail6 = np.zeros((236,2))
SI_trail6[:,0] = trailspec[27066:27302,0]
SI_trail6[:,1] = trailspec[27066:27302,1]

#Line at 1392.5878A
SI_trail7 = np.zeros((236,2))
SI_trail7[:,0] = trailspec[27776:28012,0]
SI_trail7[:,1] = trailspec[27776:28012,1]

#Line at 1401.5136A
SI_trail8 = np.zeros((236,2))
SI_trail8[:,0] = trailspec[28671:28907,0]
SI_trail8[:,1] = trailspec[28671:28907,1]

#Line at 1409.3369A
SI_trail9 = np.zeros((236,2))
SI_trail9[:,0] = trailspec[29456:29692,0]
SI_trail9[:,1] = trailspec[29456:29692,1]

#Line at 1425A - End of dataset so might not be reliable
SI_trail10 = np.zeros((236,2))
SI_trail10[:,0] = trailspec[31030:31266,0]
SI_trail10[:,1] = trailspec[31030:31266,1]

#Coadd used in plot w/ 1425A line commented out in csae it needs to be used
SI_trail_total = np.zeros((236,2))
SI_trail_total[:,1] = SI_trail1[:,1]+SI_trail2[:,1]+SI_trail3[:,1]+SI_trail4[:,1]+\
                   SI_trail5[:,1]+SI_trail6[:,1]+SI_trail7[:,1]+SI_trail8[:,1]\
                   +SI_trail9[:,1] #+SI_trail10[:,1]

#Converting the wl to km/s using same SI center wl as before
for i in np.linspace(0,235,236, dtype = np.int64):
    del_wl = SI_trail5[i,0]-SI_center
    wl_ratio = del_wl/SI_trail5[i,0]
    SI_trail_total[i,0] = wl_ratio*c

#Scaling the coadd - will change once full spec coadd is fixed
SI_trail_total[:,1] = SI_trail_total[:,1]/2.5

##Creating the overlay plots
plt.figure(figsize = (15,10))
plt.subplot(1, 2, 1)
plt.xlim(left = -285.238)
plt.xlim(right = 281.451)
plt.plot(OI1[:,0], OI_total, label = 'On-disk')
plt.plot(OI_lead_total[:,0], OI_lead_total[:,1], label = 'Leading Disk')
plt.title('O I With Leading Observation', fontsize = 24)
plt.xlabel('Velocity (km/s)', fontsize = 20)
plt.ylabel(r'Intensity (Rayleighs/$\AA$)', fontsize = 20)
plt.legend(fontsize = 16)
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(25))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)

plt.subplot(1, 2, 2)
plt.xlim(left = -285.481)
plt.xlim(right = 281.121)
plt.plot(OI1[:,0], OI_total, label = 'On-disk')
plt.plot(OI_trail_total[:,0], OI_trail_total[:,1], 'g', label = 'Trailing Disk')
plt.title('O I With Trailing Observation', fontsize = 24)
plt.xlabel('Velocity (km/s)', fontsize = 20)
#plt.ylabel(r'Intensity (Rayleighs/$\AA$)', fontsize = 20)
plt.legend(fontsize = 16)
plt.tight_layout()
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(25))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)

plt.figure(figsize = (15,10))
plt.subplot(1, 2, 1)
plt.xlim(left = -162.264)
plt.xlim(right = 161.418)
plt.plot(ClI2[:,0], ClI_total2, label = 'On-disk')
plt.plot(ClI_lead_total[:,0], ClI_lead_total[:,1], label = 'Leading Disk')
plt.title('Cl I With Leading Observation', fontsize = 24)
plt.xlabel('Velocity (km/s)', fontsize = 20)
plt.ylabel(r'Intensity (Rayleighs/$\AA$)', fontsize = 20)
plt.legend(fontsize = 16)
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(25))
grph.yaxis.set_major_locator(MultipleLocator(0.5))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)

plt.subplot(1, 2, 2)
plt.xlim(left = -162.264)
plt.xlim(right = 161.418)
plt.plot(ClI2[:,0], ClI_total2, label = 'On-disk')
plt.plot(ClI_trail_total[:,0], ClI_trail_total[:,1], 'g', label = 'Trailing Disk')
plt.title('Cl I With Trailing Observation', fontsize = 24)
plt.xlabel('Velocity (km/s)', fontsize = 20)
#plt.ylabel(r'Intensity (Rayleighs/$\AA$)', fontsize = 20)
plt.legend(fontsize = 16)
plt.tight_layout()
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(25))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)


plt.figure(figsize = (15,10))
plt.subplot(1, 2, 1)
plt.xlim(left = -256.184)
plt.xlim(right = 252.26)
plt.plot(SI5[:,0], SI_total1, label = 'On-disk')
plt.plot(SI_lead_total[:,0], SI_lead_total[:,1], label = 'Leading Disk')
plt.title('S I With Leading Observation', fontsize = 24)
plt.xlabel('Velocity (km/s)', fontsize = 20)
plt.ylabel(r'Intensity (Rayleighs/$\AA$)', fontsize = 20)
plt.legend(fontsize = 16)
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(25))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)

plt.subplot(1, 2, 2)
plt.xlim(left = -255.136)
plt.xlim(right = 253.226)
plt.plot(SI5[:,0], SI_total1, label = 'On-disk')
plt.plot(SI_trail_total[:,0], SI_trail_total[:,1], 'g', label = 'Trailing Disk')
plt.title('S I With Trailing Observation', fontsize = 24)
plt.xlabel('Velocity (km/s)', fontsize = 20)
#plt.ylabel(r'Intensity (Rayleighs/$\AA$)', fontsize = 20)
plt.legend(fontsize = 16)
plt.tight_layout()
grph = plt.gca()
grph.xaxis.set_minor_locator(MultipleLocator(25))
grph.tick_params(axis = 'both', which = 'major', length = 8, labelsize = 16)
grph.tick_params(axis = 'both', which = 'minor', length = 6)
