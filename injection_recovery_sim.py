import matplotlib
matplotlib.use('agg')

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import time
np.random.seed(1994)

import sys
sys.path.append('../corr2')
sys.path.append('/Volumes/OZ 1/Python/Packages/opticstools/opticstools/opticstools')
sys.path.append('../opticstools/opticstools/opticstools')

import inout
import fitting_clean_sim as fitting
import plotlib
#import opticstools as ot

# Set interactive plots to off and change default plot font size.
plt.ioff()
plt.rcParams.update({'font.size': 12})


"""
Just to verify the fitting routine. Use the CANDID test data (AX Cir) with un-
correlated errors. For simplicity, we don't apply wavelength smearing here.
Note that North corresponds to positive DEC and East corresponds to negative
RA.
"""
#idir = '/Users/jenskammerer/Downloads/data/testdata/'
#fitsfiles = ['AXCir.oifits']
#insname = 'PIONIER_Pnat(1.6135391/1.7698610)'
#
#wavel, dwavel, vis2, vis2_err, vis2_u, vis2_v, vis2_sta, t3, t3_err, t3_sta = inout.load_oifits(idir, fitsfiles, insname, Nbase=6, Ntria=4)
#uu, vv, cp_mat = fitting.make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta)
#vis2cov = []
#t3cov = []
#for i in range(vis2_err.shape[0]):
#    vis2cov += [vis2_err[i].flatten()**2]
#    t3cov += [t3_err[i].flatten()**2]
#fitting.gridsearch_leastsq(vis2, t3, uu, vv, cp_mat, vis2cov, t3cov, f0=0.01)


"""
Extract the correlation from a short exposure P2VM-reduced file. Only keep the
upper 50% of the visibility amplitudes which also have a standard deviation
less than 0.5 over the spectral range. Only keep the closure phases which have
a standard deviation less than 1 rad over the spectral range.

Then, fit our correlation model to the data and compute the covariance matrix
by multiplying it with the uncertainties of the data (1% for visibility
ampitudes and 1 deg for closure phases).
"""
#idir = '/Users/jenskammerer/Downloads/data/1s_ut/'
idir = '1s_ut/'
fitsfiles = ['GRAVI.2019-03-29T02-01-37.193_singlecalp2vmred.fits']
insname = 'GRAVITY_SC'

wavel, vis, vis_err, vis_sta, f1f2 = inout.load_p2vmred(idir, fitsfiles, insname)
vis2corr_fit, vis2cov_fit = fitting.fit_vis2corr(wavel, vis, vis_sta, f1f2, percentile=0., stdlim=np.inf)
t3corr_fit, t3cov_fit = fitting.fit_t3corr(wavel, vis, vis_sta, f1f2, stdlim=np.inf)
vis2cov_fit, t3cov_fit = fitting.get_cov(vis2corr_fit, t3corr_fit, vis2sigma=0.01, t3sigma=np.radians(1.))

##idir = '/Users/jenskammerer/Downloads/data/0101.C-0907(B)/'
#idir = '0101.C-0907/'
#fitsfiles = ['GRAVI.2018-04-18T08:08:19.739_singlescip2vmred.fits', 'GRAVI.2018-04-18T08:12:10.749_singlescip2vmred.fits', 'GRAVI.2018-04-18T08:20:04.769_singlescip2vmred.fits']
#insname = 'GRAVITY_SC'
#
#vis2corrs = []
#t3corrs = []
#for i in range(len(fitsfiles)):
#    wavel, vis, vis_err, vis_sta, f1f2 = inout.load_p2vmred(idir, [fitsfiles[i]], insname)
#    vis2corr_fit, vis2cov_fit = fitting.fit_vis2corr(wavel, vis, vis_sta, f1f2, percentile=0., stdlim=np.inf, visamp=True)
#    t3corr_fit, t3cov_fit = fitting.fit_t3corr(wavel, vis, vis_sta, f1f2, stdlim=np.inf)
#    vis2corrs += [vis2corr_fit]
#    t3corrs += [t3corr_fit]


"""
Extract the uv-coordinates from the OIFITS files and crop the fourth triangle
from the closure phase data (otherwise the covariance matrix would not be
invertible). We are using five OIFITS files since the number of observations
(5) must be at least the number of free parameters (companion flux, companion
RA, companion DEC, host star diameter, companion diameter=0).
"""
#idir = '/Users/jenskammerer/Downloads/data/1s_ut/'
idir = '1s_ut/'
#fitsfiles = ['GRAVI.2019-03-29T01-42-55.145_singlecalvis.fits', 'GRAVI.2019-03-29T01-46-28.155_singlecalvis.fits', 'GRAVI.2019-03-29T01-51-13.167_singlecalvis.fits', 'GRAVI.2019-03-29T01-57-13.182_singlecalvis.fits', 'GRAVI.2019-03-29T02-01-37.193_singlecalvis.fits']
fitsfiles = ['GRAVI.2019-03-29T01-42-55.145_singlecalvis.fits', 'GRAVI.2019-03-29T01-51-13.167_singlecalvis.fits', 'GRAVI.2019-03-29T02-01-37.193_singlecalvis.fits']
insname = 'GRAVITY_SC'

wavel, dwavel, vis2, vis2_err, vis2_u, vis2_v, vis2_sta, t3, t3_err, t3_sta = inout.load_oifits(idir, fitsfiles, insname)
uu, vv, cp_mat = fitting.make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta)
t3, t3cov_fit, cp_mat = fitting.crop_t3(t3, t3cov_fit, cp_mat)

#labels = ['UT4 UT3', 'UT4 UT2', 'UT4 UT1', 'UT3 UT2', 'UT3 UT1', 'UT2 UT1']
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#plt.figure()
#ax = plt.gca()
#for i in range(vis2_u.shape[1]):
##    ax.scatter(vis2_u[:, i].flatten(), vis2_v[:, i].flatten(), c=colors[i], label=str(vis2_sta[0, i]))
#    ax.scatter(vis2_u[:, i].flatten(), vis2_v[:, i].flatten(), c=colors[i], label=labels[i])
#    ax.scatter(-vis2_u[:, i].flatten(), -vis2_v[:, i].flatten(), c=colors[i])
#ax.set_xlabel('Fourier u-baseline [m]')
#ax.set_ylabel('Fourier v-baseline [m]')
#ax.set_xlim([-150, 150])
#ax.set_ylim([-150, 150])
#ax.grid()
#ax.legend()
#plt.tight_layout()
#plt.savefig('figures/uv_coverage.pdf')
#plt.show(block=True)

##idir = '/Users/jenskammerer/Downloads/data/0101.C-0907(B)/'
#idir = '0101.C-0907/'
#fitsfiles = ['GRAVI.2018-04-18T08:08:19.739_singlescivis_singlesciviscalibrated.fits', 'GRAVI.2018-04-18T08:12:10.749_singlescivis_singlesciviscalibrated.fits', 'GRAVI.2018-04-18T08:20:04.769_singlescivis_singlesciviscalibrated.fits']
#insname = 'GRAVITY_SC'
#
#wavel, dwavel, vis2, vis2_err, vis2_u, vis2_v, vis2_sta, t3, t3_err, t3_sta = inout.load_oifits(idir, fitsfiles, insname, visamp=True)
#uu, vv, cp_mat = fitting.make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta)
#
## Fix nans
#if (np.sum(np.isnan(vis2)) != 0):
#    nans, func = np.isnan(vis2), lambda x: x.nonzero()[0]
#    vis2[nans] = np.interp(func(nans), func(~nans), vis2[~nans])
#if (np.sum(np.isnan(t3)) != 0):
#    nans, func = np.isnan(t3), lambda x: x.nonzero()[0]
#    t3[nans] = np.interp(func(nans), func(~nans), t3[~nans])
#if (np.sum(np.isnan(vis2_err)) != 0):
#    nans, func = np.isnan(vis2_err), lambda x: x.nonzero()[0]
#    vis2_err[nans] = np.interp(func(nans), func(~nans), vis2_err[~nans])
#if (np.sum(np.isnan(t3_err)) != 0):
#    nans, func = np.isnan(t3_err), lambda x: x.nonzero()[0]
#    t3_err[nans] = np.interp(func(nans), func(~nans), t3_err[~nans])
#
## Compute covariance
#vis2covs = []
#t3covs = []
#for i in range(len(fitsfiles)):
#    vis2cov_fit, t3cov_fit = fitting.get_cov(vis2corrs[i], t3corrs[i], vis2sigma=vis2_err[i].flatten(), t3sigma=t3_err[i].flatten())
#    vis2covs += [vis2cov_fit]
#    t3covs += [t3cov_fit]
##t3, t3covs, cp_mat = fitting.crop_t3(t3, t3covs, cp_mat)
#vis2covs_diag = []
#t3covs_diag = []
#for i in range(len(fitsfiles)):
#    vis2covs_diag += [np.diag(vis2covs[i])]
#    t3covs_diag += [np.diag(t3covs[i])]

#Nobs = vis2.shape[0]
#f, ax = plt.subplots(1, Nobs, figsize=(6.4*Nobs, 4.8*1), sharey=True)
#for i in range(Nobs):
#    ax[i].plot(wavel[0]*1e6, vis2[i].T)
#    ax[i].grid(axis='y')
#    ax[i].set_xlabel('$\lambda$ [microns]')
#    if (i == 0):
#        ax[i].set_ylabel('$|V|$')
#plt.tight_layout()
#plt.savefig('visamp.pdf')
#plt.show()
#f, ax = plt.subplots(1, Nobs, figsize=(6.4*Nobs, 4.8*1), sharey=True)
#for i in range(Nobs):
#    ax[i].plot(wavel[0]*1e6, t3[i].T)
#    ax[i].grid(axis='y')
#    ax[i].set_xlabel('$\lambda$ [microns]')
#    if (i == 0):
#        ax[i].set_ylabel('$\\theta$ [rad]')
#plt.tight_layout()
#plt.savefig('t3.pdf')
#plt.show()
#import pdb; pdb.set_trace()


"""
Compute a contrast map for a simulated data set affected by correlated noise,
but without a companion. Then, compute its azimuthal average in order to obtain
a first guess for the empirical 1-sigma detection limit. It seems like 1e-4 is
a good start.
"""
#p0 = np.array([0., 0., 0.])
#vis2_full, t3_full = fitting.sim_bin_single(uu, vv, cp_mat, f=p0[0], alpha=[p0[1], p0[2]], vis2cov=vis2cov_fit, t3cov=t3cov_fit)
#f, alpha, chi2, fs, dfs, chi2s, alpha_u, alpha_v = fitting.gridsearch(vis2_full, t3_full, uu, vv, cp_mat, vis2cov_fit, t3cov_fit)
#plotlib.snr2(f ,alpha, chi2, fs, dfs, alpha_u, alpha_v, f ,alpha, chi2, fs, dfs, alpha_u, alpha_v)
#x, y = ot.azimuthalAverage(np.abs(fs), returnradii=True, binsize=1.)

#p0 = np.array([0., 0., 0.])
#f, alpha, chi2, fs, dfs, chi2s, alpha_u, alpha_v = fitting.gridsearch(vis2, t3, uu, vv, cp_mat, vis2covs, t3covs)
#plotlib.snr2(f ,alpha, chi2, fs, dfs, alpha_u, alpha_v, f ,alpha, chi2, fs, dfs, alpha_u, alpha_v)
#x, y = ot.azimuthalAverage(np.abs(fs), returnradii=True, binsize=1.)
#
#plt.figure()
#plt.plot(x, y)
#plt.yscale('log')
#plt.xlabel('Angular separation [mas]')
#plt.ylabel('Contrast')
#plt.tight_layout()
#plt.show(block=True)
#
#import pdb; pdb.set_trace()


"""
Go through a cube of companion fluxes, companion RAs and companion DECs. For
each cube cell, simulate a uniform disk (1 mas) with an unresolved companion,
affected by correlated noise. Then, try to recover the companion with diagonal
and full covariance. Save all relevant data from the fits with diagonal and
full covariance as FITS files.
"""
#p0 = np.array([0.0383, 0.36, -5.44, 0., 0.])
#vis2_bin, t3_bin = fitting.sim_ud_bin_single(uu, vv, cp_mat, f=p0[0], alpha=[p0[1], p0[2]], theta1=p0[3], theta2=p0[4])
#vis2 = np.true_divide(vis2, vis2_bin)
#t3 = t3-t3_bin

#odir = '/Users/jenskammerer/Downloads/data/'
odir = '/priv/mulga2/kjens/fitcube_sim_new/'
cons = np.logspace(-4, -1.5, 11)
#cons = np.logspace(-3, -0.5, 11)
Ncons = len(cons)
seps = np.linspace(-30, 30, 13)
#seps = np.linspace(-15, 15, 7)
Nseps = len(seps)
for i in range(Ncons):
    if (i == 10):
        for j in range(Nseps):
            for k in range(Nseps):
                t0 = time.time()
                
                Nobs = 3
                p0 = np.array([cons[i], seps[j], seps[k], 1., 0.])
                vis2_full, t3_full = fitting.sim_ud_bin_single_noise(uu, vv, cp_mat, f=p0[0], alpha=[p0[1], p0[2]], theta1=p0[3], theta2=p0[4], vis2cov=[vis2cov_fit]*Nobs, t3cov=[t3cov_fit]*Nobs)
                pps_diag, chi2s_diag, chi2ud_diag, sig_diag = fitting.gridsearch_leastsq(vis2_full, t3_full, uu, vv, cp_mat, [np.diag(vis2cov_fit)]*Nobs, [np.diag(t3cov_fit)]*Nobs, f0=p0[0])
                pps_full, chi2s_full, chi2ud_full, sig_full = fitting.gridsearch_leastsq(vis2_full, t3_full, uu, vv, cp_mat, [vis2cov_fit]*Nobs, [t3cov_fit]*Nobs, f0=p0[0])
                
#                p0 = np.array([cons[i], seps[j], seps[k], 0., 0.])
##                vis2_bin, t3_bin = fitting.sim_ud_bin_single(uu, vv, cp_mat, f=p0[0], alpha=[p0[1], p0[2]], theta1=p0[3], theta2=p0[4])
##                vis2_full = np.multiply(vis2.copy(), vis2_bin)
#                vis2_full = vis2.copy()
##                t3_full = t3.copy()+t3_bin
#                t3_full = t3.copy()
#                pps_diag, chi2s_diag, chi2ud_diag, sig_diag = fitting.gridsearch_leastsq(vis2_full, t3_full, uu, vv, cp_mat, vis2covs_diag, t3covs_diag, f0=p0[0])
#                pps_full, chi2s_full, chi2ud_full, sig_full = fitting.gridsearch_leastsq(vis2_full, t3_full, uu, vv, cp_mat, vis2covs, t3covs, f0=p0[0])
                
                path_diag = odir+'diag_%.4f_%+.0f_%+.0f.fits' % (p0[0], p0[1], p0[2])
                uv_hdu = pyfits.PrimaryHDU(np.array([uu, vv]))
                uv_hdu.header['EXTNAME'] = 'uv'
                vis2_hdu = pyfits.ImageHDU(vis2_full)
                vis2_hdu.header['EXTNAME'] = 'vis2'
                vis2cov_hdu = pyfits.ImageHDU(np.diag(vis2cov_fit))
                vis2cov_hdu.header['EXTNAME'] = 'vis2cov'
                t3_hdu = pyfits.ImageHDU(t3_full)
                t3_hdu.header['EXTNAME'] = 't3'
                t3cov_hdu = pyfits.ImageHDU(np.diag(t3cov_fit))
                t3cov_hdu.header['EXTNAME'] = 't3cov'
                cp_mat_hdu = pyfits.ImageHDU(cp_mat)
                cp_mat_hdu.header['EXTNAME'] = 'cp_mat'
                p0_hdu = pyfits.ImageHDU(p0)
                p0_hdu.header['EXTNAME'] = 'p0'
                pps_hdu = pyfits.ImageHDU(np.append(pps_diag, chi2s_diag[:, np.newaxis], axis=1))
                pps_hdu.header['EXTNAME'] = 'pps'
                sig_hdu = pyfits.ImageHDU(np.array([chi2ud_diag, sig_diag]))
                sig_hdu.header['EXTNAME'] = 'sig'
                hdul = pyfits.HDUList([uv_hdu, vis2_hdu, vis2cov_hdu, t3_hdu, t3cov_hdu, cp_mat_hdu, p0_hdu, pps_hdu, sig_hdu])
                hdul.writeto(path_diag, overwrite=True, output_verify='fix')
                hdul.close()
                
                path_full = odir+'full_%.4f_%+.0f_%+.0f.fits' % (p0[0], p0[1], p0[2])
                uv_hdu = pyfits.PrimaryHDU(np.array([uu, vv]))
                uv_hdu.header['EXTNAME'] = 'uv'
                vis2_hdu = pyfits.ImageHDU(vis2_full)
                vis2_hdu.header['EXTNAME'] = 'vis2'
                vis2cov_hdu = pyfits.ImageHDU(vis2cov_fit)
                vis2cov_hdu.header['EXTNAME'] = 'vis2cov'
                t3_hdu = pyfits.ImageHDU(t3_full)
                t3_hdu.header['EXTNAME'] = 't3'
                t3cov_hdu = pyfits.ImageHDU(t3cov_fit)
                t3cov_hdu.header['EXTNAME'] = 't3cov'
                cp_mat_hdu = pyfits.ImageHDU(cp_mat)
                cp_mat_hdu.header['EXTNAME'] = 'cp_mat'
                p0_hdu = pyfits.ImageHDU(p0)
                p0_hdu.header['EXTNAME'] = 'p0'
                pps_hdu = pyfits.ImageHDU(np.append(pps_full, chi2s_full[:, np.newaxis], axis=1))
                pps_hdu.header['EXTNAME'] = 'pps'
                sig_hdu = pyfits.ImageHDU(np.array([chi2ud_full, sig_full]))
                sig_hdu.header['EXTNAME'] = 'sig'
                hdul = pyfits.HDUList([uv_hdu, vis2_hdu, vis2cov_hdu, t3_hdu, t3cov_hdu, cp_mat_hdu, p0_hdu, pps_hdu, sig_hdu])
                hdul.writeto(path_full, overwrite=True, output_verify='fix')
                hdul.close()
                
                t1 = time.time()
                print('Time: %.0f s' % (t1-t0))
                
#                import pdb; pdb.set_trace()
