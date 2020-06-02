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
#import fitting_clean_sim as fitting
import fitting_clean_real as fitting
import plotlib
import opticstools as ot


"""
Extract the correlation from a short exposure P2VM-reduced file. Only keep the
upper 50% of the visibility amplitudes which also have a standard deviation
less than 0.5 over the spectral range. Only keep the closure phases which have
a standard deviation less than 1 rad over the spectral range.

Then, fit our correlation model to the data and compute the covariance matrix
by multiplying it with the uncertainties of the data (1% for visibility
ampitudes and 1 deg for closure phases).
"""
##idir = '/Users/jenskammerer/Downloads/data/1s_ut/'
#idir = '1s_ut/'
#fitsfiles = ['GRAVI.2019-03-29T02-01-37.193_singlecalp2vmred.fits']
#insname = 'GRAVITY_SC'
#
#wavel, vis, vis_err, vis_sta, f1f2 = inout.load_p2vmred(idir, fitsfiles, insname)
#vis2corr_fit, vis2cov_fit = fitting.fit_vis2corr(wavel, vis, vis_sta, f1f2, percentile=0., stdlim=np.inf)
#t3corr_fit, t3cov_fit = fitting.fit_t3corr(wavel, vis, vis_sta, f1f2, stdlim=np.inf)
#vis2cov_fit, t3cov_fit = fitting.get_cov(vis2corr_fit, t3corr_fit, vis2sigma=0.01, t3sigma=np.radians(1.))

idir = '/Users/jenskammerer/Downloads/data/0101.C-0907(B)/'
#idir = '0101.C-0907/'
fitsfiles = ['GRAVI.2018-04-18T08-08-19.739_singlescip2vmred.fits', 'GRAVI.2018-04-18T08-12-10.749_singlescip2vmred.fits', 'GRAVI.2018-04-18T08-20-04.769_singlescip2vmred.fits']
insname = 'GRAVITY_SC'

vis2corrs = []
t3corrs = []
for i in range(len(fitsfiles)):
    wavel, vis, vis_err, vis_sta, f1f2 = inout.load_p2vmred(idir, [fitsfiles[i]], insname)
    vis2corr_fit, vis2cov_fit = fitting.fit_vis2corr(wavel, vis, vis_sta, f1f2, percentile=0., stdlim=np.inf, visamp=True)
    t3corr_fit, t3cov_fit = fitting.fit_t3corr(wavel, vis, vis_sta, f1f2, stdlim=np.inf)
    vis2corrs += [vis2corr_fit]
    t3corrs += [t3corr_fit]


"""
Extract the uv-coordinates from the OIFITS files and crop the fourth triangle
from the closure phase data (otherwise the covariance matrix would not be
invertible). We are using five OIFITS files since the number of observations
(5) must be at least the number of free parameters (companion flux, companion
RA, companion DEC, host star diameter, companion diameter=0).
"""
##idir = '/Users/jenskammerer/Downloads/data/1s_ut/'
#idir = '1s_ut/'
##fitsfiles = ['GRAVI.2019-03-29T01-42-55.145_singlecalvis.fits', 'GRAVI.2019-03-29T01-46-28.155_singlecalvis.fits', 'GRAVI.2019-03-29T01-51-13.167_singlecalvis.fits', 'GRAVI.2019-03-29T01-57-13.182_singlecalvis.fits', 'GRAVI.2019-03-29T02-01-37.193_singlecalvis.fits']
#fitsfiles = ['GRAVI.2019-03-29T01-42-55.145_singlecalvis.fits', 'GRAVI.2019-03-29T01-51-13.167_singlecalvis.fits', 'GRAVI.2019-03-29T02-01-37.193_singlecalvis.fits']
#insname = 'GRAVITY_SC'
#
#wavel, dwavel, vis2, vis2_err, vis2_u, vis2_v, vis2_sta, t3, t3_err, t3_sta = inout.load_oifits(idir, fitsfiles, insname)
#uu, vv, cp_mat = fitting.make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta)
#t3, t3cov_fit, cp_mat = fitting.crop_t3(t3, t3cov_fit, cp_mat)
#
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
#plt.savefig('uv_coverage.pdf')
##plt.show(block=True)

idir = '/Users/jenskammerer/Downloads/data/0101.C-0907(B)/'
#idir = '0101.C-0907/'
fitsfiles = ['GRAVI.2018-04-18T08-08-19.739_singlescivis_singlesciviscalibrated.fits', 'GRAVI.2018-04-18T08-12-10.749_singlescivis_singlesciviscalibrated.fits', 'GRAVI.2018-04-18T08-20-04.769_singlescivis_singlesciviscalibrated.fits']
insname = 'GRAVITY_SC'

wavel, dwavel, vis2, vis2_err, vis2_u, vis2_v, vis2_sta, t3, t3_err, t3_sta = inout.load_oifits(idir, fitsfiles, insname, visamp=True)
uu, vv, cp_mat = fitting.make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta)

# Fix nans
if (np.sum(np.isnan(vis2)) != 0):
    nans, func = np.isnan(vis2), lambda x: x.nonzero()[0]
    vis2[nans] = np.interp(func(nans), func(~nans), vis2[~nans])
if (np.sum(np.isnan(t3)) != 0):
    nans, func = np.isnan(t3), lambda x: x.nonzero()[0]
    t3[nans] = np.interp(func(nans), func(~nans), t3[~nans])
if (np.sum(np.isnan(vis2_err)) != 0):
    nans, func = np.isnan(vis2_err), lambda x: x.nonzero()[0]
    vis2_err[nans] = np.interp(func(nans), func(~nans), vis2_err[~nans])
if (np.sum(np.isnan(t3_err)) != 0):
    nans, func = np.isnan(t3_err), lambda x: x.nonzero()[0]
    t3_err[nans] = np.interp(func(nans), func(~nans), t3_err[~nans])

# Compute covariance
vis2covs = []
t3covs = []
for i in range(len(fitsfiles)):
    vis2cov_fit, t3cov_fit = fitting.get_cov(vis2corrs[i], t3corrs[i], vis2sigma=vis2_err[i].flatten(), t3sigma=t3_err[i].flatten())
    vis2covs += [vis2cov_fit]
    t3covs += [t3cov_fit]
t3, t3covs, cp_mat = fitting.crop_t3(t3, t3covs, cp_mat)
vis2covs_diag = []
t3covs_diag = []
for i in range(len(fitsfiles)):
    vis2covs_diag += [np.diag(vis2covs[i])]
    t3covs_diag += [np.diag(t3covs[i])]

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
#plt.show(block=True)


"""
Compute a contrast map for a simulated data set affected by correlated noise,
but without a companion. Then, compute its azimuthal average in order to obtain
a first guess for the empirical 1-sigma detection limit. It seems like 1e-4 is
a good start.
"""
#odir = 'fit_to_noise_new/'
odir = 'test/'

p0 = np.array([0., 0., 0.])
#for i in range(100):
for i in range(1):
    
    Nobs = 3
    
#    vis2_full, t3_full = fitting.sim_ud_bin_single_noise(uu, vv, cp_mat, f=p0[0], alpha=[p0[1], p0[2]], vis2cov=[vis2cov_fit]*Nobs, t3cov=[t3cov_fit]*Nobs)
#    f_diag, alpha_diag, chi2_diag, fs_diag, dfs_diag, chi2s_diag, alpha_u_diag, alpha_v_diag = fitting.gridsearch(vis2_full, t3_full, uu, vv, cp_mat, [np.diag(vis2cov_fit)]*Nobs, [np.diag(t3cov_fit)]*Nobs)
#    f_full, alpha_full, chi2_full, fs_full, dfs_full, chi2s_full, alpha_u_full, alpha_v_full = fitting.gridsearch(vis2_full, t3_full, uu, vv, cp_mat, [vis2cov_fit]*Nobs, [t3cov_fit]*Nobs)
##    plotlib.snr2(f_diag, alpha_diag, chi2_diag, fs_diag, dfs_diag, alpha_u_diag, alpha_v_diag, f_full, alpha_full, chi2_full, fs_full, dfs_full, alpha_u_full, alpha_v_full)
    
    f_diag, alpha_diag, chi2_diag, fs_diag, dfs_diag, chi2s_diag, alpha_u_diag, alpha_v_diag = fitting.gridsearch(vis2, t3, uu, vv, cp_mat, vis2covs_diag, t3covs_diag)
    f_full, alpha_full, chi2_full, fs_full, dfs_full, chi2s_full, alpha_u_full, alpha_v_full = fitting.gridsearch(vis2, t3, uu, vv, cp_mat, vis2covs, t3covs)
#    plotlib.snr2_azavg(f_diag, alpha_diag, chi2_diag, fs_diag, dfs_diag, alpha_u_diag, alpha_v_diag, f_full, alpha_full, chi2_full, fs_full, dfs_full, alpha_u_full, alpha_v_full)
    
    path_diag = odir+'diag_%03.0f.fits' % i
    uv_hdu = pyfits.PrimaryHDU(np.array([uu, vv]))
    uv_hdu.header['EXTNAME'] = 'uv'
    vis2_hdu = pyfits.ImageHDU(vis2)
    vis2_hdu.header['EXTNAME'] = 'vis2'
    vis2cov_hdu = pyfits.ImageHDU(np.diag(vis2cov_fit))
    vis2cov_hdu.header['EXTNAME'] = 'vis2cov'
    t3_hdu = pyfits.ImageHDU(t3)
    t3_hdu.header['EXTNAME'] = 't3'
    t3cov_hdu = pyfits.ImageHDU(np.diag(t3cov_fit))
    t3cov_hdu.header['EXTNAME'] = 't3cov'
    cp_mat_hdu = pyfits.ImageHDU(cp_mat)
    cp_mat_hdu.header['EXTNAME'] = 'cp_mat'
    pp_hdu = pyfits.ImageHDU(np.array([f_diag, alpha_diag[0], alpha_diag[1], chi2_diag]))
    pp_hdu.header['EXTNAME'] = 'pp'
    fs_hdu = pyfits.ImageHDU(fs_diag)
    fs_hdu.header['EXTNAME'] = 'fs'
    dfs_hdu = pyfits.ImageHDU(dfs_diag)
    dfs_hdu.header['EXTNAME'] = 'dfs'
    hdul = pyfits.HDUList([uv_hdu, vis2_hdu, vis2cov_hdu, t3_hdu, t3cov_hdu, cp_mat_hdu, pp_hdu, fs_hdu, dfs_hdu])
    hdul.writeto(path_diag, overwrite=True, output_verify='fix')
    hdul.close()
    
    path_full = odir+'full_%03.0f.fits' % i
    uv_hdu = pyfits.PrimaryHDU(np.array([uu, vv]))
    uv_hdu.header['EXTNAME'] = 'uv'
    vis2_hdu = pyfits.ImageHDU(vis2)
    vis2_hdu.header['EXTNAME'] = 'vis2'
    vis2cov_hdu = pyfits.ImageHDU(vis2cov_fit)
    vis2cov_hdu.header['EXTNAME'] = 'vis2cov'
    t3_hdu = pyfits.ImageHDU(t3)
    t3_hdu.header['EXTNAME'] = 't3'
    t3cov_hdu = pyfits.ImageHDU(t3cov_fit)
    t3cov_hdu.header['EXTNAME'] = 't3cov'
    cp_mat_hdu = pyfits.ImageHDU(cp_mat)
    cp_mat_hdu.header['EXTNAME'] = 'cp_mat'
    pp_hdu = pyfits.ImageHDU(np.array([f_full, alpha_full[0], alpha_full[1], chi2_full]))
    pp_hdu.header['EXTNAME'] = 'pp'
    fs_hdu = pyfits.ImageHDU(fs_full)
    fs_hdu.header['EXTNAME'] = 'fs'
    dfs_hdu = pyfits.ImageHDU(dfs_full)
    dfs_hdu.header['EXTNAME'] = 'dfs'
    hdul = pyfits.HDUList([uv_hdu, vis2_hdu, vis2cov_hdu, t3_hdu, t3cov_hdu, cp_mat_hdu, pp_hdu, fs_hdu, dfs_hdu])
    hdul.writeto(path_full, overwrite=True, output_verify='fix')
    hdul.close()
    
#    import pdb; pdb.set_trace()
