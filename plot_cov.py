import astropy.io.fits as pyfits
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import time
np.random.seed(1994)

import sys
sys.path.append('../corr2')
sys.path.append('/Volumes/OZ 1/Python/Packages/opticstools/opticstools/opticstools')
sys.path.append('../opticstools/opticstools/opticstools')

import inout
import fitting_clean as fitting
import plotlib
import opticstools as ot

# Set interactive plots to off and change default plot font size.
plt.ioff()
plt.rcParams.update({'font.size': 12})


"""
Extract the correlation from a short exposure P2VM-reduced file. Only keep the
upper 50% of the visibility amplitudes which also have a standard deviation
less than 0.5 over the spectral range. Only keep the closure phases which have
a standard deviation less than 1 rad over the spectral range.
"""
idir = '/Users/jenskammerer/Downloads/data/1s_ut/'
fitsfiles = ['GRAVI.2019-03-29T02-01-37.193_singlecalp2vmred.fits']

#fitsfiles = ['GRAVI.2019-03-29T01-42-55.145_singlecalp2vmred.fits'] # ~bad
#fitsfiles = ['GRAVI.2019-03-29T01-45-04.151_singlecalp2vmred.fits'] # bad
#fitsfiles = ['GRAVI.2019-03-29T01-46-28.155_singlecalp2vmred.fits']
#fitsfiles = ['GRAVI.2019-03-29T01-48-43.160_singlecalp2vmred.fits'] # ~bad
#fitsfiles = ['GRAVI.2019-03-29T01-51-13.167_singlecalp2vmred.fits'] # bad
#fitsfiles = ['GRAVI.2019-03-29T01-53-52.173_singlecalp2vmred.fits'] # bad
#fitsfiles = ['GRAVI.2019-03-29T01-57-13.182_singlecalp2vmred.fits']
#fitsfiles = ['GRAVI.2019-03-29T01-59-31.188_singlecalp2vmred.fits']
#fitsfiles = ['GRAVI.2019-03-29T02-03-46.198_singlecalp2vmred.fits'] # bad
#fitsfiles = ['GRAVI.2019-03-29T02-32-37.272_singlecalp2vmred.fits'] # bad
#fitsfiles = ['GRAVI.2019-03-29T02-34-52.277_singlecalp2vmred.fits'] # bad

#idir = '/Users/jenskammerer/Downloads/data/1s_ut/'
#fitsfiles = ['GRAVI.2019-03-29T01-42-55.145_singlecalp2vmred.fits']
#fitsfiles = ['GRAVI.2019-03-29T01-46-28.155_singlecalp2vmred.fits']
#fitsfiles = ['GRAVI.2019-03-29T01-51-13.167_singlecalp2vmred.fits']
#fitsfiles = ['GRAVI.2019-03-29T01-57-13.182_singlecalp2vmred.fits']

#idir = '/Users/jenskammerer/Downloads/data/0101.C-0907(B)/'
#fitsfiles = ['GRAVI.2018-04-18T08-08-19.739_singlescip2vmred.fits']
#fitsfiles = ['GRAVI.2018-04-18T08-12-10.749_singlescip2vmred.fits']
#fitsfiles = ['GRAVI.2018-04-18T08-20-04.769_singlescip2vmred.fits']

#insname = 'GRAVITY_FT'
insname = 'GRAVITY_SC'
Nbase = 6
Ntria = 4
#Nwave = 5
Nwave = 210

wavel, vis, vis_err, vis_sta, f1f2 = inout.load_p2vmred(idir, fitsfiles, insname)
#vis2corr_fit, vis2cov_fit, vis2corr, vis2cov = fitting.fit_vis2corr(wavel, vis, vis_sta, f1f2, percentile=0., stdlim=np.inf, full_output=True) # FT
#t3corr_fit, t3cov_fit, t3corr, t3cov = fitting.fit_t3corr(wavel, vis, vis_sta, f1f2, stdlim=np.inf, full_output=True) # FT
vis2corr_fit, vis2cov_fit, vis2corr, vis2cov = fitting.fit_vis2corr(wavel, vis, vis_sta, f1f2, percentile=0., stdlim=np.inf, full_output=True, visamp=False) # SC
t3corr_fit, t3cov_fit, t3corr, t3cov = fitting.fit_t3corr(wavel, vis, vis_sta, f1f2, stdlim=np.inf, full_output=True) # SC

vis2corr_binned = np.zeros_like(vis2corr)
dd = np.diag(vis2corr).copy()
np.fill_diagonal(vis2corr, np.nan)
for i in range(Nbase):
    for j in range(Nbase):
        vis2corr_binned[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = np.nanmean(vis2corr[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave])
np.fill_diagonal(vis2corr, dd)
np.fill_diagonal(vis2corr_binned, dd)
vis2corr_fit_binned = np.zeros_like(vis2corr_fit)
dd = np.diag(vis2corr_fit).copy()
np.fill_diagonal(vis2corr_fit, np.nan)
for i in range(Nbase):
    for j in range(Nbase):
        vis2corr_fit_binned[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = np.nanmean(vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave])
np.fill_diagonal(vis2corr_fit, dd)
np.fill_diagonal(vis2corr_fit_binned, dd)

t3corr_binned = np.zeros_like(t3corr)
dd = np.diag(t3corr).copy()
np.fill_diagonal(t3corr, np.nan)
for i in range(Ntria):
    for j in range(Ntria):
        temp = np.diag(t3corr[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave]).copy()
        np.fill_diagonal(t3corr[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], np.nan)
        t3corr_binned[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = np.nanmean(t3corr[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave])
        np.fill_diagonal(t3corr[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], temp)
np.fill_diagonal(t3corr, dd)
np.fill_diagonal(t3corr_binned, dd)
t3corr_fit_binned = np.zeros_like(t3corr_fit)
dd = np.diag(t3corr_fit).copy()
np.fill_diagonal(t3corr_fit, np.nan)
for i in range(Ntria):
    for j in range(Ntria):
        temp = np.diag(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave]).copy()
        np.fill_diagonal(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], np.nan)
        t3corr_fit_binned[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = np.nanmean(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave])
        np.fill_diagonal(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], temp)
np.fill_diagonal(t3corr_fit, dd)
np.fill_diagonal(t3corr_fit_binned, dd)

#f, ax = plt.subplots(2, 2, figsize=(6.4*2, 4.8*1.25), gridspec_kw = {'height_ratios': [4, 1]})
#p00 = ax[0, 0].imshow(vis2corr, cmap='seismic_r', vmin=-1, vmax=1, extent=[0, vis2corr.shape[1], vis2corr.shape[0], 0], zorder=0)
#c00 = plt.colorbar(p00, ax=ax[0, 0])
#c00.set_label('Correlation', rotation=270, labelpad=20)
#margin = vis2corr.shape[0]/150.
#for i in range(Nbase):
#    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
#    ax[0, 0].add_patch(rect)
#    if (i != 0):
#        ax[0, 0].axhline(i*Nwave, color='gray', zorder=1)
#        ax[0, 0].axvline(i*Nwave, color='gray', zorder=1)
#    for j in range(Nbase):
#        if (i != j):
#            if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
#                rect = patches.Rectangle((j*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
#                ax[0, 0].add_patch(rect)
#ax[0, 0].set_xlabel('Index $j$')
#ax[0, 0].set_ylabel('Index $i$')
#temp = np.linspace(0, vis2corr.shape[1], Nbase+1)
#ax[0, 0].set_xlim([temp[0], temp[-1]])
#ax[0, 0].set_xticks(temp)
#temp = np.linspace(0, vis2corr.shape[0], Nbase+1)
#ax[0, 0].set_ylim([temp[-1], temp[0]])
#ax[0, 0].set_yticks(temp)
#temp = np.linspace(0.5, vis2cov.shape[0]-0.5, vis2cov.shape[0])
#ax[1, 0].plot(temp, np.diag(vis2cov), zorder=0)
#labels = ['UT4 UT3', 'UT4 UT2', 'UT4 UT1', 'UT3 UT2', 'UT3 UT1', 'UT2 UT1']
#for i in range(Nbase):
#    if (i != 0):
#        ax[1, 0].axvline(i*Nwave, lw=3, color='red', zorder=2)
##    ax[1, 0].text((1./Nbase)*(i+0.5), 0.8, str(vis_sta[0, i]), va='center', ha='center', transform=ax[1, 0].transAxes)
#    ax[1, 0].text((1./Nbase)*(i+0.5), 0.8, labels[i], va='center', ha='center', transform=ax[1, 0].transAxes)
#temp = np.linspace(0, vis2cov.shape[0], Nbase+1)
#ax[1, 0].set_xlabel('Index $j$')
#ax[1, 0].set_ylabel('Variance')
#ax[1, 0].set_xlim([temp[0], temp[-1]])
#ax[1, 0].set_xticks(temp)
##ax[1, 0].set_ylim([0, 4])
##ax[1, 0].set_ylim([0, 0.4])
#ax[1, 0].set_ylim([0, 0.04])
#ax[1, 0].grid(axis='y')
#p01 = ax[0, 1].imshow(t3corr, cmap='seismic_r', vmin=-1, vmax=1, extent=[0, t3corr.shape[1], t3corr.shape[0], 0], zorder=0)
#c01 = plt.colorbar(p01, ax=ax[0, 1])
#c01.set_label('Correlation', rotation=270, labelpad=20)
#margin = t3corr.shape[0]/150.
#for i in range(Ntria):
#    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
#    ax[0, 1].add_patch(rect)
#    if (i != 0):
#        ax[0, 1].axhline(i*Nwave, color='gray', zorder=1)
#        ax[0, 1].axvline(i*Nwave, color='gray', zorder=1)
#ax[0, 1].set_xlabel('Index $j$')
#ax[0, 1].set_ylabel('Index $i$')
##temp = np.linspace(0, t3corr.shape[1], Ntria+1)
##ax[0, 1].set_xlim([temp[0], temp[-1]])
##ax[0, 1].set_xticks(temp)
##temp = np.linspace(0, t3corr.shape[0], Ntria+1)
##ax[0, 1].set_ylim([temp[-1], temp[0]])
##ax[0, 1].set_yticks(temp)
#ax[0, 1].set_xlim([160, 200])
#ax[0, 1].set_xticks([160, 180, 200])
#ax[0, 1].set_ylim([620, 580])
#ax[0, 1].set_yticks([580, 600, 620])
#temp = np.linspace(0.5, t3cov.shape[0]-0.5, t3cov.shape[0])
#ax[1, 1].plot(temp, np.diag(t3cov), zorder=0)
#trias = [[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]]
#labels = ['UT2 UT3 UT4', 'UT1 UT3 UT4', 'UT1 UT2 UT4', 'UT1 UT2 UT3']
#for i in range(Ntria):
#    if (i != 0):
#        ax[1, 1].axvline(i*Nwave, lw=3, color='red', zorder=2)
##    ax[1, 1].text((1./Ntria)*(i+0.5), 0.8, str(np.unique(vis_sta[0, trias[i]])), va='center', ha='center', transform=ax[1, 1].transAxes)
#    ax[1, 1].text((1./Ntria)*(i+0.5), 0.8, labels[i], va='center', ha='center', transform=ax[1, 1].transAxes)
#temp = np.linspace(0, t3cov.shape[0], Ntria+1)
#ax[1, 1].set_xlabel('Index $j$')
#ax[1, 1].set_ylabel('Variance')
#ax[1, 1].set_xlim([temp[0], temp[-1]])
#ax[1, 1].set_xticks(temp)
##ax[1, 1].set_ylim([0, 4])
##ax[1, 1].set_ylim([0, 0.4])
#ax[1, 1].set_ylim([0, 0.04])
#ax[1, 1].grid(axis='y')
#plt.tight_layout()
##plt.savefig('figures/corr_FT.pdf')
##plt.savefig('figures/y3_corr_SC.pdf')
#plt.savefig('figures/y3_corr_SC_zoom.pdf')
#plt.show()

import pdb; pdb.set_trace()

f, ax = plt.subplots(2, 2, figsize=(6.4*2, 4.8*2))
p00 = ax[0, 0].imshow(vis2corr, cmap='seismic_r', vmin=-1, vmax=1, extent=[0, vis2corr.shape[1], vis2corr.shape[0], 0], zorder=0)
c00 = plt.colorbar(p00, ax=ax[0, 0])
c00.set_label('Correlation', rotation=270, labelpad=20)
margin = vis2corr.shape[0]/150.
for i in range(Nbase):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[0, 0].add_patch(rect)
    if (i != 0):
        ax[0, 0].axhline(i*Nwave, color='gray', zorder=1)
        ax[0, 0].axvline(i*Nwave, color='gray', zorder=1)
    for j in range(Nbase):
        text = ax[0, 0].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % vis2corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center', size=10)
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        if (i != j):
            if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                rect = patches.Rectangle((j*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                ax[0, 0].add_patch(rect)
ax[0, 0].set_xlabel('Index $j$')
ax[0, 0].set_ylabel('Index $i$')
temp = np.linspace(0, vis2corr.shape[1], Nbase+1)
ax[0, 0].set_xlim([temp[0], temp[-1]])
ax[0, 0].set_xticks(temp)
temp = np.linspace(0, vis2corr.shape[0], Nbase+1)
ax[0, 0].set_ylim([temp[-1], temp[0]])
ax[0, 0].set_yticks(temp)
p10 = ax[1, 0].imshow(vis2cov, vmin=-0.01, vmax=0.01, extent=[0, vis2cov.shape[1], vis2cov.shape[0], 0], zorder=0)
c10 = plt.colorbar(p10, ax=ax[1, 0])
c10.set_label('Covariance', rotation=270, labelpad=20)
margin = vis2cov.shape[0]/150.
for i in range(Nbase):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[1, 0].add_patch(rect)
    if (i != 0):
        ax[1, 0].axhline(i*Nwave, color='gray', zorder=1)
        ax[1, 0].axvline(i*Nwave, color='gray', zorder=1)
    for j in range(Nbase):
        if (i != j):
            if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                rect = patches.Rectangle((j*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                ax[1, 0].add_patch(rect)
ax[1, 0].set_xlabel('Index $j$')
ax[1, 0].set_ylabel('Index $i$')
temp = np.linspace(0, vis2cov.shape[1], Nbase+1)
ax[1, 0].set_xlim([temp[0], temp[-1]])
ax[1, 0].set_xticks(temp)
temp = np.linspace(0, vis2cov.shape[0], Nbase+1)
ax[1, 0].set_ylim([temp[-1], temp[0]])
ax[1, 0].set_yticks(temp)
p01 = ax[0, 1].imshow(vis2corr_fit, cmap='seismic_r', vmin=-1, vmax=1, extent=[0, vis2corr_fit.shape[1], vis2corr_fit.shape[0], 0], zorder=0)
c01 = plt.colorbar(p01, ax=ax[0, 1])
c01.set_label('Correlation', rotation=270, labelpad=20)
margin = vis2corr_fit.shape[0]/150.
for i in range(Nbase):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[0, 1].add_patch(rect)
    if (i != 0):
        ax[0, 1].axhline(i*Nwave, color='gray', zorder=1)
        ax[0, 1].axvline(i*Nwave, color='gray', zorder=1)
    for j in range(Nbase):
        text = ax[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % vis2corr_fit_binned[i*Nwave+1, j*Nwave], va='center', ha='center', size=10)
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        if (i != j):
            if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                rect = patches.Rectangle((j*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                ax[0, 1].add_patch(rect)
ax[0, 1].set_xlabel('Index $j$')
ax[0, 1].set_ylabel('Index $i$')
temp = np.linspace(0, vis2corr_fit.shape[1], Nbase+1)
ax[0, 1].set_xlim([temp[0], temp[-1]])
ax[0, 1].set_xticks(temp)
temp = np.linspace(0, vis2corr_fit.shape[0], Nbase+1)
ax[0, 1].set_ylim([temp[-1], temp[0]])
ax[0, 1].set_yticks(temp)
p11 = ax[1, 1].imshow(vis2cov_fit, vmin=-0.01, vmax=0.01, extent=[0, vis2cov_fit.shape[1], vis2cov_fit.shape[0], 0], zorder=0)
c11 = plt.colorbar(p11, ax=ax[1, 1])
c11.set_label('Covariance', rotation=270, labelpad=20)
margin = vis2cov_fit.shape[0]/150.
for i in range(Nbase):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[1, 1].add_patch(rect)
    if (i != 0):
        ax[1, 1].axhline(i*Nwave, color='gray', zorder=1)
        ax[1, 1].axvline(i*Nwave, color='gray', zorder=1)
    for j in range(Nbase):
        if (i != j):
            if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                rect = patches.Rectangle((j*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                ax[1, 1].add_patch(rect)
ax[1, 1].set_xlabel('Index $j$')
ax[1, 1].set_ylabel('Index $i$')
temp = np.linspace(0, vis2cov_fit.shape[1], Nbase+1)
ax[1, 1].set_xlim([temp[0], temp[-1]])
ax[1, 1].set_xticks(temp)
temp = np.linspace(0, vis2cov_fit.shape[0], Nbase+1)
ax[1, 1].set_ylim([temp[-1], temp[0]])
ax[1, 1].set_yticks(temp)
plt.tight_layout()
plt.savefig('figures/vis2fit_SC.pdf')
plt.show()

f, ax = plt.subplots(2, 2, figsize=(6.4*2, 4.8*2))
p00 = ax[0, 0].imshow(t3corr, cmap='seismic_r', vmin=-1, vmax=1, extent=[0, t3corr.shape[1], t3corr.shape[0], 0], zorder=0)
c00 = plt.colorbar(p00, ax=ax[0, 0])
c00.set_label('Correlation', rotation=270, labelpad=20)
margin = t3corr.shape[0]/150.
for i in range(Ntria):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[0, 0].add_patch(rect)
    if (i != 0):
        ax[0, 0].axhline(i*Nwave, color='gray', zorder=1)
        ax[0, 0].axvline(i*Nwave, color='gray', zorder=1)
    for j in range(Ntria):
        text = ax[0, 0].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % t3corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center', size=10)
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
ax[0, 0].set_xlabel('Index $j$')
ax[0, 0].set_ylabel('Index $i$')
temp = np.linspace(0, t3corr.shape[1], Nbase+1)
ax[0, 0].set_xlim([temp[0], temp[-1]])
ax[0, 0].set_xticks(temp)
temp = np.linspace(0, t3corr.shape[0], Nbase+1)
ax[0, 0].set_ylim([temp[-1], temp[0]])
ax[0, 0].set_yticks(temp)
p10 = ax[1, 0].imshow(t3cov, vmin=-0.01, vmax=0.01, extent=[0, t3cov.shape[1], t3cov.shape[0], 0], zorder=0)
c10 = plt.colorbar(p10, ax=ax[1, 0])
c10.set_label('Covariance', rotation=270, labelpad=20)
margin = t3cov.shape[0]/150.
for i in range(Ntria):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[1, 0].add_patch(rect)
    if (i != 0):
        ax[1, 0].axhline(i*Nwave, color='gray', zorder=1)
        ax[1, 0].axvline(i*Nwave, color='gray', zorder=1)
ax[1, 0].set_xlabel('Index $j$')
ax[1, 0].set_ylabel('Index $i$')
temp = np.linspace(0, t3cov.shape[1], Nbase+1)
ax[1, 0].set_xlim([temp[0], temp[-1]])
ax[1, 0].set_xticks(temp)
temp = np.linspace(0, t3cov.shape[0], Nbase+1)
ax[1, 0].set_ylim([temp[-1], temp[0]])
ax[1, 0].set_yticks(temp)
p01 = ax[0, 1].imshow(t3corr_fit, cmap='seismic_r', vmin=-1, vmax=1, extent=[0, t3corr_fit.shape[1], t3corr_fit.shape[0], 0], zorder=0)
c01 = plt.colorbar(p01, ax=ax[0, 1])
c01.set_label('Correlation', rotation=270, labelpad=20)
margin = t3corr_fit.shape[0]/150.
for i in range(Ntria):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[0, 1].add_patch(rect)
    if (i != 0):
        ax[0, 1].axhline(i*Nwave, color='gray', zorder=1)
        ax[0, 1].axvline(i*Nwave, color='gray', zorder=1)
    for j in range(Ntria):
        text = ax[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % t3corr_fit_binned[i*Nwave+1, j*Nwave], va='center', ha='center', size=10)
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
ax[0, 1].set_xlabel('Index $j$')
ax[0, 1].set_ylabel('Index $i$')
temp = np.linspace(0, t3corr_fit.shape[1], Nbase+1)
ax[0, 1].set_xlim([temp[0], temp[-1]])
ax[0, 1].set_xticks(temp)
temp = np.linspace(0, t3corr_fit.shape[0], Nbase+1)
ax[0, 1].set_ylim([temp[-1], temp[0]])
ax[0, 1].set_yticks(temp)
p11 = ax[1, 1].imshow(t3cov_fit, vmin=-0.01, vmax=0.01, extent=[0, t3cov_fit.shape[1], t3cov_fit.shape[0], 0], zorder=0)
c11 = plt.colorbar(p11, ax=ax[1, 1])
c11.set_label('Covariance', rotation=270, labelpad=20)
margin = t3cov_fit.shape[0]/150.
for i in range(Ntria):
    rect = patches.Rectangle((i*Nwave+margin, i*Nwave+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
    ax[1, 1].add_patch(rect)
    if (i != 0):
        ax[1, 1].axhline(i*Nwave, color='gray', zorder=1)
        ax[1, 1].axvline(i*Nwave, color='gray', zorder=1)
ax[1, 1].set_xlabel('Index $j$')
ax[1, 1].set_ylabel('Index $i$')
temp = np.linspace(0, t3cov_fit.shape[1], Nbase+1)
ax[1, 1].set_xlim([temp[0], temp[-1]])
ax[1, 1].set_xticks(temp)
temp = np.linspace(0, t3cov_fit.shape[0], Nbase+1)
ax[1, 1].set_ylim([temp[-1], temp[0]])
ax[1, 1].set_yticks(temp)
plt.tight_layout()
plt.savefig('figures/t3fit_SC.pdf')
plt.show()

import pdb; pdb.set_trace()


"""
Compute the basis transform from the visibility phases to the closure phases
from the closure phase matrix and demonstrate that the structure of the closure
phase correlation matrix can be obtained by assuming uncorrelated visibility
phases and a simple basis transform.
"""
#idir = '/Users/jenskammerer/Downloads/data/1s_ut/'
#fitsfiles = ['GRAVI.2019-03-29T02-01-37.193_singlecalvis.fits']
#insname = 'GRAVITY_SC'
#
#wavel, dwavel, vis2, vis2_err, vis2_u, vis2_v, vis2_sta, t3, t3_err, t3_sta = inout.load_oifits(idir, fitsfiles, insname)
#uu, vv, cp_mat = fitting.make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta)
#
#trans = np.zeros((cp_mat.shape[0]*wavel.shape[1], cp_mat.shape[1]*wavel.shape[1]))
#for i in range(wavel.shape[1]):
#    for j in range(cp_mat.shape[0]):
#        for k in range(cp_mat.shape[1]):
#            trans[wavel.shape[1]*j+i, i+wavel.shape[1]*k] = cp_mat[j, k]
#visphicorr = np.eye(trans.shape[1])
#t3corr = trans.dot(visphicorr).dot(trans.T)
#
#import pdb; pdb.set_trace()
