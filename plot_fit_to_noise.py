import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os
import time
np.random.seed(1994)

import sys
sys.path.append('../corr2')
sys.path.append('/Volumes/OZ 1/Python/Packages/opticstools/opticstools/opticstools')
sys.path.append('../opticstools/opticstools/opticstools')

import inout
import fitting
import plotlib
import opticstools as ot

from scipy.linalg import block_diag
from scipy.optimize import leastsq
from scipy.special import j1
import scipy.stats as stats

# Set interactive plots to off and change default plot font size.
plt.ioff()
plt.rcParams.update({'font.size': 12})


"""
Functions.
"""
def azavg(img):
    """
    TODO
    """
    
    # Compute azimuthal average using opticstools
    radii, azavg = ot.azimuthalAverage(img, returnradii=True, binsize=1.)
    
    # Return azimuthal average, the first value is always inf (this is
    # because the grid search finds an infinite contrast for the central
    # pixel where the binary model is not defined)
    return radii[1:], azavg[1:]


"""
Reading.
"""
#idir = 'fit_to_noise_new/'
idir = 'test/'
fitsfiles = [f for f in os.listdir(idir) if f.startswith('diag') and f.endswith('.fits')]
fitsfiles = np.sort(fitsfiles)
Nf = len(fitsfiles)

fs_diag = []
dfs_diag = []
for i in range(Nf):
    hdul = pyfits.open(idir+fitsfiles[i])
    fs_diag += [hdul['fs'].data]
    dfs_diag += [hdul['dfs'].data]
fs_diag = np.array(fs_diag)
dfs_diag = np.array(dfs_diag)

#idir = 'fit_to_noise_new/'
idir = 'test/'
fitsfiles = [f for f in os.listdir(idir) if f.startswith('full') and f.endswith('.fits')]
fitsfiles = np.sort(fitsfiles)
Nf = len(fitsfiles)

fs_full = []
dfs_full = []
for i in range(Nf):
    hdul = pyfits.open(idir+fitsfiles[i])
    fs_full += [hdul['fs'].data]
    dfs_full += [hdul['dfs'].data]
fs_full = np.array(fs_full)
dfs_full = np.array(dfs_full)


"""
Analysis.
"""
xs = []
ys_diag = []
ys_full = []
for i in range(Nf):
    radiis, azavgs = azavg(np.abs(fs_diag[i]))
    xs += [radiis]
    ys_diag += [azavgs]
    radiis, azavgs = azavg(np.abs(fs_full[i]))
    ys_full += [azavgs]
xs = np.array(xs)
ys_diag = np.array(ys_diag)
ys_full = np.array(ys_full)

plt.figure()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax1 = plt.gca()
l1 = ax1.plot(xs[0], np.mean(ys_diag, axis=0), color=colors[0], label='uncorrelated errors')
ax1.fill_between(xs[0], np.mean(ys_diag, axis=0)-np.std(ys_diag, axis=0), np.mean(ys_diag, axis=0)+np.std(ys_diag, axis=0), edgecolor=None, facecolor=colors[0], alpha=1./3.)
l2 = ax1.plot(xs[0], np.mean(ys_full, axis=0), color=colors[1], label='correlated errors')
ax1.fill_between(xs[0], np.mean(ys_full, axis=0)-np.std(ys_full, axis=0), np.mean(ys_full, axis=0)+np.std(ys_full, axis=0), edgecolor=None, facecolor=colors[1], alpha=1./3.)
ax1.set_xlim([0, 50])
ax1.set_xlabel('Angular separation [mas]')
#temp = [-13./3., -8./3.]
temp = [-7./3., -2./3.]
ax1.set_ylim([10**temp[0], 10**temp[1]])
ax1.set_yscale('log')
ax1.set_ylabel('Brightness of noise')
ax2 = ax1.twinx()
l3 = ax2.plot(xs[0], np.mean(ys_diag, axis=0)/np.mean(ys_full, axis=0), label='ratio', color='black', ls='--')
ax2.set_ylim([0., 10.])
ax2.set_ylabel('Ratio', rotation=270, labelpad=20)
ax2.grid()
ls = l1+l2+l3
la = [l.get_label() for l in ls]
ax1.legend(ls, la, loc='upper right')
plt.tight_layout()
#plt.savefig('fit_to_noise_sim.pdf')
plt.savefig('fit_to_noise_real.pdf')
plt.show(block=True)

import pdb; pdb.set_trace()
