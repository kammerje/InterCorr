from __future__ import division

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import time

# Set interactive plots to off and change default plot font size.
plt.ioff()
plt.rcParams.update({'font.size': 12})

#import sys
#sys.path.append('/Volumes/OZ 1/Python/Packages/CANDID/CANDID')
#import candid

import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from scipy.interpolate import Rbf
from scipy.linalg import block_diag
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.special import j1
import scipy.stats as stats
import sys

def construct_cp_mat(vis2_sta,
                     t3_sta):
    """
    """
    
    print('--> Computing closure phase matrix')
    
    Nsta = len(np.unique(vis2_sta[0]))
    Nbase = int(Nsta*(Nsta-1)/2)
    Ntria = int(Nsta*(Nsta-1)*(Nsta-2)/6)
    
    # Initialize the closure phase matrix
    cp_mat = np.zeros((Ntria, Nbase))
    
    # Fill the closure phase matrix
    for i in range(Ntria):
        b1 = t3_sta[0, i][[0, 1]]
        b2 = t3_sta[0, i][[1, 2]]
        b3 = t3_sta[0, i][[2, 0]]
        for j in range(Nbase):
            temp = vis2_sta[0, j]
            if (np.array_equal(b1, temp)):
                cp_mat[i, j] = 1
            elif (np.array_equal(b2, temp)):
                cp_mat[i, j] = 1
            elif (np.array_equal(b3, temp)):
                cp_mat[i, j] = 1
            elif (np.array_equal(b1[::-1], temp)):
                cp_mat[i, j] = -1
            elif (np.array_equal(b2[::-1], temp)):
                cp_mat[i, j] = -1
            elif (np.array_equal(b3[::-1], temp)):
                cp_mat[i, j] = -1
    
    return cp_mat

def linear_fit(y0, y):
    """
    """
    
    return y-y0

def fit_vis2corr(wavel, vis, vis_sta, f1f2, percentile=50., stdlim=0.5, full_output=False, visamp=False):
    """
    """
    
    Nsta = len(np.unique(vis_sta[0]))
    Nbase = int(Nsta*(Nsta-1)/2)
    Nob = vis.shape[0]
    Nobp2vmred = int(vis.shape[1]/Nbase)
    Nwave = vis.shape[2]
    
    # NOTE: compute visibility amplitudes and select equal number of good
    # measurements for all baselines.
    vis2 = np.zeros((vis.shape[0], Nobp2vmred, Nwave*Nbase))
    flag = np.zeros((vis.shape[0], Nobp2vmred, Nwave*Nbase))
    Ngood = []
    for i in range(Nob):
        for j in range(Nbase):
            if (visamp == False):
                vis2[i, :, j*Nwave:(j+1)*Nwave] = np.abs(vis[i, j::Nbase])**2/f1f2[i, j::Nbase]
            else:
                vis2[i, :, j*Nwave:(j+1)*Nwave] = np.abs(vis[i, j::Nbase])/np.sqrt(f1f2[i, j::Nbase])
            pp = np.nanpercentile(vis2[i, :, j*Nwave:(j+1)*Nwave], percentile)
            ww1 = np.nanmedian(vis2[i, :, j*Nwave:(j+1)*Nwave], axis=1) > pp
            ww2 = np.std(vis2[i, :, j*Nwave:(j+1)*Nwave], axis=1) < stdlim
            flag[i, :, j*Nwave:(j+1)*Nwave][ww1 & ww2] = 1
        Ngood += [np.sum(flag[i, :, 0::Nwave], axis=0)]
        if (vis2.shape[1] <= 1000):
            f, axarr = plt.subplots(1, Nbase, sharey=True)
            for j in range(Nbase):
                dt = vis2[0, :, j*Nwave:(j+1)*Nwave][flag[0, :, j*Nwave:(j+1)*Nwave] <= 0.5]
                sz = dt.shape[0]
                if (sz > 0):
                    dt = dt.reshape((int(sz/Nwave), Nwave)).T
                    axarr[j].plot(dt, alpha=1./3.)
                dt = vis2[0, :, j*Nwave:(j+1)*Nwave][flag[0, :, j*Nwave:(j+1)*Nwave] > 0.5]
                sz = dt.shape[0]
                if (sz > 0):
                    dt = dt.reshape((int(sz/Nwave), Nwave)).T
                    axarr[j].plot(dt)
#            plt.show()
            plt.close()
    flag = flag > 0.5
    print('Median of vis2 computed from p2vmred: %.3f' % np.nanmedian(vis2))
    print('Median of vis2 after filtering: %.3f' % np.nanmedian(vis2[flag]))
    
    # NOTE: from here only first p2vmred file is considered!
    Ngood_min = int(np.min(Ngood[0]))
    print('Number of good measurements for each baseline: '+str(Ngood[0]))
    print('Cropping vis2 matrix to %.0f measurements' % Ngood_min)
    vis2_good = np.zeros((vis2.shape[0], Ngood_min, vis2.shape[2]))
    for i in range(1):
        for j in range(Nbase):
            vis2_good[i, :, j*Nwave:(j+1)*Nwave] = vis2[i, :, j*Nwave:(j+1)*Nwave][flag[i, :, j*Nwave]][:Ngood_min]
    
#    f, ax = plt.subplots(1, 6, figsize=(6.4*2, 4.8*1))
#    for i in range(1):
#        for j in range(Nbase):
#            temp = vis2[i, :, j*Nwave:(j+1)*Nwave]
#            temp = np.nanmedian(temp, axis=1)
#            if (j == Nbase-1):
#                ax[j].hist(temp, range=(0.2, 0.6), label='no cut')
#            else:
#                ax[j].hist(temp, range=(0.2, 0.6))
#            temp = vis2_good[i, :, j*Nwave:(j+1)*Nwave]
#            temp = np.nanmedian(temp, axis=1)
#            if (j == Nbase-1):
#                ax[j].hist(temp, range=(0.2, 0.6), label='50% cut')
#            else:
#                ax[j].hist(temp, range=(0.2, 0.6))
#            ax[j].set_xlabel('$|V|^2$')
#            if (j == 0):
#                ax[j].set_ylabel('Number')
#            ax[j].set_title('Baseline %.0f' % j)
#            if (j == Nbase-1):
#                ax[j].legend()
#    plt.tight_layout()
#    plt.show()
#    
#    import pdb; pdb.set_trace()
    
#    vis2cov = np.cov(vis2[0].T)
    vis2cov_good = np.cov(vis2_good[0].T)
    
#    dd = np.diag(vis2cov)
#    vis2corr = np.true_divide(vis2cov, np.sqrt(dd[:, None]*dd[None, :]))
    dd = np.diag(vis2cov_good)
    vis2corr_good = np.true_divide(vis2cov_good, np.sqrt(dd[:, None]*dd[None, :]))
    
    # NOTE: computing a binned version of the correlation matrix (disregarding
    # the diagonal)
    vis2corr_binned = np.zeros_like(vis2corr_good)
    dd = np.diag(vis2corr_good).copy()
    np.fill_diagonal(vis2corr_good, np.nan)
    for i in range(Nbase):
        for j in range(Nbase):
            vis2corr_binned[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = np.nanmean(vis2corr_good[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave])
    np.fill_diagonal(vis2corr_good, dd)
    np.fill_diagonal(vis2corr_binned, dd)
    
    # 2 parameter fit
    vis2corr_fit = np.zeros_like(vis2corr_good)
    di = []
    oi = []
    for i in range(Nbase):
        for j in range(i+1):
            if (i == j):
                di += [vis2corr_binned[i*Nwave, j*Nwave+1]]
            else:
                bb = np.intersect1d(vis_sta[0, 0:6][i], vis_sta[0, 0:6][j])
                if (len(bb) == 1):
                    oi += [vis2corr_binned[i*Nwave, j*Nwave+1]]
    di = np.array(di)
    oi = np.array(oi)
#    do = leastsq(linear_fit, np.mean(di), args=(di))[0]
#    oo = leastsq(linear_fit, np.mean(oi), args=(oi))[0]
    ci = np.append(di, 2*oi)
    co = leastsq(linear_fit, np.mean(ci), args=(ci))[0]
    for i in range(Nbase):
        for j in range(Nbase):
            if (i == j):
#                vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = do
                vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = co
            else:
#                vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = 0.
                bb = np.intersect1d(vis_sta[0, 0:6][i], vis_sta[0, 0:6][j])
                if (len(bb) == 1):
                    vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = 1./2.*co
                else:
                    vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = 0.
    np.fill_diagonal(vis2corr_fit, 1.)
    
    dd = np.diag(vis2cov_good)
    vis2cov_fit = np.multiply(vis2corr_fit, np.sqrt(dd[:, None]*dd[None, :]))
    
    margin = vis2corr_good.shape[0]/125.
    f, axarr = plt.subplots(2, 2, figsize=(6.4*2, 4.8*2), gridspec_kw = {'height_ratios': [1, 1]})
    p = axarr[0, 0].imshow(vis2corr_good, cmap='seismic_r', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 0], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Nbase):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 0].add_patch(rect)
        if (i != 0):
            axarr[0, 0].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 0].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Nbase):
        for j in range(Nbase):
            text = axarr[0, 0].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % vis2corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
            if (i != j):
                if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                    rect = patches.Rectangle((i*Nwave-0.5+margin, j*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                    axarr[0, 0].add_patch(rect)
    axarr[0, 0].set_title('Correlation of visibility amplitudes (extracted)')
    p = axarr[0, 1].imshow(vis2corr_fit, cmap='seismic_r', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 1], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Nbase):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 1].add_patch(rect)
        if (i != 0):
            axarr[0, 1].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 1].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Nbase):
        for j in range(Nbase):
            text = axarr[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % vis2corr_fit[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
            if (i != j):
                if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                    rect = patches.Rectangle((i*Nwave-0.5+margin, j*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                    axarr[0, 1].add_patch(rect)
    axarr[0, 1].set_title('Correlation of visibility amplitudes (model)')
    p = axarr[1, 0].imshow(vis2cov_good, vmin=-0.02, vmax=0.02)
    c = plt.colorbar(p, ax=axarr[1, 0])
    c.set_label('Covariance', rotation=270, labelpad=10)
    for i in range(Nbase):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[1, 0].add_patch(rect)
        if (i != 0):
            axarr[1, 0].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[1, 0].axvline(i*Nwave-0.5, color='gray', zorder=1)
        for j in range(Nbase):
            if (i != j):
                if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                    rect = patches.Rectangle((i*Nwave-0.5+margin, j*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                    axarr[1, 0].add_patch(rect)
    axarr[1, 0].set_title('Covariance of visibility amplitudes (extracted)')
    p = axarr[1, 1].imshow(vis2cov_fit, vmin=-0.02, vmax=0.02)
    c = plt.colorbar(p, ax=axarr[1, 1])
    c.set_label('Covariance', rotation=270, labelpad=10)
    for i in range(Nbase):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[1, 1].add_patch(rect)
        if (i != 0):
            axarr[1, 1].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[1, 1].axvline(i*Nwave-0.5, color='gray', zorder=1)
        for j in range(Nbase):
            if (i != j):
                if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                    rect = patches.Rectangle((i*Nwave-0.5+margin, j*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                    axarr[1, 1].add_patch(rect)
    axarr[1, 1].set_title('Covariance of visibility amplitudes (model)')
    plt.tight_layout()
#    plt.savefig('vis2fit.pdf')
#    plt.show(block=True)
    plt.close()
    
    if (full_output == True):
        return vis2corr_fit, vis2cov_fit, vis2corr_good, vis2cov_good
    else:
        return vis2corr_fit, vis2cov_fit

def fit_t3corr(wavel, vis, vis_sta, f1f2, stdlim=1., full_output=False):
    """
    """
    
    Nsta = len(np.unique(vis_sta[0]))
    Nbase = int(Nsta*(Nsta-1)/2)
    Ntria = int(Nsta*(Nsta-1)*(Nsta-2)/6)
    Nob = vis.shape[0]
    Nobp2vmred = int(vis.shape[1]/Nbase)
    Nwave = vis.shape[2]
    
    trias = [[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]]
    
    # NOTE: compute closure phases and select equal number of good
    # measurements for all triangles.
    t3 = np.zeros((vis.shape[0], Nobp2vmred, Nwave*Ntria))
    flag = np.zeros((vis.shape[0], Nobp2vmred, Nwave*Ntria))
    Ngood = []
    for i in range(Nob):
        for j in range(Ntria):
            t3[i, :, j*Nwave:(j+1)*Nwave] = np.angle(vis[i][trias[j][0]::Nbase]*vis[i][trias[j][1]::Nbase]*np.conj(vis[i][trias[j][2]::Nbase]))
            ww = np.std(t3[i, :, j*Nwave:(j+1)*Nwave], axis=1) < stdlim
            flag[i, :, j*Nwave:(j+1)*Nwave][ww] = 1
        Ngood += [np.sum(flag[i, :, 0::Nwave], axis=0)]
        if (t3.shape[1] <= 1000):
            f, axarr = plt.subplots(1, Ntria, sharey=True)
            for j in range(Ntria):
                dt = t3[0, :, j*Nwave:(j+1)*Nwave][flag[0, :, j*Nwave:(j+1)*Nwave] <= 0.5]
                sz = dt.shape[0]
                if (sz > 0):
                    dt = dt.reshape((int(sz/Nwave), Nwave)).T
                    axarr[j].plot(dt, alpha=1./3.)
                dt = t3[0, :, j*Nwave:(j+1)*Nwave][flag[0, :, j*Nwave:(j+1)*Nwave] > 0.5]
                sz = dt.shape[0]
                if (sz > 0):
                    dt = dt.reshape((int(sz/Nwave), Nwave)).T
                    axarr[j].plot(dt)
#            plt.show()
            plt.close()
    flag = flag > 0.5
    print('Median of t3 computed from p2vmred: %.3f' % np.nanmedian(t3))
    print('Median of t3 after filtering: %.3f' % np.nanmedian(t3[flag]))
    
    # NOTE: from here only first p2vmred file is considered!
    Ngood_min = int(np.min(Ngood[0]))
    print('Number of good measurements for each triangle: '+str(Ngood[0]))
    print('Cropping t3 matrix to %.0f measurements' % Ngood_min)
    t3_good = np.zeros((t3.shape[0], Ngood_min, t3.shape[2]))
    for i in range(1):
        for j in range(Ntria):
            t3_good[i, :, j*Nwave:(j+1)*Nwave] = t3[i, :, j*Nwave:(j+1)*Nwave][flag[i, :, j*Nwave]][:Ngood_min]
    
#    f, ax = plt.subplots(1, 4, figsize=(6.4*2, 4.8*1))
#    for i in range(1):
#        for j in range(Ntria):
#            temp = t3[i, :, j*Nwave:(j+1)*Nwave]
#            temp = np.nanmedian(temp, axis=1)
#            if (j == Ntria-1):
#                ax[j].hist(temp, label='no cut')
#            else:
#                ax[j].hist(temp)
#            temp = t3_good[i, :, j*Nwave:(j+1)*Nwave]
#            temp = np.nanmedian(temp, axis=1)
#            if (j == Ntria-1):
#                ax[j].hist(temp, label='50% cut')
#            else:
#                ax[j].hist(temp)
#            ax[j].set_xlabel('$\\theta$')
#            if (j == 0):
#                ax[j].set_ylabel('Number')
#            ax[j].set_title('Triangle %.0f' % j)
#            if (j == Nbase-1):
#                ax[j].legend()
#    plt.tight_layout()
#    plt.show()
#    
#    import pdb; pdb.set_trace()
    
#    t3cov = np.cov(t3[0].T)
    t3cov_good = np.cov(t3_good[0].T)
    
#    dd = np.diag(t3cov)
#    t3corr = np.true_divide(t3cov, np.sqrt(dd[:, None]*dd[None, :]))
    dd = np.diag(t3cov_good)
    t3corr_good = np.true_divide(t3cov_good, np.sqrt(dd[:, None]*dd[None, :]))
    
    # NOTE: computing a binned version of the correlation matrix (disregarding
    # the diagonal)
    t3corr_binned = np.zeros_like(t3corr_good)
    dd = np.diag(t3corr_good).copy()
    np.fill_diagonal(t3corr_good, np.nan)
    for i in range(Ntria):
        for j in range(Ntria):
            temp = np.diag(t3corr_good[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave]).copy()
            np.fill_diagonal(t3corr_good[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], np.nan)
            t3corr_binned[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = np.nanmean(t3corr_good[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave])
            np.fill_diagonal(t3corr_good[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], temp)
    np.fill_diagonal(t3corr_good, dd)
    np.fill_diagonal(t3corr_binned, dd)
    
    # 1 parameter fit
    t3corr_fit = np.zeros_like(t3corr_good)
    di = []
    oi = []
    for i in range(Ntria):
        for j in range(i+1):
            if (i == j):
                di += [t3corr_binned[i*Nwave, j*Nwave+1]]
            else:
                oi += [np.abs(t3corr_binned[i*Nwave, j*Nwave+1])]
    di = np.array(di)
    oi = np.array(oi)
#    do = leastsq(linear_fit, np.mean(di), args=(di))[0]
#    oo = leastsq(linear_fit, np.mean(oi), args=(oi))[0]
    ci = np.append(di, 3*oi)
    co = leastsq(linear_fit, np.mean(ci), args=(ci))[0]
    for i in range(Ntria):
        for j in range(Ntria):
            if (i == j):
#                t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = do
                t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = co
            else:
                bb = np.intersect1d(trias[i], trias[j])
                if (len(bb) == 1):
                    w1 = np.where(np.array(trias[i]) == bb)[0][0]
                    w2 = np.where(np.array(trias[j]) == bb)[0][0]
                    if ((w1 == w2) or (w1 != 2 and w2 != 2)):
#                        t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = oo
                        t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = 1./3.*co
                    else:
#                        t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = -oo
                        t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = -1./3.*co
    for i in range(Ntria):
        for j in range(Ntria):
            if (i != j):
                bb = np.intersect1d(trias[i], trias[j])
                if (len(bb) == 1):
                    w1 = np.where(np.array(trias[i]) == bb)[0][0]
                    w2 = np.where(np.array(trias[j]) == bb)[0][0]
                    if ((w1 == w2) or (w1 != 2 and w2 != 2)):
                        np.fill_diagonal(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], 1./3.)
                    else:
                        np.fill_diagonal(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], -1./3.)
    np.fill_diagonal(t3corr_fit, 1.)
    
    dd = np.diag(t3cov_good)
    t3cov_fit = np.multiply(t3corr_fit, np.sqrt(dd[:, None]*dd[None, :]))
    
    margin = t3corr_good.shape[0]/125.
    f, axarr = plt.subplots(2, 2, figsize=(6.4*2, 4.8*2), gridspec_kw = {'height_ratios': [1, 1]})
    p = axarr[0, 0].imshow(t3corr_good, cmap='seismic_r', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 0], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Ntria):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 0].add_patch(rect)
        if (i != 0):
            axarr[0, 0].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 0].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Ntria):
        for j in range(Ntria):
            text = axarr[0, 0].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % t3corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 0].set_title('Correlation of closure phases (extracted)')
    p = axarr[0, 1].imshow(t3corr_fit, cmap='seismic_r', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 1], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Ntria):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 1].add_patch(rect)
        if (i != 0):
            axarr[0, 1].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 1].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Ntria):
        for j in range(Ntria):
            text = axarr[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % t3corr_fit[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 1].set_title('Correlation of closure phases (model)')
    p = axarr[1, 0].imshow(t3cov_good, vmin=-0.02, vmax=0.02)
    c = plt.colorbar(p, ax=axarr[1, 0])
    c.set_label('Covariance', rotation=270, labelpad=10)
    for i in range(Ntria):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[1, 0].add_patch(rect)
        if (i != 0):
            axarr[1, 0].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[1, 0].axvline(i*Nwave-0.5, color='gray', zorder=1)
    axarr[1, 0].set_title('Covariance of closure phases (extracted)')
    p = axarr[1, 1].imshow(t3cov_fit, vmin=-0.02, vmax=0.02)
    c = plt.colorbar(p, ax=axarr[1, 1])
    c.set_label('Covariance', rotation=270, labelpad=10)
    for i in range(Ntria):
        rect = patches.Rectangle((i*Nwave-0.5+margin, i*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[1, 1].add_patch(rect)
        if (i != 0):
            axarr[1, 1].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[1, 1].axvline(i*Nwave-0.5, color='gray', zorder=1)
    axarr[1, 1].set_title('Covariance of closure phases (model)')
    plt.tight_layout()
#    plt.savefig('t3fit.pdf')
#    plt.show(block=True)
    plt.close()
    
    if (full_output == True):
        return t3corr_fit, t3cov_fit, t3corr_good, t3cov_good
    else:
        return t3corr_fit, t3cov_fit

def get_cov(vis2corr,
            t3corr,
            vis2sigma=0.01,
            t3sigma=np.radians(1.)):
    """
    """
    
    if (isinstance(vis2sigma, float) == True):
        dd = np.repeat(vis2sigma, vis2corr.shape[0], axis=0)
    else:
        dd = vis2sigma
    vis2cov = np.multiply(vis2corr, dd[:, None]*dd[None, :])
    
    if (isinstance(t3sigma, float) == True):
        dd = np.repeat(t3sigma, t3corr.shape[0], axis=0)
    else:
        dd = t3sigma
    t3cov = np.multiply(t3corr, dd[:, None]*dd[None, :])
    
    return vis2cov, t3cov

def get_cov_from_param(vis2,
                       t3,
                       vis2param,
                       t3param,
                       vis2sigma=0.01,
                       t3sigma=np.radians(1.)):
    """
    """
    
    Nm = np.prod(vis2[0].shape)
    Nbase = vis2.shape[1]
    Nwave = vis2.shape[2]
    vis2corr_fit = np.zeros((Nm, Nm))
    for i in range(Nbase):
        for j in range(Nbase):
            if (i == j):
                vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = vis2param
            else:
                vis2corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = 0.
    np.fill_diagonal(vis2corr_fit, 1.)
    
    if (isinstance(vis2sigma, float)):
        dd = np.array([vis2sigma**2]*Nm)
        vis2cov_fit = np.multiply(vis2corr_fit, np.sqrt(dd[:, None]*dd[None, :]))
    else:
        vis2cov_fit = []
        for i in range(vis2sigma.shape[0]):
            dd = vis2sigma[i].flatten()**2
            vis2cov_fit += [np.multiply(vis2corr_fit, np.sqrt(dd[:, None]*dd[None, :]))]
    
    Nm = np.prod(t3[0].shape)
    Ntria = t3.shape[1]
    Nwave = t3.shape[2]
    trias = [[1, 4, 2], [2, 0, 3], [4, 0, 5], [1, 5, 3]]
    t3corr_fit = np.zeros((Nm, Nm))
    for i in range(Ntria):
        for j in range(Ntria):
            if (i == j):
                t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = t3param
            else:
                bb = np.intersect1d(trias[i], trias[j])
                if (len(bb) == 1):
                    w1 = np.where(np.array(trias[i]) == bb)[0][0]
                    w2 = np.where(np.array(trias[j]) == bb)[0][0]
                    if ((w1 == w2) or (w1 != 2 and w2 != 2)):
                        t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = 1./3.*t3param
                    else:
                        t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave] = -1./3.*t3param
    for i in range(Ntria):
        for j in range(Ntria):
            if (i != j):
                bb = np.intersect1d(trias[i], trias[j])
                if (len(bb) == 1):
                    w1 = np.where(np.array(trias[i]) == bb)[0][0]
                    w2 = np.where(np.array(trias[j]) == bb)[0][0]
                    if ((w1 == w2) or (w1 != 2 and w2 != 2)):
                        np.fill_diagonal(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], 1./3.)
                    else:
                        np.fill_diagonal(t3corr_fit[i*Nwave:(i+1)*Nwave, j*Nwave:(j+1)*Nwave], -1./3.)
    np.fill_diagonal(t3corr_fit, 1.)
    
    if (isinstance(t3sigma, float)):
        dd = np.array([t3sigma**2]*Nm)
        t3cov_fit = np.multiply(t3corr_fit, np.sqrt(dd[:, None]*dd[None, :]))
    else:
        t3cov_fit = []
        for i in range(t3sigma.shape[0]):
            dd = t3sigma[i].flatten()**2
            t3cov_fit += [np.multiply(t3corr_fit, np.sqrt(dd[:, None]*dd[None, :]))]
    
    return vis2cov_fit, t3cov_fit

def make_uv_single(wavel,
                   vis2_u,
                   vis2_v,
                   vis2_sta,
                   t3_sta,
                   Nsmear=None,
                   dwavel=None):
    """
    """
    
    if (Nsmear is None):
        Nwave = wavel.shape[1]
        
        uu = np.repeat(vis2_u[:, :, np.newaxis], Nwave, axis=2)
        uu = np.divide(uu, wavel[0])
        vv = np.repeat(vis2_v[:, :, np.newaxis], Nwave, axis=2)
        vv = np.divide(vv, wavel[0])
        
        cp_mat = construct_cp_mat(vis2_sta, t3_sta)
    
    else:
        wavel_smeared = np.zeros((wavel.shape[0], wavel.shape[1]*Nsmear))
        for i in range(wavel.shape[0]):
            for j in range(wavel.shape[1]):
                wavel_smeared[i, j*Nsmear:(j+1)*Nsmear] = np.linspace(wavel[i, j]-dwavel[i, j], wavel[i, j]+dwavel[i, j], Nsmear)
        Nwave = wavel_smeared.shape[1]
        
        uu = np.repeat(vis2_u[:, :, np.newaxis], Nwave, axis=2)
        uu = np.divide(uu, wavel_smeared[0])
        vv = np.repeat(vis2_v[:, :, np.newaxis], Nwave, axis=2)
        vv = np.divide(vv, wavel_smeared[0])
        
        cp_mat = None
    
    return uu, vv, cp_mat

def crop_t3(t3,
            t3cov=None,
            cp_mat=None):
    """
    """
    
    Nwave = t3.shape[2]
    
    t3_cropped = t3.copy()[:, :-1, :]
    
    if (t3cov is not None):
        if (isinstance(t3cov, list) == True):
            t3cov_cropped = []
            if (len(t3cov[0].shape) == 1):
                for i in range(len(t3cov)):
                    t3cov_cropped += [t3cov[i].copy()[:-Nwave]]
            else:
                for i in range(len(t3cov)):
                    t3cov_cropped += [t3cov[i].copy()[:-Nwave, :-Nwave]]
        else:
            if (len(t3cov.shape) == 1):
                t3cov_cropped = t3cov.copy()[:-Nwave]
            else:
                t3cov_cropped = t3cov.copy()[:-Nwave, :-Nwave]
    if (cp_mat is not None):
        cp_mat_cropped = cp_mat.copy()[:-1]
    
    if (t3cov is not None and cp_mat is not None):
        return t3_cropped, t3cov_cropped, cp_mat_cropped
    else:
        return t3_cropped

def vis2vis2(vis):
    """
    """
    
#    vis2 = np.abs(vis)**2
    vis2 = np.abs(vis)
    
    return vis2

def vis2t3(vis,
           cp_mat):
    """
    """
    
    Nobs = vis.shape[0]
    
    t3 = []
    for i in range(Nobs):
        t3 += [cp_mat.dot(np.angle(vis[i]))]
    t3 = np.array(t3)
    
    # NOTE: make sure that the closure phase lies within [-pi, pi)
    t3 = ((t3+np.pi) % (2.*np.pi))-np.pi
    
    return t3

def sim_ud_single(uu,
                  vv,
                  cp_mat,
                  theta=0.,
                  vis2cov=None,
                  t3cov=None,
                  Nsmear=None):
    """
    """
    
    mas2rad = 1./1000./60./60./180.*np.pi
    
    # Complex visibility
    x = np.pi*theta*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    vis = 2.*j1(x)/x
    
    vis2 = vis2vis2(vis)
    t3 = vis2t3(vis,
                cp_mat)
    
    return vis2, t3

def sim_bin_single(uu,
                   vv,
                   cp_mat,
                   f=0.01,
                   alpha=[10, 10],
                   vis2cov=None,
                   t3cov=None):
    """
    """
    
    mas2rad = 1./1000./60./60./180.*np.pi
    
    # Complex visibility
    V1 = 1.+0j
    V2 = 1.+0j
    temp = V2*f*np.exp(-2.*np.pi*1j*(-uu*alpha[0]*mas2rad+vv*alpha[1]*mas2rad))
    vis = (V1+temp)/(1.+f)
    
    vis2 = vis2vis2(vis)
    t3 = vis2t3(vis,
                cp_mat)
    
    return vis2, t3

def sim_ud_bin_single(uu,
                      vv,
                      cp_mat,
                      f=0.01,
                      alpha=[10, 10],
                      theta1=0.,
                      theta2=0.,
                      vis2cov=None,
                      t3cov=None,
                      Nsmear=None):
    """
    """
    
    mas2rad = 1./1000./60./60./180.*np.pi
    
    # Complex visibility
    x = np.pi*theta1*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    V1 = 2.*j1(x)/x
    x = np.pi*theta2*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    V2 = 2.*j1(x)/x
    temp = V2*f*np.exp(-2.*np.pi*1j*(-uu*alpha[0]*mas2rad+vv*alpha[1]*mas2rad))
    vis = (V1+temp)/(1.+f)
    
    vis2 = vis2vis2(vis)
    t3 = vis2t3(vis,
                cp_mat)
    
    return vis2, t3

def sim_ud_bin_single_noise(uu,
                            vv,
                            cp_mat,
                            f=0.01,
                            alpha=[10, 10],
                            theta1=0.,
                            theta2=0.,
                            vis2cov=None,
                            t3cov=None,
                            Nsmear=None):
    """
    """
    
    mas2rad = 1./1000./60./60./180.*np.pi
    Nobs = uu.shape[0]
    
    # Complex visibility
    x = np.pi*theta1*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    V1 = 2.*j1(x)/x
    x = np.pi*theta2*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    V2 = 2.*j1(x)/x
    temp = V2*f*np.exp(-2.*np.pi*1j*(-uu*alpha[0]*mas2rad+vv*alpha[1]*mas2rad))
    vis = (V1+temp)/(1.+f)
    
    if (Nsmear is not None):
        vis = vis.reshape((vis.shape[0], vis.shape[1], vis.shape[2]/Nsmear, Nsmear))
        vis = np.mean(vis, axis=3)
    
    vis2 = vis2vis2(vis)
    t3 = vis2t3(vis,
                cp_mat)
    
    if (vis2cov is not None):
        vis2_temp = []
        for i in range(Nobs):
            sh = vis2[i].shape
            if (len(vis2cov[i].shape) == 1):
                temp = np.random.normal(vis2[i].flatten(), np.sqrt(vis2cov[i]))
            else:
                temp = np.random.multivariate_normal(vis2[i].flatten(), vis2cov[i])
            vis2_temp += [temp.reshape(sh)]
        vis2 = np.array(vis2_temp)
    if (t3cov is not None):
        t3_temp = []
        for i in range(Nobs):
            sh = t3[i].shape
            if (len(t3cov[i].shape) == 1):
                temp = np.random.normal(t3[i].flatten(), np.sqrt(t3cov[i]))
            else:
                temp = np.random.multivariate_normal(t3[i].flatten(), t3cov[i])
            t3_temp += [temp.reshape(sh)]
        t3 = np.array(t3_temp)
    
    return vis2, t3

def chi2_ud(p0, # np.array([theta])
            uu,
            vv,
            cp_mat,
            vis2,
            t3,
            icv=None,
            diag=False):
    """
    """
    
    Nobs = vis2.shape[0]
    
    # Compute visibility amplitudes & closure phases
    vis2_mod, t3_mod = sim_ud_single(uu, vv, cp_mat, theta=p0[0])
    
    # Compute residuals
    sig = np.concatenate((vis2, t3), axis=1)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    res = sig-mod
    
    # Compute chi2
    chi2 = 0.
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (icv is not None):
            if (diag == False):
                chi2 += res_temp.dot(icv).dot(res_temp) # SLOW
            else:
                chi2 += np.multiply(res_temp, icv).dot(res_temp) # FAST
        else:
            chi2 += res_temp.dot(res_temp)
    
    return chi2

def chi2_bin(p0, # np.array([f, alpha[0], alpha[1]])
             uu,
             vv,
             cp_mat,
             vis2,
             t3,
             icv=None,
             diag=False):
    """
    """
    
    Nobs = vis2.shape[0]
    
    # Compute visibility amplitudes & closure phases
    vis2_mod, t3_mod = sim_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:])
    
    # Compute residuals
    sig = np.concatenate((vis2, t3), axis=1)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    res = sig-mod
    
    # Compute chi2
    chi2 = 0.
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (icv is not None):
            if (isinstance(icv, list)):
                if (diag == False):
                    chi2 += res_temp.dot(icv[i]).dot(res_temp) # SLOW
                else:
                    chi2 += np.multiply(res_temp, icv[i]).dot(res_temp) # FAST
            else:
                if (diag == False):
                    chi2 += res_temp.dot(icv).dot(res_temp) # SLOW
                else:
                    chi2 += np.multiply(res_temp, icv).dot(res_temp) # FAST
        else:
            chi2 += res_temp.dot(res_temp)
    
    return chi2

def chi2_bin_hat(p0, # np.array([f, alpha[0], alpha[1]])
                 uu,
                 vv,
                 cp_mat,
                 vis2,
                 t3,
                 icv=None,
                 diag=False):
    """
    """
    
    Nobs = vis2.shape[0]
    
    # Compute visibility amplitudes & closure phases
    vis2_mod, t3_mod = sim_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:])
    
    # Compute residuals
    sig = np.concatenate((vis2, t3), axis=1)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    res = sig-mod
    
    # Compute chi2
    chi2 = 0.
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (icv is not None):
            if (diag == False):
                chi2 += res_temp.dot(icv).dot(res_temp) # SLOW
            else:
                chi2 += np.multiply(res_temp, icv).dot(res_temp) # FAST
        else:
            chi2 += res_temp.dot(res_temp)
    
    import pdb; pdb.set_trace()
    
    X = res.flatten()
    temp = 1./(X.T.dot(icv).dot(X))
    H = (X.dot(temp).dot(X.T))*icv
    Ndof = H.trace()
    
    import pdb; pdb.set_trace()
    
    return chi2

def chi2_ud_bin(p0, # np.array([f, alpha[0], alpha[1], theta1, theta2])
                uu,
                vv,
                cp_mat,
                vis2,
                t3,
                icv=None,
                diag=False):
    """
    """
    
    Nobs = vis2.shape[0]
    
    # Compute visibility amplitudes & closure phases
    vis2_mod, t3_mod = sim_ud_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:3], theta1=p0[3])
    
    # Compute residuals
    sig = np.concatenate((vis2, t3), axis=1)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    res = sig-mod
    
    # Compute chi2
    chi2 = 0.
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (icv is not None):
            if (isinstance(icv, list)):
                if (diag == False):
                    chi2 += res_temp.dot(icv[i]).dot(res_temp) # SLOW
                else:
                    chi2 += np.multiply(res_temp, icv[i]).dot(res_temp) # FAST
            else:
                if (diag == False):
                    chi2 += res_temp.dot(icv).dot(res_temp) # SLOW
                else:
                    chi2 += np.multiply(res_temp, icv).dot(res_temp) # FAST
        else:
            chi2 += res_temp.dot(res_temp)
    
    return chi2

def gridsearch(vis2,
               t3,
               uu,
               vv,
               cp_mat,
               vis2cov,
               t3cov,
               f0=0.001):
    """
    """
    
    print('--> Performing gridsearch')
    
    # Build covariance matrix
    if (isinstance(vis2cov, list)):
        icv = []
        if (len(vis2cov[0].shape) == 2 and len(t3cov[0].shape) == 2):
            for i in range(len(vis2cov)):
                vis2icv = np.linalg.inv(vis2cov[i])
#                t3cov[i] += np.diag(np.array([1e-8]*t3cov[i].shape[0]))
                t3icv = np.linalg.inv(t3cov[i])
                if (np.allclose(vis2cov[i].dot(vis2icv), np.eye(vis2cov[i].shape[0])) == False):
                    raise UserWarning()
                if (np.allclose(t3cov[i].dot(t3icv), np.eye(t3cov[i].shape[0])) == False):
                    raise UserWarning()
                icv += [block_diag(vis2icv, t3icv)]
            diag = False
        elif (len(vis2cov[0].shape) == 1 and len(t3cov[0].shape) == 1):
            for i in range(len(vis2cov)):
                vis2icv = 1./vis2cov[i]
                t3icv = 1./t3cov[i]
                icv += [np.append(vis2icv, t3icv)]
            diag = True
        else:
            raise UserWarning('Covariance has wrong shape')
        Nobs = vis2.shape[0]
        Ndof = (vis2cov[0].shape[0]+t3cov[0].shape[0])*Nobs
    else: 
        if (len(vis2cov.shape) == 2 and len(t3cov.shape) == 2):
            vis2icv = np.linalg.inv(vis2cov)
#            t3cov += np.diag(np.array([1e-8]*t3cov.shape[0]))
            t3icv = np.linalg.inv(t3cov)
            if (np.allclose(vis2cov.dot(vis2icv), np.eye(vis2cov.shape[0])) == False):
                raise UserWarning()
            if (np.allclose(t3cov.dot(t3icv), np.eye(t3cov.shape[0])) == False):
                raise UserWarning()
            icv = block_diag(vis2icv, t3icv)
            diag = False
        elif (len(vis2cov.shape) == 1 and len(t3cov.shape) == 1):
            vis2icv = 1./vis2cov
            t3icv = 1./t3cov
            icv = np.append(vis2icv, t3icv)
            diag = True
        else:
            raise UserWarning('Covariance has wrong shape')
        Nobs = vis2.shape[0]
        Ndof = icv.shape[0]*Nobs
    
    # Compute grid
#    temp = np.linspace(-25, 25, 101) # From -25 to 25 mas in steps of 0.5 mas
#    alpha_u, alpha_v = np.meshgrid(temp, temp)
    temp = np.linspace(-40, 40, 81) # From -40 to 40 mas in steps of 1 mas
    alpha_u, alpha_v = np.meshgrid(temp, temp)
    
    # Go through grid
    chi2s = []
    chi2s_f = []
    fs = []
    dfs = []
    cells = np.prod(alpha_u.shape)
    count = 0
    for i in range(alpha_u.shape[0]):
        for j in range(alpha_u.shape[1]):
            count += 1
            sys.stdout.write('\rCell %.0f of %.0f' % (count, cells))
            sys.stdout.flush()
            
            # Chi2 with f = f0
            p0 = np.array([f0, alpha_u[i, j], alpha_v[i, j]])
            chi2s += [chi2_bin(p0, uu, vv, cp_mat, vis2, t3, icv, diag)]
            
            # Chi2 with f = f_fit from linearization
            vis2_mod, t3_mod = sim_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:])
            vis2_mod = (vis2_mod-1.)/f0
            vis2_sig = vis2-1.
            t3_mod = t3_mod/f0
            t3_sig = t3
            
            MOD = np.concatenate((vis2_mod, t3_mod), axis=1)
            MOD_ICV = []
            SIG = np.concatenate((vis2_sig, t3_sig), axis=1)
            for k in range(Nobs):
                if (isinstance(icv, list)):
                    if (diag == False):
                        MOD_ICV += [MOD[k].flatten().dot(icv[k])]
                    else:
                        MOD_ICV += [np.multiply(MOD[k].flatten(), icv[k])]
                else:
                    if (diag == False):
                        MOD_ICV += [MOD[k].flatten().dot(icv)]
                    else:
                        MOD_ICV += [np.multiply(MOD[k].flatten(), icv)]
            MOD = MOD.flatten()
            MOD_ICV = np.array(MOD_ICV).flatten()
            SIG = SIG.flatten()
            
            fs += [MOD_ICV.dot(SIG)/MOD_ICV.dot(MOD)]
            dfs += [1./np.sqrt(MOD_ICV.dot(MOD))]
            p0[0] = fs[-1]
            chi2s_f += [chi2_bin(p0, uu, vv, cp_mat, vis2, t3, icv, diag)]
    print('')
    
    # Reshape arrays
    chi2s = np.array(chi2s).reshape(alpha_u.shape)
    chi2s_f = np.array(chi2s_f).reshape(alpha_u.shape)
    chi2s_f[np.isnan(chi2s_f)] = np.inf
    
    # Find grid position with minimal chi2
    fs = np.array(fs).reshape(alpha_u.shape)
    dfs = np.array(dfs).reshape(alpha_u.shape)
    red_chi2s = chi2s_f/Ndof
    
    temp = chi2s.copy()
    temp[fs < 0.] = np.inf
    ww = np.argmin(temp)
    alpha = np.array([alpha_u.flatten()[ww], alpha_v.flatten()[ww]])
#    print(alpha)
    temp = chi2s_f.copy()
    temp[fs < 0.] = np.inf
    ww = np.argmin(temp)
    alpha = np.array([alpha_u.flatten()[ww], alpha_v.flatten()[ww]])
#    print(alpha)
    red_chi2 = chi2s_f.flatten()[ww]/Ndof
    
    print('Reduced chi2 = %.3f' % red_chi2)
    print('Best fit contrast = %.3f' % fs.flatten()[ww])
    print('Best fit separation = (%.1f, %.1f) mas' % (alpha[0], alpha[1]))
    
    return fs.flatten()[ww], alpha, red_chi2, fs, dfs, red_chi2s, alpha_u, alpha_v

def chi2_bin_t3(p0, # np.array([f, alpha[0], alpha[1]])
                uu,
                vv,
                cp_mat,
                t3,
                icv=None,
                diag=False):
    """
    """
    
    Nobs = t3.shape[0]
    
    # Compute visibility amplitudes & closure phases
    _, t3_mod = sim_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:])
    
    # Compute residuals
    res = t3-t3_mod
    
    # Compute chi2
    chi2 = 0.
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (icv is not None):
            if (diag == False):
                chi2 += res_temp.dot(icv).dot(res_temp) # SLOW
            else:
                chi2 += np.multiply(res_temp, icv).dot(res_temp) # FAST
        else:
            chi2 += res_temp.dot(res_temp)
    
    return chi2

def gridsearch_t3(t3,
                  uu,
                  vv,
                  cp_mat,
                  t3cov,
                  f0=0.001):
    """
    """
    
    print('--> Performing gridsearch (t3)')
    
    # Build covariance matrix
    if (len(t3cov.shape) == 2):
        icv = np.linalg.inv(t3cov)
        diag = False
    elif (len(t3cov.shape) == 1):
        icv = 1./t3cov
        diag = True
    else:
        raise UserWarning('Covariance has wrong shape')
    
    # Compute grid
#    temp = np.linspace(-25, 25, 101) # From -25 to 25 mas in steps of 0.5 mas
#    alpha_u, alpha_v = np.meshgrid(temp, temp)
    temp = np.linspace(-40, 40, 81) # From -40 to 40 mas in steps of 2 mas
    alpha_u, alpha_v = np.meshgrid(temp, temp)
    
    # Go through grid
    Nobs = t3.shape[0]
    chi2s = []
    chi2s_f = []
    fs = []
    dfs = []
    cells = np.prod(alpha_u.shape)
    count = 0
    for i in range(alpha_u.shape[0]):
        for j in range(alpha_u.shape[1]):
            count += 1
            sys.stdout.write('\rCell %.0f of %.0f' % (count, cells))
            sys.stdout.flush()
            
            # Chi2 with f = f0
            p0 = np.array([f0, alpha_u[i, j], alpha_v[i, j]])
            chi2s += [chi2_bin_t3(p0, uu, vv, cp_mat, t3, icv, diag)]
            
            # Chi2 with f = f_fit from linearization
            _, t3_mod = sim_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:])
            t3_mod = t3_mod/f0
            t3_sig = t3
            
            MOD = t3_mod
            MOD_ICV = []
            SIG = t3_sig
            for k in range(Nobs):
                if (diag == False):
                    MOD_ICV += [MOD[k].flatten().dot(icv)]
                else:
                    MOD_ICV += [np.multiply(MOD[k].flatten(), icv)]
            MOD = MOD.flatten()
            MOD_ICV = np.array(MOD_ICV).flatten()
            SIG = SIG.flatten()
            
            fs += [MOD_ICV.dot(SIG)/MOD_ICV.dot(MOD)]
            dfs += [1./np.sqrt(MOD_ICV.dot(MOD))]
            p0[0] = fs[-1]
            chi2s_f += [chi2_bin_t3(p0, uu, vv, cp_mat, t3, icv, diag)]
    print('')
    
    # Reshape arrays
    chi2s = np.array(chi2s).reshape(alpha_u.shape)
    chi2s_f = np.array(chi2s_f).reshape(alpha_u.shape)
    chi2s_f[np.isnan(chi2s_f)] = np.inf
    
    # Find grid position with minimal chi2
    fs = np.array(fs).reshape(alpha_u.shape)
    dfs = np.array(dfs).reshape(alpha_u.shape)
    red_chi2s = chi2s_f/icv.shape[0]/Nobs
    
    temp = chi2s.copy()
    temp[fs < 0.] = np.inf
    ww = np.argmin(temp)
    alpha = np.array([alpha_u.flatten()[ww], alpha_v.flatten()[ww]])
    print(alpha)
    temp = chi2s_f.copy()
    temp[fs < 0.] = np.inf
    ww = np.argmin(temp)
    alpha = np.array([alpha_u.flatten()[ww], alpha_v.flatten()[ww]])
    print(alpha)
    red_chi2 = chi2s_f.flatten()[ww]/icv.shape[0]/Nobs
    
    print('Reduced chi2 = %.3f' % red_chi2)
    print('Best fit contrast = %.3f' % fs.flatten()[ww])
    print('Best fit separation = (%.1f, %.1f) mas' % (alpha[0], alpha[1]))
    
    return fs.flatten()[ww], alpha, red_chi2, fs, dfs, red_chi2s, alpha_u, alpha_v

def Nsigma(chi2r_test, chi2r_true, Ndof):
    """
    """
    
#    p = stats.chi2.cdf(Ndof, Ndof*chi2r_test/chi2r_true)
#    Nsigma = np.sqrt(stats.chi2.ppf(1.-p, 1.))
#    
#    return Nsigma
    
    p = stats.chi2.cdf(Ndof, Ndof*chi2r_test/chi2r_true)
    log10p = np.log10(np.maximum(p, 1e-161)) # 50 sigma max.
    Nsigma = np.sqrt(stats.chi2.ppf(1.-p, 1.))
    
    # Not sure what is going on here...
#    c = np.array([-0.25028407,  9.66640457])
    c = np.array([-0.29842513,  3.55829518])
    if (isinstance(Nsigma, np.ndarray)):
        Nsigma[log10p < -15.] = np.polyval(c, log10p[log10p < -15.])
        Nsigma = np.nan_to_num(Nsigma)
        Nsigma += 90*(Nsigma == 0)
    else:
        if (log10p < -15.):
            Nsigma =  np.polyval(c, log10p)
        if (np.isnan(Nsigma)):
            Nsigma = 90.
    
    return Nsigma

def chi2_ud_leastsq(p0, # np.array([theta])
                    xdata,
                    ydata,
                    sigma_inv,
                    cp_mat,
                    Nobs,
                    diag=False,
                    Nsmear=None):
    """
    """
    
    uu = xdata[0]
    vv = xdata[1]
    
    vis2_mod, t3_mod = sim_ud_single(uu, vv, cp_mat, theta=p0[0], Nsmear=Nsmear)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    
    chi2 = []
    for i in range(Nobs):
        r = ydata[i]-mod[i].flatten()
        if (diag == True):
            chi2 += [np.sum((r*sigma_inv[i])**2)]
        else:
            chi2 += [r.dot(sigma_inv[i]).dot(r)]
    
    return np.sqrt(np.array(chi2)) # Minimize chi2 (i.e. square of this array)

def chi2_ud_minimize(p0, # np.array([theta])
                     xdata,
                     ydata,
                     sigma_inv,
                     cp_mat,
                     Nobs,
                     diag=False,
                     Nsmear=None):
    """
    """
    
    uu = xdata[0]
    vv = xdata[1]
    
    vis2_mod, t3_mod = sim_ud_single(uu, vv, cp_mat, theta=p0[0], Nsmear=Nsmear)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    
    chi2 = []
    for i in range(Nobs):
        r = ydata[i]-mod[i].flatten()
        if (diag == True):
            chi2 += [np.sum((r*sigma_inv[i])**2)]
        else:
            chi2 += [r.dot(sigma_inv[i]).dot(r)]
    
    return np.sum(chi2)

def chi2_ud_bin_leastsq(p0, # np.array([f, alpha[0], alpha[1], theta1])
                        xdata,
                        ydata,
                        sigma_inv,
                        cp_mat,
                        Nobs,
                        diag=False,
                        Nsmear=None):
    """
    """
    
    uu = xdata[0]
    vv = xdata[1]
    
    vis2_mod, t3_mod = sim_ud_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:3], theta1=p0[3], Nsmear=Nsmear)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    
    chi2 = []
    for i in range(Nobs):
        r = ydata[i]-mod[i].flatten()
        if (diag == True):
            chi2 += [np.sum((r*sigma_inv[i])**2)]
        else:
            chi2 += [r.dot(sigma_inv[i]).dot(r)]
    
    return np.sqrt(np.array(chi2)) # Minimize chi2 (i.e. square of this array)

def chi2_ud_bin_minimize(p0, # np.array([f, alpha[0], alpha[1], theta1])
                         xdata,
                         ydata,
                         sigma_inv,
                         cp_mat,
                         Nobs,
                         diag=False,
                         Nsmear=None):
    """
    """
    
    uu = xdata[0]
    vv = xdata[1]
    
    vis2_mod, t3_mod = sim_ud_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:3], theta1=p0[3], Nsmear=Nsmear)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    
    chi2 = []
    for i in range(Nobs):
        r = ydata[i]-mod[i].flatten()
        if (diag == True):
            chi2 += [np.sum((r*sigma_inv[i])**2)]
        else:
            chi2 += [r.dot(sigma_inv[i]).dot(r)]
    
    return np.sum(np.array(chi2))

class chi2_ud_curvefit_helper():
    
    def __init__(self,
                 cp_mat,
                 Nobs,
                 Nsmear=None):
        """
        """
        
        self.cp_mat = cp_mat
        self.Nobs = Nobs
        self.Nsmear = Nsmear
        
        pass
    
    def chi2_ud_curvefit(self,
                         xdata,
                         p0): # np.array([theta])
        """
        """
        
        uu = xdata[0]
        vv = xdata[1]
        
        vis2_mod, t3_mod = sim_ud_single(uu, vv, self.cp_mat, theta=p0, Nsmear=self.Nsmear)
        ydata = []
        for i in range(self.Nobs):
            ydata += [np.concatenate((vis2_mod[i].flatten(), t3_mod[i].flatten()))]
        ydata = np.concatenate(ydata)
        
        return ydata
    
    def chi2_ud_bin_curvefit(self,
                             xdata,
                             p0,
                             p1,
                             p2,
                             p3): # np.array([f, alpha[0], alpha[1], theta1])
        """
        """
        
        uu = xdata[0]
        vv = xdata[1]
        
        vis2_mod, t3_mod = sim_ud_bin_single(uu, vv, self.cp_mat, f=p0, alpha=[p1, p2], theta1=p3, Nsmear=self.Nsmear)
        ydata = []
        for i in range(self.Nobs):
            ydata += [np.concatenate((vis2_mod[i].flatten(), t3_mod[i].flatten()))]
        ydata = np.concatenate(ydata)
        
        return ydata

def gridsearch_leastsq(vis2,
                       t3,
                       uu,
                       vv,
                       cp_mat,
                       vis2cov,
                       t3cov,
                       f0=0.001,
                       Nsmear=None,
                       wavel=None,
                       dwavel=None,
                       vis2_u=None,
                       vis2_v=None,
                       vis2_sta=None,
                       t3_sta=None):
    """
    """
    
    print('--> Performing gridsearch_leastsq')
    
    # Build covariance matrix
    if (isinstance(vis2cov, list)):
        icv = []
        if (len(vis2cov[0].shape) == 2 and len(t3cov[0].shape) == 2):
            for i in range(len(vis2cov)):
                vis2icv = np.linalg.inv(vis2cov[i])
#                t3cov[i] += np.diag(np.array([1e-8]*t3cov[i].shape[0]))
                t3icv = np.linalg.inv(t3cov[i])
                if (np.allclose(vis2cov[i].dot(vis2icv), np.eye(vis2cov[i].shape[0])) == False):
                    raise UserWarning()
                if (np.allclose(t3cov[i].dot(t3icv), np.eye(t3cov[i].shape[0])) == False):
                    raise UserWarning()
                icv += [block_diag(vis2icv, t3icv)]
            diag = False
        elif (len(vis2cov[0].shape) == 1 and len(t3cov[0].shape) == 1):
            for i in range(len(vis2cov)):
                vis2icv = 1./vis2cov[i]
                t3icv = 1./t3cov[i]
                icv += [np.append(vis2icv, t3icv)]
            diag = True
        else:
            raise UserWarning('Covariance has wrong shape')
        Nobs = vis2.shape[0]
        Ndof = (vis2cov[0].shape[0]+t3cov[0].shape[0])*Nobs
    else: 
        if (len(vis2cov.shape) == 2 and len(t3cov.shape) == 2):
            vis2icv = np.linalg.inv(vis2cov)
#            t3cov += np.diag(np.array([1e-8]*t3cov.shape[0]))
            t3icv = np.linalg.inv(t3cov)
            if (np.allclose(vis2cov.dot(vis2icv), np.eye(vis2cov.shape[0])) == False):
                raise UserWarning()
            if (np.allclose(t3cov.dot(t3icv), np.eye(t3cov.shape[0])) == False):
                raise UserWarning()
            icv = block_diag(vis2icv, t3icv)
            diag = False
        elif (len(vis2cov.shape) == 1 and len(t3cov.shape) == 1):
            vis2icv = 1./vis2cov
            t3icv = 1./t3cov
            icv = np.append(vis2icv, t3icv)
            diag = True
        else:
            raise UserWarning('Covariance has wrong shape')
        Nobs = vis2.shape[0]
        Ndof = icv.shape[0]*Nobs
    
    # Compute grid
#    temp = np.linspace(-25, 25, 51) # From -25 to 25 mas in steps of 1 mas
#    alpha_u, alpha_v = np.meshgrid(temp, temp)
#    temp = np.linspace(-25, 25, 26) # From -25 to 25 mas in steps of 2 mas
#    alpha_u_sparse, alpha_v_sparse = np.meshgrid(temp, temp)
#    extent = [-25.5, 25.5]
    temp = np.linspace(-40, 40, 81) # From -40 to 40 mas in steps of 1 mas
    alpha_u, alpha_v = np.meshgrid(temp, temp)
    temp = np.linspace(-40, 40, 41) # From -40 to 40 mas in steps of 2 mas
    alpha_u_sparse, alpha_v_sparse = np.meshgrid(temp, temp)
    """
    Note: choose 21 steps for injection and recovery tests.
    """
    extent = [-41., 41.]
    
    # Bandwidth smearing
    if (Nsmear is not None):
        Nsmear = int(Nsmear)
        uu, vv, _ = make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta, Nsmear=Nsmear, dwavel=dwavel)
    
    # Fit uniform disk
    theta0 = np.array([1.5]) # Uniform disk diameter in mas
    xdata = [uu, vv]
    ydata_lst = []
    for i in range(Nobs):
        ydata_lst += [np.concatenate((vis2[i].flatten(), t3[i].flatten()))]
#    ydata = np.concatenate(ydata_lst)
    if (diag == True):
        for i in range(Nobs):
            icv[i] = np.sqrt(icv[i])
#        sigma = 1./np.concatenate(icv)
#    else:
#        sigma_inv = block_diag(*icv)
#        sigma = np.linalg.inv(sigma_inv)
    
#    t0 = time.time()
#    theta1 = leastsq(chi2_ud_leastsq,
#                     theta0,
#                     args=(xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear),
#                     full_output=True,
#                     epsfcn=1e-8,
#                     ftol=1e-5,
#                     maxfev=1000)
#    t1 = time.time()
#    print('theta = %.3f' % theta1[0])
#    print('chi2 = %.3f' % (np.sum(theta1[2]['fvec']**2)/Ndof))
#    print('time = %.3f ms' % ((t1-t0)*1e3))
    
#    t0 = time.time()
#    theta9 = least_squares(chi2_ud_leastsq,
#                           theta0,
#                           args=(xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear),
#                           ftol=1e-5,
#                           max_nfev=1000)
#    t1 = time.time()
#    print('theta = %.3f' % theta9['x'])
#    print('chi2 = %.3f' % (np.sum(theta9['fun']**2)/Ndof))
#    print('time = %.3f ms' % ((t1-t0)*1e3))
    
    t0 = time.time()
    theta2 = minimize(chi2_ud_minimize,
                      theta0,
                      args=(xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear),
                      bounds=[(0., np.inf)],
                      tol=1e-5,
                      options={'maxiter': 1000})
    t1 = time.time()
    print('theta = %.3f' % theta2['x'])
    print('chi2 = %.3f' % (theta2['fun']/Ndof))
    print('time = %.3f ms' % ((t1-t0)*1e3))
    
#    helper = chi2_ud_curvefit_helper(cp_mat, Nobs, Nsmear)
#    t0 = time.time()
#    theta3 = curve_fit(helper.chi2_ud_curvefit,
#                       xdata,
#                       ydata,
#                       theta0,
#                       sigma=sigma,
#                       absolute_sigma=True,
#                       full_output=True,
#                       epsfcn=1e-8,
#                       ftol=1e-5,
#                       maxfev=1000)
#    t1 = time.time()
#    chi2 = chi2_ud_minimize(theta3[0], xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear)
#    print('theta = %.3f' % theta3[0])
#    print('chi2 = %.3f' % (chi2/Ndof))
#    print('time = %.3f ms' % ((t1-t0)*1e3))
    
#    import pdb; pdb.set_trace()
    
    # Go through grid
    p0s = []
#    p1s = []
#    p1s_chi2s = []
#    p9s = []
#    p9s_chi2s = []
    p2s = []
    p2s_chi2s = []
#    p3s = []
#    p3s_chi2s = []
    cells = np.prod(alpha_u_sparse.shape)
    count = 0
    t0 = time.time()
    for i in range(alpha_u_sparse.shape[0]):
        for j in range(alpha_u_sparse.shape[1]):
            count += 1
            sys.stdout.write('\rCell %.0f of %.0f' % (count, cells))
            sys.stdout.flush()
            
            p0 = np.array([f0, alpha_u_sparse[i, j], alpha_v_sparse[i, j], theta2['x']], dtype='float64')
            p0s += [p0]
            
#            p1 = leastsq(chi2_ud_bin_leastsq,
#                         p0,
#                         args=(xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear),
#                         full_output=True,
#                         epsfcn=1e-8,
#                         ftol=1e-5,
#                         maxfev=1000)
#            p1s += [p1[0]]
#            p1s_chi2s += [np.sum(p1[2]['fvec']**2)/Ndof]
            
#            p9 = least_squares(chi2_ud_bin_leastsq,
#                               p0,
#                               args=(xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear),
#                               ftol=1e-5,
#                               max_nfev=1000)
#            p9s += [p9['x']]
#            p9s_chi2s += [np.sum(p9['fun']**2)/Ndof]
            
            p2 = minimize(chi2_ud_bin_minimize,
                          p0,
                          args=(xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear),
                          bounds=[(0, 1), (-np.inf, np.inf), (-np.inf, np.inf), (0., np.inf)],
                          tol=1e-5,
                          options={'maxiter': 1000})
            p2s += [p2['x']]
            p2s_chi2s += [p2['fun']/Ndof]
            
#            p3 = curve_fit(helper.chi2_ud_bin_curvefit,
#                           xdata,
#                           ydata,
#                           p0,
#                           sigma=sigma,
#                           absolute_sigma=True,
#                           full_output=True,
#                           epsfcn=1e-8,
#                           ftol=1e-5,
#                           maxfev=1000)
#            chi2 = chi2_ud_bin_minimize(p3[0], xdata, ydata_lst, icv, cp_mat, Nobs, diag, Nsmear)
#            p3s += [p3[0]]
#            p3s_chi2s += [chi2/Ndof]
            
#            import pdb; pdb.set_trace()
    
    t1 = time.time()
    print('')
    print('time = %.3f s' % (t1-t0))
    
#    import pdb; pdb.set_trace()
    
    # Reshape arrays
    p0s = np.array(p0s)
    s0s = np.array(p2s)
    s0s_chi2s = np.array(p2s_chi2s)
    
    # Find unique minima
    s0s_unique = [s0s[0]]
    s0s_chi2s_unique = [s0s_chi2s[0]]
    for i in range(1, s0s.shape[0]):
        diffs = (np.array(s0s_unique)-s0s[i])[:, 1:]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if (np.sum(dists < 0.5) > 0):
            continue
        s0s_unique += [s0s[i]]
        s0s_chi2s_unique += [s0s_chi2s[i]]
    s0s_unique = np.array(s0s_unique)
    s0s_chi2s_unique = np.array(s0s_chi2s_unique)
    
    dist = np.sqrt(s0s_unique[:, 1]**2+s0s_unique[:, 2]**2)
    mask = dist <= np.abs(extent[0])
    x = s0s_unique[:, 1][mask]
    y = s0s_unique[:, 2][mask]
    z = s0s_chi2s_unique[mask]
    
    # Fit with radial basis function
    func = Rbf(s0s_unique[:, 1], s0s_unique[:, 2], s0s_chi2s_unique, function='linear')
    chi2_map = np.flipud(func(alpha_u.flatten(), alpha_v.flatten()).reshape(alpha_u.shape))
    dist = np.sqrt(alpha_u**2+alpha_v**2)
    chi2_map[dist > np.abs(extent[0])] = z.max()
    
    # Print
    best = s0s[np.argmin(s0s_chi2s)]
    chi2r_test = theta2['fun']/Ndof
    chi2r_true = np.min(s0s_chi2s)
    sig = Nsigma(chi2r_test, chi2r_true, Ndof)
    print('Binary fit:')
    print('--> Companion flux = %.4f of host star flux' % best[0])
    print('--> Companion offset = (%.2f, %.2f) mas' % (best[1], best[2]))
    print('--> Disk diameter = %.3f mas' % best[3])
    print('--> Reduced chi2 (ud) = %.3f' % chi2r_test)
    print('--> Reduced chi2 (bin) = %.3f' % chi2r_true)
    print('--> Nsigma of detection = %.3f' % sig)
    
#    # Plot
#    cmap = plt.cm.get_cmap('cubehelix_r')
##    targ = [-15, -15]
#    step = (np.max(alpha_u)-np.min(alpha_u))/10.
#    plt.figure()
#    plt.imshow(chi2_map, cmap=cmap, extent=[extent[0], extent[1], extent[0], extent[1]], vmin=z.min(), vmax=z.max())
#    plt.scatter(x, y, c=z, cmap=cmap, vmin=z.min(), vmax=z.max())
#    cb = plt.colorbar()
#    cb.set_label('Reduced $\chi^2$', rotation=270, labelpad=20)
#    for i in range(p0s.shape[0]):
#        plt.plot([p0s[i, 1], s0s[i, 1]], [p0s[i, 2], s0s[i, 2]], color='yellow', alpha=1./3.)
##    cc = plt.Rectangle((targ[0]-step, targ[1]-step), 2.*step, 2.*step, color='black', lw=5, fill=False, zorder=1)
##    plt.gca().add_artist(cc)
##    cc = plt.Rectangle((targ[0]-step, targ[1]-step), 2.*step, 2.*step, color='cyan', lw=2.5, fill=False, zorder=1)
##    plt.gca().add_artist(cc)    
#    cc = plt.Circle((best[1], best[2]), step, color='black', lw=5, fill=False, zorder=1)
#    plt.gca().add_artist(cc)
#    cc = plt.Circle((best[1], best[2]), step, color='cyan', lw=2.5, fill=False, zorder=1)
#    plt.gca().add_artist(cc)    
#    text = plt.text(0.05, 0.95, '$f_{fit}$ = %.3f%%' % (best[0]*100.)+', $\Delta$RA = %.1f' % best[1]+', $\Delta$DEC = %.1f' % best[2], ha='left', va='top', transform=plt.gca().transAxes, zorder=2)
#    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
#    text = plt.text(0.05, 0.05, '$\chi^2$ = %.3f (best)' % np.min(s0s_chi2s), ha='left', va='bottom', transform=plt.gca().transAxes, zorder=2)
#    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
#    text = plt.text(0.05, 0.125, 'N$\\sigma$ = %.3f' % sig, ha='left', va='bottom', transform=plt.gca().transAxes, zorder=2)
#    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])    
#    plt.xlim([extent[0], extent[1]])
#    plt.xlabel('$\Delta$RA [$mas$]')
#    plt.ylim([extent[0], extent[1]])
#    plt.ylabel('$\Delta$DEC [$mas$]')
#    plt.tight_layout()
##    plt.savefig('diag.pdf')
#    plt.show(block=True)
#    plt.close()
#    import pdb; pdb.set_trace()
    
    return s0s, s0s_chi2s, chi2r_test, sig
#    return s0s, s0s_chi2s, best, sig, alpha_u, alpha_v

def chi2_bin_leastsq_t3(p0, # np.array([f, alpha[0], alpha[1]])
                        uu,
                        vv,
                        cp_mat,
                        t3,
                        icv=None,
                        diag=False,
                        Nsmear=None):
    """
    """
    
    if (Nsmear is not None):
        raise UserWarning
    
    Nobs = t3.shape[0]
    
    # Compute visibility amplitudes & closure phases
    _, t3_mod = sim_bin_single(uu, vv, cp_mat, f=p0[0], alpha=p0[1:])
    
    # Compute residuals
    sig = t3
    mod = t3_mod
    res = sig-mod
    
    # Compute chi2
    chi2 = []
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (icv is not None):
            if (isinstance(icv, list)):
                if (diag == False):
                    chi2 += [res_temp.dot(icv[i]).dot(res_temp)] # SLOW
                else:
                    chi2 += [np.multiply(res_temp, icv[i]).dot(res_temp)] # FAST
            else:
                if (diag == False):
                    chi2 += [res_temp.dot(icv).dot(res_temp)] # SLOW
                else:
                    chi2 += [np.multiply(res_temp, icv).dot(res_temp)] # FAST
        else:
            chi2 += [res_temp.dot(res_temp)]
    
    return np.array(chi2) # Minimize chi2 (i.e. square of this array)

def gridsearch_leastsq_t3(t3,
                          uu,
                          vv,
                          cp_mat,
                          t3cov,
                          f0=0.001,
                          Nsmear=None,
                          wavel=None,
                          dwavel=None,
                          vis2_u=None,
                          vis2_v=None,
                          vis2_sta=None,
                          t3_sta=None):
    """
    """
    
    print('--> Performing gridsearch_leastsq')
    
    # Build covariance matrix
    if (isinstance(t3cov, list)):
        icv = []
        if (len(t3cov[0].shape)):
            for i in range(len(t3cov)):
                icv += [np.linalg.inv(t3cov[i])]
            diag = False
        elif (len(t3cov[0].shape) == 1):
            for i in range(len(t3cov)):
                icv += [1./t3cov[i]]
            diag = True
        else:
            raise UserWarning('Covariance has wrong shape')
        Nobs = t3.shape[0]
        Ndof = t3cov[0].shape[0]*Nobs
    else: 
        if (len(t3cov.shape) == 2):
            icv = np.linalg.inv(t3cov)
            diag = False
        elif (len(t3cov.shape) == 1):
            icv = 1./t3cov
            diag = True
        else:
            raise UserWarning('Covariance has wrong shape')
        Nobs = t3.shape[0]
        Ndof = icv.shape[0]*Nobs
    
    # Compute grid
#    temp = np.linspace(-25, 25, 51) # From -25 to 25 mas in steps of 1 mas
#    alpha_u, alpha_v = np.meshgrid(temp, temp)
#    temp = np.linspace(-25, 25, 26) # From -25 to 25 mas in steps of 2 mas
#    alpha_u_sparse, alpha_v_sparse = np.meshgrid(temp, temp)
#    extent = [-25.5, 25.5]
    temp = np.linspace(-40, 40, 41) # From -40 to 40 mas in steps of 2 mas
    alpha_u, alpha_v = np.meshgrid(temp, temp)
    temp = np.linspace(-40, 40, 21) # From -40 to 40 mas in steps of 4 mas
    alpha_u_sparse, alpha_v_sparse = np.meshgrid(temp, temp)
    extent = [-41., 41.]
    
    # Bandwidth smearing
    if (Nsmear is not None):
        Nsmear = int(Nsmear)
        uu, vv, _ = make_uv_single(wavel, vis2_u, vis2_v, vis2_sta, t3_sta, Nsmear=Nsmear, dwavel=dwavel)
    
    # Fit uniform disk
    theta = np.array([0.]) # Uniform disk diameter in mas
    import pdb; pdb.set_trace()
    
    # Go through grid
    p0s = []
    s0s = []
    s0s_chi2s = []
    cells = np.prod(alpha_u.shape)
    count = 0
    for i in range(alpha_u.shape[0]):
        for j in range(alpha_u.shape[1]):
            count += 1
            sys.stdout.write('\rCell %.0f of %.0f' % (count, cells))
            sys.stdout.flush()
            
            # Least squares fitting
            p0 = np.array([f0, alpha_u[i, j], alpha_v[i, j], theta[0], 0.])
            if ((alpha_u[i, j] in alpha_u_sparse) and (alpha_v[i, j] in alpha_v_sparse)):
                p0s += [p0]
                s0 = leastsq(chi2_bin_leastsq_t3,
                             p0,
                             args=(uu, vv, cp_mat, t3, icv, diag, Nsmear),
                             full_output=True,
                             epsfcn=1e-8,
                             ftol=1e-5,
                             maxfev=1000)
                s0s += [s0[0]]
                s0s_chi2s += [np.sum(chi2_bin_leastsq_t3(s0[0], uu, vv, cp_mat, t3, icv, diag, Nsmear))]
    print('')
    
    # Reshape arrays
    p0s = np.array(p0s)
    s0s = np.array(s0s)
    s0s_chi2s = np.array(s0s_chi2s)/Ndof
    
    # Find unique minima
    s0s_unique = [s0s[0]]
    s0s_chi2s_unique = [s0s_chi2s[0]]
    for i in range(1, s0s.shape[0]):
        diffs = (np.array(s0s_unique)-s0s[i])[:, 1:]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if (np.sum(dists < 0.5) > 0):
            continue
        s0s_unique += [s0s[i]]
        s0s_chi2s_unique += [s0s_chi2s[i]]
    s0s_unique = np.array(s0s_unique)
    s0s_chi2s_unique = np.array(s0s_chi2s_unique)
    
    dist = np.sqrt(s0s_unique[:, 1]**2+s0s_unique[:, 2]**2)
    mask = dist <= np.abs(extent[0])
    x = s0s_unique[:, 1][mask]
    y = s0s_unique[:, 2][mask]
    z = s0s_chi2s_unique[mask]
    
    # Fit with radial basis function
    func = Rbf(s0s_unique[:, 1], s0s_unique[:, 2], s0s_chi2s_unique, function='linear')
    chi2_map = np.flipud(func(alpha_u.flatten(), alpha_v.flatten()).reshape(alpha_u.shape))
    dist = np.sqrt(alpha_u**2+alpha_v**2)
    chi2_map[dist > np.abs(extent[0])] = z.max()
    
    # Print
#    chi2r_test = np.sum(theta[2]['fvec'])/Ndof
#    chi2r_true = np.min(s0s_chi2s)
#    sig = Nsigma(chi2r_test, chi2r_true, Ndof)
#    print('Reduced chi2 (ud) = %.3f' % chi2r_test)
#    print('Reduced chi2 (bin) = %.3f' % chi2r_true)
#    print('Nsigma of detection = %.3f' % sig)
    
    # Plot
    cmap = plt.cm.get_cmap('cubehelix_r')
    best = s0s[np.argmin(s0s_chi2s)]
    targ = [-10, 10]
    step = (np.max(alpha_u)-np.min(alpha_u))/10.
    plt.figure()
    plt.imshow(chi2_map, cmap=cmap, extent=[extent[0], extent[1], extent[0], extent[1]], vmin=z.min(), vmax=z.max())
    plt.scatter(x, y, c=z, cmap=cmap, vmin=z.min(), vmax=z.max())
    cb = plt.colorbar()
    cb.set_label('Reduced $\chi^2$', rotation=270, labelpad=20)
    for i in range(p0s.shape[0]):
        plt.plot([p0s[i, 1], s0s[i, 1]], [p0s[i, 2], s0s[i, 2]], color='yellow', alpha=1./3.)
    cc = plt.Rectangle((targ[0]-step, targ[1]-step), 2.*step, 2.*step, color='black', lw=5, fill=False, zorder=1)
    plt.gca().add_artist(cc)
    cc = plt.Rectangle((targ[0]-step, targ[1]-step), 2.*step, 2.*step, color='cyan', lw=2.5, fill=False, zorder=1)
    plt.gca().add_artist(cc)
    cc = plt.Circle((best[1], best[2]), step, color='black', lw=5, fill=False, zorder=1)
    plt.gca().add_artist(cc)
    cc = plt.Circle((best[1], best[2]), step, color='cyan', lw=2.5, fill=False, zorder=1)
    plt.gca().add_artist(cc)    
    text = plt.text(0.05, 0.95, '$f_{fit}$ = %.5f' % best[0]+', $\Delta$RA = %.1f' % best[1]+', $\Delta$DEC = %.1f' % best[2], ha='left', va='top', transform=plt.gca().transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = plt.text(0.05, 0.05, '$\chi^2$ = %.3f (best)' % np.min(s0s_chi2s), ha='left', va='bottom', transform=plt.gca().transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
#    text = plt.text(0.05, 0.125, 'N$\\sigma$ = %.3f' % sig, ha='left', va='bottom', transform=plt.gca().transAxes, zorder=2)
#    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])    
    plt.xlim([extent[0], extent[1]])
    plt.xlabel('$\Delta$RA [$mas$]')
    plt.ylim([extent[0], extent[1]])
    plt.ylabel('$\Delta$DEC [$mas$]')
    plt.tight_layout()
#    plt.savefig('chi2_diag_t3.pdf')
    plt.show(block=True)
    plt.close()
    
    import pdb; pdb.set_trace()
    
    pass
