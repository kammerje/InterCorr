import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import FormatStrFormatter

import sys
sys.path.append('/Volumes/OZ 1/Python/Packages/opticstools/opticstools/opticstools')
sys.path.append('../opticstools/opticstools/opticstools')

import opticstools as ot

def azavg(img):
    """
    """
    
    # Compute azimuthal average using opticstools
    radii, azavg = ot.azimuthalAverage(img, returnradii=True, binsize=1.)
    
    # Return azimuthal average, the first value is always inf (this is
    # because the grid search finds an infinite contrast for the central
    # pixel where the binary model is not defined)
    return radii[1:], azavg[1:]

def vis2(wavel,
         vis2,
         vis2_err,
         vis2_sta,
         binning=None):
    """
    """
    
    Nbase = vis2.shape[1]
    Nob = vis2.shape[0]
    
    if (binning is not None):
        binning = int(binning)
        Nbins = wavel.shape[1]/binning
        plot_wavel = np.zeros((wavel.shape[0], Nbins))
        plot_vis2 = np.zeros((vis2.shape[0], vis2.shape[1], Nbins))
        plot_vis2_err = np.zeros((vis2.shape[0], vis2.shape[1], Nbins))
        for i in range(wavel.shape[0]):
            for j in range(Nbins):
                plot_wavel[i, j] = np.nanmean(wavel[i, j*binning:(j+1)*binning])
        for i in range(vis2.shape[0]):
            for j in range(vis2.shape[1]):
                for k in range(Nbins):
                    plot_vis2[i, j, k] = np.nanmedian(vis2[i, j, k*binning:(k+1)*binning])
                    plot_vis2_err[i, j, k] = np.nanstd(vis2[i, j, k*binning:(k+1)*binning])
        print('Median of vis2 error from OIFITS: %.3f' % np.nanmedian(vis2_err))
        print('Median of vis2 error from binning: %.3f' % np.nanmedian(plot_vis2_err))
        
    else:
        plot_wavel = wavel
        plot_vis2 = vis2
        plot_vis2_err = vis2_err
    
    f, axarr = plt.subplots(1, Nbase, figsize=(6.4*2., 4.8), sharey=True, gridspec_kw = {'wspace': 1./3., 'hspace': 0.})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(Nbase):
        for j in range(Nob):
            axarr[i].plot(plot_wavel[j]*1e6, plot_vis2[j, i], color=colors[j % len(colors)], label='OB %.0f' % (j+1))
            axarr[i].fill_between(plot_wavel[j]*1e6, plot_vis2[j, i]-plot_vis2_err[j, i], plot_vis2[j, i]+plot_vis2_err[j, i], edgecolor=None, facecolor=colors[j % len(colors)], alpha=0.5)
        axarr[i].set_xlim([np.min(wavel[0])*1e6, np.max(wavel[0])*1e6])
        axarr[i].set_xticks([np.min(wavel[0])*1e6, np.mean(wavel[0])*1e6, np.max(wavel[0])*1e6])
        axarr[i].set_xticklabels([np.min(wavel[0])*1e6, np.mean(wavel[0])*1e6, np.max(wavel[0])*1e6])
        axarr[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axarr[i].set_xlabel('$\lambda$ [$\mu m$]')
        axarr[i].set_ylim([0.7, 1.1])
        if (i == 0):
            axarr[i].set_ylabel('$|V|^2$')
        axarr[i].grid(axis='y')
        if (i == Nbase-1):
            axarr[i].legend(loc='upper right', framealpha=1)
        axarr[i].set_title(str(vis2_sta[0][i]))
    plt.tight_layout()
    plt.savefig('vis2.pdf')
    plt.show(block=True)
    plt.close()
    
    pass

def t3(wavel,
       t3,
       t3_err,
       t3_sta,
       binning=None):
    """
    """
    
    Ntria = t3.shape[1]
    Nob = t3.shape[0]
    
    if (binning is not None):
        binning = int(binning)
        Nbins = wavel.shape[1]/binning
        plot_wavel = np.zeros((wavel.shape[0], Nbins))
        plot_t3 = np.zeros((t3.shape[0], t3.shape[1], Nbins))
        plot_t3_err = np.zeros((t3.shape[0], t3.shape[1], Nbins))
        for i in range(wavel.shape[0]):
            for j in range(Nbins):
                plot_wavel[i, j] = np.nanmean(wavel[i, j*binning:(j+1)*binning])
        for i in range(t3.shape[0]):
            for j in range(t3.shape[1]):
                for k in range(Nbins):
                    plot_t3[i, j, k] = np.nanmedian(t3[i, j, k*binning:(k+1)*binning])
                    plot_t3_err[i, j, k] = np.nanstd(t3[i, j, k*binning:(k+1)*binning])
        print('Median of t3 error from OIFITS: %.3f' % np.nanmedian(t3_err))
        print('Median of t3 error from binning: %.3f' % np.nanmedian(plot_t3_err))
        
    else:
        plot_wavel = wavel
        plot_t3 = t3
        plot_t3_err = t3_err
    
    f, axarr = plt.subplots(1, Ntria, figsize=(6.4*2., 4.8), sharey=True, gridspec_kw = {'wspace': 1./3., 'hspace': 0.})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(Ntria):
        for j in range(Nob):
            axarr[i].plot(plot_wavel[j]*1e6, plot_t3[j, i], color=colors[j % len(colors)], label='OB %.0f' % (j+1))
            axarr[i].fill_between(plot_wavel[j]*1e6, plot_t3[j, i]-plot_t3_err[j, i], plot_t3[j, i]+plot_t3_err[j, i], edgecolor=None, facecolor=colors[j % len(colors)], alpha=0.5)
        axarr[i].set_xlim([np.min(wavel[0])*1e6, np.max(wavel[0])*1e6])
        axarr[i].set_xticks([np.min(wavel[0])*1e6, np.mean(wavel[0])*1e6, np.max(wavel[0])*1e6])
        axarr[i].set_xticklabels([np.min(wavel[0])*1e6, np.mean(wavel[0])*1e6, np.max(wavel[0])*1e6])
        axarr[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axarr[i].set_xlabel('$\lambda$ [$\mu m$]')
        axarr[i].set_ylim([-0.2, 0.2])
        if (i == 0):
            axarr[i].set_ylabel('$\\theta$ [$rad$]')
        axarr[i].grid(axis='y')
        if (i == Ntria-1):
            axarr[i].legend(loc='upper right', framealpha=1)
        axarr[i].set_title(str(t3_sta[0][i]))
    plt.tight_layout()
    plt.savefig('t3.pdf')
    plt.show(block=True)
    plt.close()
    
    pass

def vis2_corr_p2vmred(wavel,
                      vis,
                      vis_sta,
                      f1f2,
                      percentile=50.,
                      stdlim=0.5,
                      return_good=False):
    """
    """
    
    Nsta = len(np.unique(vis_sta[0]))
    Nbase = Nsta*(Nsta-1)/2
    Nob = vis.shape[0]
    Nobp2vmred = vis.shape[1]/Nbase
    Nwave = vis.shape[2]
    
    # NOTE: compute visibility amplitudes and select equal number of good
    # measurements for all baselines.
    vis2 = np.zeros((vis.shape[0], Nobp2vmred, Nwave*Nbase))
    flag = np.zeros((vis.shape[0], Nobp2vmred, Nwave*Nbase))
    Ngood = []
    for i in range(Nob):
        for j in range(Nbase):
            vis2[i, :, j*Nwave:(j+1)*Nwave] = np.abs(vis[i, j::Nbase])**2/f1f2[i, j::Nbase]
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
                    dt = dt.reshape((sz/Nwave, Nwave)).T
                    axarr[j].plot(dt, alpha=1./3.)
                dt = vis2[0, :, j*Nwave:(j+1)*Nwave][flag[0, :, j*Nwave:(j+1)*Nwave] > 0.5]
                sz = dt.shape[0]
                if (sz > 0):
                    dt = dt.reshape((sz/Nwave, Nwave)).T
                    axarr[j].plot(dt)
                axarr[j].set_ylim([0, 5])
            plt.show()
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
    
    if (return_good == True):
        return vis2_good[0]
    
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
    
    f, axarr = plt.subplots(2, 2, figsize=(6.4*2, 4.8*1.25), gridspec_kw = {'height_ratios': [4, 1]})
    p = axarr[0, 0].imshow(vis2corr_good, cmap='RdBu', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 0], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Nbase):
        rect = patches.Rectangle((i*Nwave-0.5, i*Nwave-0.5), Nwave, Nwave, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 0].add_patch(rect)
        if (i != 0):
            axarr[0, 0].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 0].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Nbase):
        for j in range(Nbase):
            text = axarr[0, 0].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % vis2corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 0].set_title('Correlation of visibility amplitudes (%.0f meas.)' % Ngood_min)
    p = axarr[0, 1].imshow(vis2corr_binned, cmap='RdBu', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 1], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Nbase):
        rect = patches.Rectangle((i*Nwave-0.5, i*Nwave-0.5), Nwave, Nwave, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 1].add_patch(rect)
        if (i != 0):
            axarr[0, 1].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 1].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Nbase):
        for j in range(Nbase):
            text = axarr[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % vis2corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 1].set_title('Correlation of visibility amplitude (binned)')
    axarr[1, 0].plot(np.diag(vis2cov_good))
    for i in range(Nbase):
        if (i != 0):
            axarr[1, 0].axvline(i*Nwave-0.5, lw=3, color='red')
        axarr[1, 0].text((1./Nbase)*(i+0.5), 0.8, str(vis_sta[0, i]), va='center', ha='center', transform=axarr[1, 0].transAxes)
    axarr[1, 0].set_xlim([0, Nwave*Nbase-1])
    axarr[1, 0].set_ylim(bottom=0.)
    axarr[1, 0].set_ylabel('Variance')
    axarr[1, 0].grid(axis='y')
    axarr[1, 1].plot(np.diag(vis2cov_good))
    for i in range(Nbase):
        if (i != 0):
            axarr[1, 1].axvline(i*Nwave-0.5, lw=3, color='red')
        axarr[1, 1].text((1./Nbase)*(i+0.5), 0.8, str(vis_sta[0, i]), va='center', ha='center', transform=axarr[1, 1].transAxes)
    axarr[1, 1].set_xlim([0, Nwave*Nbase-1])
    axarr[1, 1].set_ylim(bottom=0.)
    axarr[1, 1].set_ylabel('Variance')
    axarr[1, 1].grid(axis='y')
    plt.tight_layout()
    plt.savefig('vis2corr.pdf')
    plt.show(block=True)
    plt.close()
    
    pass

def vis2_corr(wavel,
              vis_sta,
              vis2_good):
    """
    """
    
    Nsta = len(np.unique(vis_sta[0]))
    Nbase = Nsta*(Nsta-1)/2
    Nwave = wavel.shape[1]
    Ngood_min = vis2_good.shape[0]
    
    #    vis2cov = np.cov(vis2[0].T)
    vis2cov_good = np.cov(vis2_good.T)
    
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
    
    margin = vis2corr_good.shape[0]/125.
    f, axarr = plt.subplots(2, 2, figsize=(6.4*2, 4.8*1.25), gridspec_kw = {'height_ratios': [4, 1]})
    p = axarr[0, 0].imshow(vis2corr_good, cmap='RdBu', vmin=-1, vmax=1)
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
    axarr[0, 0].set_title('Correlation of visibility amplitudes (%.0f meas.)' % Ngood_min)
    p = axarr[0, 1].imshow(vis2corr_binned, cmap='RdBu', vmin=-1, vmax=1)
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
            text = axarr[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % vis2corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
            if (i != j):
                if (len(np.intersect1d(vis_sta[0, i], vis_sta[0, j])) > 0):
                    rect = patches.Rectangle((i*Nwave-0.5+margin, j*Nwave-0.5+margin), Nwave-2.*margin, Nwave-2.*margin, lw=3, edgecolor='orange', facecolor='none', zorder=2)
                    axarr[0, 1].add_patch(rect)
    axarr[0, 1].set_title('Correlation of visibility amplitudes (binned)')
    axarr[1, 0].plot(np.diag(vis2cov_good))
    for i in range(Nbase):
        if (i != 0):
            axarr[1, 0].axvline(i*Nwave-0.5, lw=3, color='red')
        axarr[1, 0].text((1./Nbase)*(i+0.5), 0.8, str(vis_sta[0, i]), va='center', ha='center', transform=axarr[1, 0].transAxes)
    axarr[1, 0].set_xlim([0, Nwave*Nbase-1])
    axarr[1, 0].set_ylim(bottom=0.)
    axarr[1, 0].set_ylabel('Variance')
    axarr[1, 0].grid(axis='y')
    axarr[1, 1].plot(np.diag(vis2cov_good))
    for i in range(Nbase):
        if (i != 0):
            axarr[1, 1].axvline(i*Nwave-0.5, lw=3, color='red')
        axarr[1, 1].text((1./Nbase)*(i+0.5), 0.8, str(vis_sta[0, i]), va='center', ha='center', transform=axarr[1, 1].transAxes)
    axarr[1, 1].set_xlim([0, Nwave*Nbase-1])
    axarr[1, 1].set_ylim(bottom=0.)
    axarr[1, 1].set_ylabel('Variance')
    axarr[1, 1].grid(axis='y')
    plt.tight_layout()
    plt.savefig('vis2corr.pdf')
    plt.show(block=True)
    plt.close()
    
    pass

def t3_corr_p2vmred(wavel,
                    vis,
                    vis_sta,
                    f1f2,
                    stdlim=1.,
                    return_good=False):
    """
    """
    
    Nsta = len(np.unique(vis_sta[0]))
    Nbase = Nsta*(Nsta-1)/2
    Ntria = Nsta*(Nsta-1)*(Nsta-2)/6
    Nob = vis.shape[0]
    Nobp2vmred = vis.shape[1]/Nbase
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
                    dt = dt.reshape((sz/Nwave, Nwave)).T
                    axarr[j].plot(dt, alpha=1./3.)
                dt = t3[0, :, j*Nwave:(j+1)*Nwave][flag[0, :, j*Nwave:(j+1)*Nwave] > 0.5]
                sz = dt.shape[0]
                if (sz > 0):
                    dt = dt.reshape((sz/Nwave, Nwave)).T
                    axarr[j].plot(dt)
            plt.show()
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
    
    if (return_good == True):
        return t3_good[0]
    
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
    
    f, axarr = plt.subplots(2, 2, figsize=(6.4*2, 4.8*1.25), gridspec_kw = {'height_ratios': [4, 1]})
    p = axarr[0, 0].imshow(t3corr_good, cmap='RdBu', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 0], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Ntria):
        rect = patches.Rectangle((i*Nwave-0.5, i*Nwave-0.5), Nwave, Nwave, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 0].add_patch(rect)
        if (i != 0):
            axarr[0, 0].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 0].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Ntria):
        for j in range(Ntria):
            text = axarr[0, 0].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % t3corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 0].set_title('Correlation of closure phases (%.0f meas.)' % Ngood_min)
    p = axarr[0, 1].imshow(t3corr_binned, cmap='RdBu', vmin=-1, vmax=1)
    c = plt.colorbar(p, ax=axarr[0, 1], ticks=[-1.0, -0.5, 0.0, 0.5, 1.0], format='%.1f')
    c.set_label('Correlation', rotation=270, labelpad=10)
    for i in range(Ntria):
        rect = patches.Rectangle((i*Nwave-0.5, i*Nwave-0.5), Nwave, Nwave, lw=3, edgecolor='red', facecolor='none', zorder=2)
        axarr[0, 1].add_patch(rect)
        if (i != 0):
            axarr[0, 1].axhline(i*Nwave-0.5, color='gray', zorder=1)
            axarr[0, 1].axvline(i*Nwave-0.5, color='gray', zorder=1)
    for i in range(Ntria):
        for j in range(Ntria):
            text = axarr[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % t3corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 1].set_title('Correlation of closure phase (binned)')
    axarr[1, 0].plot(np.diag(t3cov_good))
    for i in range(Ntria):
        if (i != 0):
            axarr[1, 0].axvline(i*Nwave-0.5, lw=3, color='red')
        axarr[1, 0].text((1./Ntria)*(i+0.5), 0.8, str(trias[i]), va='center', ha='center', transform=axarr[1, 0].transAxes)
    axarr[1, 0].set_xlim([0, Nwave*Ntria-1])
    axarr[1, 0].set_ylim(bottom=0.)
    axarr[1, 0].set_ylabel('Variance')
    axarr[1, 0].grid(axis='y')
    axarr[1, 1].plot(np.diag(t3cov_good))
    for i in range(Ntria):
        if (i != 0):
            axarr[1, 1].axvline(i*Nwave-0.5, lw=3, color='red')
        axarr[1, 1].text((1./Ntria)*(i+0.5), 0.8, str(trias[i]), va='center', ha='center', transform=axarr[1, 1].transAxes)
    axarr[1, 1].set_xlim([0, Nwave*Ntria-1])
    axarr[1, 1].set_ylim(bottom=0.)
    axarr[1, 1].set_ylabel('Variance')
    axarr[1, 1].grid(axis='y')
    plt.tight_layout()
    plt.savefig('t3corr.pdf')
    plt.show(block=True)
    plt.close()
    
    pass

def t3_corr(wavel,
            vis_sta,
            t3_good):
    """
    """
    
    Nsta = len(np.unique(vis_sta[0]))
    Ntria = Nsta*(Nsta-1)*(Nsta-2)/6
    Nwave = wavel.shape[1]
    Ngood_min = t3_good.shape[0]
    
    trias = [[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]]
    
#    t3cov = np.cov(t3[0].T)
    t3cov_good = np.cov(t3_good.T)
    
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
    
    margin = t3corr_good.shape[0]/125.
    f, axarr = plt.subplots(2, 2, figsize=(6.4*2, 4.8*1.25), gridspec_kw = {'height_ratios': [4, 1]})
    p = axarr[0, 0].imshow(t3corr_good, cmap='RdBu', vmin=-1, vmax=1)
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
    axarr[0, 0].set_title('Correlation of closure phases (%.0f meas.)' % Ngood_min)
    p = axarr[0, 1].imshow(t3corr_binned, cmap='RdBu', vmin=-1, vmax=1)
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
            text = axarr[0, 1].text((i+0.5)*Nwave-0.5, (j+0.5)*Nwave-0.5, '%.1e' % t3corr_binned[i*Nwave+1, j*Nwave], va='center', ha='center')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 1].set_title('Correlation of closure phase (binned)')
    axarr[1, 0].plot(np.diag(t3cov_good))
    for i in range(Ntria):
        if (i != 0):
            axarr[1, 0].axvline(i*Nwave-0.5, lw=3, color='red')
        stas = [vis_sta[0][j] for j in trias[i]]
        axarr[1, 0].text((1./Ntria)*(i+0.5), 0.8, str(np.unique(np.concatenate(stas))), va='center', ha='center', transform=axarr[1, 0].transAxes)
    axarr[1, 0].set_xlim([0, Nwave*Ntria-1])
    axarr[1, 0].set_ylim(bottom=0.)
    axarr[1, 0].set_ylabel('Variance')
    axarr[1, 0].grid(axis='y')
    axarr[1, 1].plot(np.diag(t3cov_good))
    for i in range(Ntria):
        if (i != 0):
            axarr[1, 1].axvline(i*Nwave-0.5, lw=3, color='red')
        stas = [vis_sta[0][j] for j in trias[i]]
        axarr[1, 1].text((1./Ntria)*(i+0.5), 0.8, str(np.unique(np.concatenate(stas))), va='center', ha='center', transform=axarr[1, 1].transAxes)
    axarr[1, 1].set_xlim([0, Nwave*Ntria-1])
    axarr[1, 1].set_ylim(bottom=0.)
    axarr[1, 1].set_ylabel('Variance')
    axarr[1, 1].grid(axis='y')
    plt.tight_layout()
    plt.savefig('t3corr.pdf')
    plt.show(block=True)
    plt.close()
    
    pass

def snr2(f1,
         alpha1,
         red_chi21,
         fs1,
         dfs1,
         alpha_u1,
         alpha_v1,
         f2,
         alpha2,
         red_chi22,
         fs2,
         dfs2,
         alpha_u2,
         alpha_v2,
         f1_in=None,
         alpha1_in=None,
         f2_in=None,
         alpha2_in=None,
         filename='snr2'):
    """
    """
    
    snr1 = np.true_divide(fs1, dfs1)
    dist = np.sqrt(alpha_u1**2+alpha_v1**2)
    snr1[dist > np.max(alpha_u1)] = 0.
    snr2 = np.true_divide(fs2, dfs2)
    dist = np.sqrt(alpha_u2**2+alpha_v2**2)
    snr2[dist > np.max(alpha_u2)] = 0.
    
    Npix1 = alpha_u1.shape[0]
    imhs1 = (Npix1-1.)/2.
    step1 = alpha_u1[0, 1]-alpha_u1[0, 0]
    Npix2 = alpha_u2.shape[0]
    imhs2 = (Npix2-1.)/2.
    step2 = alpha_u2[0, 1]-alpha_u2[0, 0]
    
    xticklabels1 = np.linspace(np.min(alpha_u1), np.max(alpha_u1), 11)
    xticks1 = xticklabels1/step1+imhs1
    xticklabels2 = np.linspace(np.min(alpha_u2), np.max(alpha_u2), 11)
    xticks2 = xticklabels2/step2+imhs2
    
    f, axarr = plt.subplots(1, 2, figsize=(6.4*2, 4.8*1))
    p0 = axarr[0].imshow(snr1, cmap='hot', vmin=0, zorder=0, origin='lower')
    c0 = plt.colorbar(p0, ax=axarr[0])
    c0.set_label('SNR', rotation=270, labelpad=10)
    if (alpha1_in is not None):
        cc = plt.Rectangle((imhs1+alpha1_in[0]/step1-Npix1/10., imhs1+alpha1_in[1]/step1-Npix1/10.), Npix1/5., Npix1/5., color='black', lw=5, fill=False, zorder=1)
        axarr[0].add_artist(cc)
        cc = plt.Rectangle((imhs1+alpha1_in[0]/step1-Npix1/10., imhs1+alpha1_in[1]/step1-Npix1/10.), Npix1/5., Npix1/5., color='cyan', lw=2.5, fill=False, zorder=1)
        axarr[0].add_artist(cc)    
    cc = plt.Circle((imhs1+alpha1[0]/step1, imhs1+alpha1[1]/step1), Npix1/10., color='black', lw=5, fill=False, zorder=1)
    axarr[0].add_artist(cc)
    cc = plt.Circle((imhs1+alpha1[0]/step1, imhs1+alpha1[1]/step1), Npix1/10., color='cyan', lw=2.5, fill=False, zorder=1)
    axarr[0].add_artist(cc)
    text = axarr[0].text(0.05, 0.95, '$f_{fit}$ = %.5f' % f1+', $\Delta$RA = %.1f' % alpha1[0]+', $\Delta$DEC = %.1f' % alpha1[1], ha='left', va='top', transform=axarr[0].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[0].text(0.05, 0.05, '$\chi^2$ = %.3f (best)' % red_chi21, ha='left', va='bottom', transform=axarr[0].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[0].text(0.05, 0.125, 'SNR = %.1f' % snr1[int(imhs1+alpha1[1]/step1), int(imhs1+alpha1[0]/step1)], ha='left', va='bottom', transform=axarr[0].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0].set_xticks(xticks1)
    axarr[0].set_xticklabels(['%.0f' % label for label in xticklabels1])
    axarr[0].set_xlabel('$\Delta$RA [$mas$]')
    axarr[0].set_yticks(xticks1)
    axarr[0].set_yticklabels(['%.0f' % label for label in xticklabels1])
    axarr[0].set_ylabel('$\Delta$DEC [$mas$]')
    
    p1 = axarr[1].imshow(snr2, cmap='hot', vmin=0, zorder=0, origin='lower')
    c1 = plt.colorbar(p1, ax=axarr[1])
    c1.set_label('SNR', rotation=270, labelpad=10)
    if (alpha2_in is not None):
        cc = plt.Rectangle((imhs2+alpha2_in[0]/step2-Npix2/10., imhs2+alpha2_in[1]/step2-Npix2/10.), Npix2/5., Npix2/5., color='black', lw=5, fill=False, zorder=1)
        axarr[1].add_artist(cc)
        cc = plt.Rectangle((imhs2+alpha2_in[0]/step2-Npix2/10., imhs2+alpha2_in[1]/step2-Npix2/10.), Npix2/5., Npix2/5., color='cyan', lw=2.5, fill=False, zorder=1)
        axarr[1].add_artist(cc)
    cc = plt.Circle((imhs2+alpha2[0]/step2, imhs2+alpha2[1]/step2), Npix2/10., color='black', lw=5, fill=False, zorder=1)
    axarr[1].add_artist(cc)
    cc = plt.Circle((imhs2+alpha2[0]/step2, imhs2+alpha2[1]/step2), Npix2/10., color='cyan', lw=2.5, fill=False, zorder=1)
    axarr[1].add_artist(cc)
    text = axarr[1].text(0.05, 0.95, '$f_{fit}$ = %.5f' % f2+', $\Delta$RA = %.1f' % alpha2[0]+', $\Delta$DEC = %.1f' % alpha2[1], ha='left', va='top', transform=axarr[1].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[1].text(0.05, 0.05, '$\chi^2$ = %.3f (best)' % red_chi22, ha='left', va='bottom', transform=axarr[1].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[1].text(0.05, 0.125, 'SNR = %.1f' % snr2[int(imhs2+alpha2[1]/step2), int(imhs2+alpha2[0]/step2)], ha='left', va='bottom', transform=axarr[1].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])    
    axarr[1].set_xticks(xticks2)
    axarr[1].set_xticklabels(['%.0f' % label for label in xticklabels2])
    axarr[1].set_xlabel('$\Delta$RA [$mas$]')
    axarr[1].set_yticks(xticks2)
    axarr[1].set_yticklabels(['%.0f' % label for label in xticklabels2])
    axarr[1].set_ylabel('$\Delta$DEC [$mas$]')
    
    plt.tight_layout()
#    plt.savefig(filename+'.pdf')
    plt.show(block=True)
    plt.close()
    
    pass

def snr2_azavg(f1,
               alpha1,
               red_chi21,
               fs1,
               dfs1,
               alpha_u1,
               alpha_v1,
               f2,
               alpha2,
               red_chi22,
               fs2,
               dfs2,
               alpha_u2,
               alpha_v2,
               f1_in=None,
               alpha1_in=None,
               f2_in=None,
               alpha2_in=None,
               filename='snr2'):
    """
    """
    
    snr1 = np.true_divide(fs1, dfs1)
    dist = np.sqrt(alpha_u1**2+alpha_v1**2)
    snr1[dist > np.max(alpha_u1)] = 0.
    snr2 = np.true_divide(fs2, dfs2)
    dist = np.sqrt(alpha_u2**2+alpha_v2**2)
    snr2[dist > np.max(alpha_u2)] = 0.
    
    Npix1 = alpha_u1.shape[0]
    imhs1 = (Npix1-1.)/2.
    step1 = alpha_u1[0, 1]-alpha_u1[0, 0]
    Npix2 = alpha_u2.shape[0]
    imhs2 = (Npix2-1.)/2.
    step2 = alpha_u2[0, 1]-alpha_u2[0, 0]
    
    xticklabels1 = np.linspace(np.min(alpha_u1), np.max(alpha_u1), 11)
    xticks1 = xticklabels1/step1+imhs1
    xticklabels2 = np.linspace(np.min(alpha_u2), np.max(alpha_u2), 11)
    xticks2 = xticklabels2/step2+imhs2
    
    f, axarr = plt.subplots(2, 2, figsize=(6.4*2, 4.8*2))
    p00 = axarr[0, 0].imshow(snr1, cmap='hot', vmin=0, zorder=0, origin='lower')
    c00 = plt.colorbar(p00, ax=axarr[0, 0])
    c00.set_label('SNR', rotation=270, labelpad=10)
    if (alpha1_in is not None):
        cc = plt.Rectangle((imhs1+alpha1_in[0]/step1-Npix1/10., imhs1+alpha1_in[1]/step1-Npix1/10.), Npix1/5., Npix1/5., color='black', lw=5, fill=False, zorder=1)
        axarr[0, 0].add_artist(cc)
        cc = plt.Rectangle((imhs1+alpha1_in[0]/step1-Npix1/10., imhs1+alpha1_in[1]/step1-Npix1/10.), Npix1/5., Npix1/5., color='cyan', lw=2.5, fill=False, zorder=1)
        axarr[0, 0].add_artist(cc)    
    cc = plt.Circle((imhs1+alpha1[0]/step1, imhs1+alpha1[1]/step1), Npix1/10., color='black', lw=5, fill=False, zorder=1)
    axarr[0, 0].add_artist(cc)
    cc = plt.Circle((imhs1+alpha1[0]/step1, imhs1+alpha1[1]/step1), Npix1/10., color='cyan', lw=2.5, fill=False, zorder=1)
    axarr[0, 0].add_artist(cc)
    text = axarr[0, 0].text(0.05, 0.95, '$f_{fit}$ = %.5f' % f1+', $\Delta$RA = %.1f' % alpha1[0]+', $\Delta$DEC = %.1f' % alpha1[1], ha='left', va='top', transform=axarr[0, 0].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[0, 0].text(0.05, 0.05, '$\chi^2$ = %.3f (best)' % red_chi21, ha='left', va='bottom', transform=axarr[0, 0].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[0, 0].text(0.05, 0.125, 'SNR = %.1f' % snr1[int(imhs1+alpha1[1]/step1), int(imhs1+alpha1[0]/step1)], ha='left', va='bottom', transform=axarr[0, 0].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    axarr[0, 0].set_xticks(xticks1)
    axarr[0, 0].set_xticklabels(['%.0f' % label for label in xticklabels1])
    axarr[0, 0].set_xlabel('$\Delta$RA [$mas$]')
    axarr[0, 0].set_yticks(xticks1)
    axarr[0, 0].set_yticklabels(['%.0f' % label for label in xticklabels1])
    axarr[0, 0].set_ylabel('$\Delta$DEC [$mas$]')
    
    p01 = axarr[0, 1].imshow(snr2, cmap='hot', vmin=0, zorder=0, origin='lower')
    c01 = plt.colorbar(p01, ax=axarr[0, 1])
    c01.set_label('SNR', rotation=270, labelpad=10)
    if (alpha2_in is not None):
        cc = plt.Rectangle((imhs2+alpha2_in[0]/step2-Npix2/10., imhs2+alpha2_in[1]/step2-Npix2/10.), Npix2/5., Npix2/5., color='black', lw=5, fill=False, zorder=1)
        axarr[0, 1].add_artist(cc)
        cc = plt.Rectangle((imhs2+alpha2_in[0]/step2-Npix2/10., imhs2+alpha2_in[1]/step2-Npix2/10.), Npix2/5., Npix2/5., color='cyan', lw=2.5, fill=False, zorder=1)
        axarr[0, 1].add_artist(cc)
    cc = plt.Circle((imhs2+alpha2[0]/step2, imhs2+alpha2[1]/step2), Npix2/10., color='black', lw=5, fill=False, zorder=1)
    axarr[0, 1].add_artist(cc)
    cc = plt.Circle((imhs2+alpha2[0]/step2, imhs2+alpha2[1]/step2), Npix2/10., color='cyan', lw=2.5, fill=False, zorder=1)
    axarr[0, 1].add_artist(cc)
    text = axarr[0, 1].text(0.05, 0.95, '$f_{fit}$ = %.5f' % f2+', $\Delta$RA = %.1f' % alpha2[0]+', $\Delta$DEC = %.1f' % alpha2[1], ha='left', va='top', transform=axarr[0, 1].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[0, 1].text(0.05, 0.05, '$\chi^2$ = %.3f (best)' % red_chi22, ha='left', va='bottom', transform=axarr[0, 1].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = axarr[0, 1].text(0.05, 0.125, 'SNR = %.1f' % snr2[int(imhs2+alpha2[1]/step2), int(imhs2+alpha2[0]/step2)], ha='left', va='bottom', transform=axarr[0, 1].transAxes, zorder=2)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])    
    axarr[0, 1].set_xticks(xticks2)
    axarr[0, 1].set_xticklabels(['%.0f' % label for label in xticklabels2])
    axarr[0, 1].set_xlabel('$\Delta$RA [$mas$]')
    axarr[0, 1].set_yticks(xticks2)
    axarr[0, 1].set_yticklabels(['%.0f' % label for label in xticklabels2])
    axarr[0, 1].set_ylabel('$\Delta$DEC [$mas$]')
    
    rad, avg = azavg(fs1)
    axarr[1, 0].plot(rad, avg, label='diag')
    rad, avg = azavg(fs2)
    axarr[1, 0].plot(rad, avg, label='full')
    axarr[1, 0].set_yscale('log')
    axarr[1, 0].grid(which='both', axis='y')
    axarr[1, 0].set_xlabel('Angular separation [mas]')
    axarr[1, 0].set_ylabel('Contrast')
    axarr[1, 0].legend()
    
    plt.tight_layout()
#    plt.savefig(filename+'.pdf')
    plt.show(block=True)
    plt.close()
    
    import pdb; pdb.set_trace()
    
    pass
