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

from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.special import j1
import scipy.stats as stats

# Set interactive plots to off and change default plot font size.
plt.ioff()
plt.rcParams.update({'font.size': 12})


"""
Functions.
"""
def vis2vis2(vis):
    
    vis2 = np.abs(vis)**2
    
    return vis2

def vis2t3(vis,
           cp_mat):
    
    Nobs = vis.shape[0]
    
    t3 = []
    for i in range(Nobs):
        t3 += [cp_mat.dot(np.angle(vis[i]))]
    t3 = np.array(t3)
    
    t3 = ((t3+np.pi) % (2.*np.pi))-np.pi
    
    return t3

def mod_ud(p0,
           uu,
           vv):
    
    mas2rad = 1./1000./60./60./180.*np.pi
    
    x = np.pi*p0[0]*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    vis = 2.*j1(x)/x
    
    vis2 = vis2vis2(vis)
    t3 = vis2t3(vis, cp_mat)
    
    return vis2, t3

def mod_ud_bin(p0,
               uu,
               vv):
    
    mas2rad = 1./1000./60./60./180.*np.pi
    
    x = np.pi*p0[3]*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    V1 = 2.*j1(x)/x
    x = np.pi*0.*mas2rad*np.sqrt(uu**2+vv**2)
    x += 1e-6*(x == 0)
    V2 = 2.*j1(x)/x
    temp = V2*p0[0]*np.exp(-2.*np.pi*1j*(-uu*p0[1]*mas2rad+vv*p0[2]*mas2rad))
    vis = (V1+temp)/(1.+p0[0])
    
    vis2 = vis2vis2(vis)
    t3 = vis2t3(vis, cp_mat)
    
    return vis2, t3

def chi2_ud(p0,
            vis2,
            t3,
            icv,
            uu,
            vv,
            cp_mat,
            diag):
    
    Nobs = vis2.shape[0]
    
    vis2_mod, t3_mod = mod_ud(p0, uu, vv)
    sig = np.concatenate((vis2, t3), axis=1)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    res = sig-mod
    
    chi2 = []
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (diag == True):
            chi2 += [np.multiply(res_temp, icv).dot(res_temp)]
        else:
            chi2 += [res_temp.dot(icv).dot(res_temp)]
    
    return np.sqrt(np.array(chi2))

def chi2_ud_bin(p0,
                vis2,
                t3,
                icv,
                uu,
                vv,
                cp_mat,
                diag):
    
    Nobs = vis2.shape[0]
    
    vis2_mod, t3_mod = mod_ud_bin(p0, uu, vv)
    sig = np.concatenate((vis2, t3), axis=1)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    res = sig-mod
    
    chi2 = []
    for i in range(Nobs):
        res_temp = res[i].flatten()
        if (diag == True):
            chi2 += [np.multiply(res_temp, icv).dot(res_temp)]
        else:
            chi2 += [res_temp.dot(icv).dot(res_temp)]
    
    return np.sqrt(np.array(chi2))

def fit_ud(p0,
           vis2,
           t3,
           icv,
           uu,
           vv,
           cp_mat,
           diag):
    
    pp = leastsq(chi2_ud,
                 p0,
                 args=(vis2, t3, icv, uu, vv, cp_mat, diag),
                 full_output=True,
                 epsfcn=1e-8,
                 ftol=1e-5,
                 maxfev=1000)
    
    Ndof = np.prod(vis2.shape)+np.prod(t3.shape)
    chi2 = np.sum(pp[2]['fvec']**2)/Ndof
    
    return chi2, Ndof, pp[0]

def fit_ud_bin(p0,
               vis2,
               t3,
               icv,
               uu,
               vv,
               cp_mat,
               diag):
    
    pp = leastsq(chi2_ud_bin,
                 p0,
                 args=(vis2, t3, icv, uu, vv, cp_mat, diag),
                 full_output=True,
                 epsfcn=1e-8,
                 ftol=1e-5,
                 maxfev=1000)
    
    Ndof = np.prod(vis2.shape)+np.prod(t3.shape)
    chi2 = np.sum(pp[2]['fvec']**2)/Ndof
    
    return chi2, Ndof, pp[0]

def linearize(fun,
              p0,
              uu,
              vv):
    """
    Let's assume the parameter vector p0 = (1, f, dra, ddec, theta). Then, the
    design matix X has shape N times 5, where N is the number of observabels.
    """
    vis2_mod, t3_mod = mod_ud_bin(p0, uu, vv)
    mod = np.concatenate((vis2_mod, t3_mod), axis=1)
    
    X = np.zeros((mod.shape[1]*mod.shape[2], p0.shape[0]+1))
    
    # 1 (constant)
    X[:, 0] = mod[0].flatten()
    
    # f
    p0_temp = p0.copy()
    p0_temp[0] += 1e-5
    vis2_mod, t3_mod = mod_ud_bin(p0_temp, uu, vv)
    mod_temp = np.concatenate((vis2_mod, t3_mod), axis=1)
    X[:, 1] = (mod_temp[0].flatten()-X[:, 0])/1e-5
    
    # dra
    p0_temp = p0.copy()
    p0_temp[1] += 1e-2
    vis2_mod, t3_mod = mod_ud_bin(p0_temp, uu, vv)
    mod_temp = np.concatenate((vis2_mod, t3_mod), axis=1)
    X[:, 2] = (mod_temp[0].flatten()-X[:, 0])/1e-2
    
    # ddec
    p0_temp = p0.copy()
    p0_temp[2] += 1e-2
    vis2_mod, t3_mod = mod_ud_bin(p0_temp, uu, vv)
    mod_temp = np.concatenate((vis2_mod, t3_mod), axis=1)
    X[:, 3] = (mod_temp[0].flatten()-X[:, 0])/1e-2
    
    # theta
    p0_temp = p0.copy()
    p0_temp[3] += 1e-3
    vis2_mod, t3_mod = mod_ud_bin(p0_temp, uu, vv)
    mod_temp = np.concatenate((vis2_mod, t3_mod), axis=1)
    X[:, 4] = (mod_temp[0].flatten()-X[:, 0])/1e-3
    
    return X

def Nsigma(chi2r_test,
           chi2r_true,
           Ndof):
    
    p = stats.chi2.cdf(Ndof, Ndof*chi2r_test/chi2r_true)
    log10p = np.log10(np.maximum(p, 1e-161)) # 50 sigma max.
    Nsigma = np.sqrt(stats.chi2.ppf(1.-p, 1.))
    
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


"""
Reading.
"""
#idir = 'fitcube_sim/'
idir = 'fitcube_real/'
fitsfiles = [f for f in os.listdir(idir) if f.startswith('diag') and f.endswith('.fits')]
fitsfiles = np.sort(fitsfiles)
Nf = len(fitsfiles)

#chi2s_ud = []
#chi2s_ud_bin = []
#sigs = []
#for i in range(Nf):
#    sys.stdout.write('\rFile %.0f of %.0f' % (i+1, Nf))
#    sys.stdout.flush()
#    
#    vis2 = pyfits.getdata(idir+fitsfiles[i], 'vis2')
#    vis2cov = pyfits.getdata(idir+fitsfiles[i], 'vis2cov')
#    t3 = pyfits.getdata(idir+fitsfiles[i], 't3')
#    t3cov = pyfits.getdata(idir+fitsfiles[i], 't3cov')
#    uu = pyfits.getdata(idir+fitsfiles[i], 'uv')[0]
#    vv = pyfits.getdata(idir+fitsfiles[i], 'uv')[1]
#    cp_mat = pyfits.getdata(idir+fitsfiles[i], 'cp_mat')
#    pps = pyfits.getdata(idir+fitsfiles[i], 'pps')
#    diag = True
#    
#    if (diag == True):
#        icv = np.hstack((1./vis2cov, 1./t3cov))
#    else:
#        vis2icv = np.linalg.inv(vis2cov)
#        t3icv = np.linalg.inv(t3cov)
#        icv = block_diag(vis2icv, t3icv)
#    
#    p0 = np.array([1.])
#    chi2_red_ud, Ndof_ud, pp_ud = fit_ud(p0, vis2, t3, icv, uu, vv, cp_mat, diag)
#    
#    ww = np.argmin(pps[:, -1])
#    p0 = pps[ww][0:4]
#    chi2_red_ud_bin = np.sum(chi2_ud_bin(p0, vis2, t3, icv, uu, vv, cp_mat, diag)**2)/Ndof_ud
#    if (chi2_red_ud_bin-pps[ww][-1] > 1e-10):
#        import pdb; pdb.set_trace()
#    
#    ###
#    X = linearize(mod_ud_bin, p0, uu, vv)
#    Sigma_inv = np.diag(icv)
#    
#    import pdb; pdb.set_trace()
#    temp = np.linalg.inv(X.T.dot(Sigma_inv).dot(X))
#    H = X.dot(temp).dot(X.T).dot(Sigma_inv)
#    temp = np.eye(H.shape[0])-H
#    Ndof = np.trace(temp.T.dot(temp))
#    import pdb; pdb.set_trace()
#    ###
#    
#    sig = Nsigma(chi2_red_ud,
#                 chi2_red_ud_bin,
#                 Ndof_ud)
#    
#    chi2s_ud += [chi2_red_ud]
#    chi2s_ud_bin += [chi2_red_ud_bin]
#    sigs += [sig]
#print('')
#chi2s_ud = np.array(chi2s_ud)
#np.save('chi2s_ud_diag', chi2s_ud)
#chi2s_ud_bin = np.array(chi2s_ud_bin)
#np.save('chi2s_ud_bin_diag', chi2s_ud_bin)
#sigs = np.array(sigs)
#np.save('sigs_diag', sigs)
#import pdb; pdb.set_trace()

p0s_diag = []
chi2s_ud_diag = []
chi2s_ud_bin_diag = []
sigs_diag = []
dist_diag = []
cons_diag = []
for i in range(Nf):
    try:
        sys.stdout.write('\rFile %.0f of %.0f' % (i+1, Nf))
        sys.stdout.flush()
        hdul = pyfits.open(idir+fitsfiles[i])
        
        p0s_diag += [hdul['p0'].data.copy()]
        pps = hdul['pps'].data
        chi2s_ud_diag += [hdul['sig'].data.copy()[0]]
        sigs_diag += [hdul['sig'].data.copy()[1]]
        
        hdul.close()
        
        ww = np.argmin(pps[:, -1])
        chi2s_ud_bin_diag += [pps[ww, -1]]
        dist_diag += [np.array([pps[ww, 1]-p0s_diag[-1][1], pps[ww, 2]-p0s_diag[-1][2]])]
        cons_diag += [(pps[ww, 0]-p0s_diag[-1][0])/p0s_diag[-1][0]]
    except:
        continue
print('')
p0s_diag = np.array(p0s_diag)
chi2s_ud_diag = np.array(chi2s_ud_diag)
chi2s_ud_bin_diag = np.array(chi2s_ud_bin_diag)
sigs_diag = np.array(sigs_diag)
dist_diag = np.array(dist_diag)
cons_diag = np.array(cons_diag)

#idir = 'fitcube_sim/'
idir = 'fitcube_real/'
fitsfiles = [f for f in os.listdir(idir) if f.startswith('full') and f.endswith('.fits')]
fitsfiles = np.sort(fitsfiles)
Nf = len(fitsfiles)

#chi2s_ud = []
#chi2s_ud_bin = []
#sigs = []
#for i in range(Nf):
#    sys.stdout.write('\rFile %.0f of %.0f' % (i+1, Nf))
#    sys.stdout.flush()
#    
#    vis2 = pyfits.getdata(idir+fitsfiles[i], 'vis2')
#    vis2cov = pyfits.getdata(idir+fitsfiles[i], 'vis2cov')
#    t3 = pyfits.getdata(idir+fitsfiles[i], 't3')
#    t3cov = pyfits.getdata(idir+fitsfiles[i], 't3cov')
#    uu = pyfits.getdata(idir+fitsfiles[i], 'uv')[0]
#    vv = pyfits.getdata(idir+fitsfiles[i], 'uv')[1]
#    cp_mat = pyfits.getdata(idir+fitsfiles[i], 'cp_mat')
#    pps = pyfits.getdata(idir+fitsfiles[i], 'pps')
#    diag = False
#    
#    if (diag == True):
#        icv = np.hstack((1./vis2cov, 1./t3cov))
#    else:
#        vis2icv = np.linalg.inv(vis2cov)
#        t3icv = np.linalg.inv(t3cov)
#        icv = block_diag(vis2icv, t3icv)
#    
#    p0 = np.array([1.])
#    chi2_red_ud, Ndof_ud, pp_ud = fit_ud(p0, vis2, t3, icv, uu, vv, cp_mat, diag)
#    
#    ww = np.argmin(pps[:, -1])
#    p0 = pps[ww][0:4]
#    chi2_red_ud_bin = np.sum(chi2_ud_bin(p0, vis2, t3, icv, uu, vv, cp_mat, diag)**2)/Ndof_ud
#    if (chi2_red_ud_bin-pps[ww][-1] > 1e-10):
#        import pdb; pdb.set_trace()
#    
#    ###
#    X = linearize(mod_ud_bin, p0, uu, vv)
#    Sigma_inv = icv
#    
#    import pdb; pdb.set_trace()
#    temp = np.linalg.inv(X.T.dot(Sigma_inv).dot(X))
#    H = X.dot(temp).dot(X.T).dot(Sigma_inv)
#    temp = np.eye(H.shape[0])-H
#    Ndof = np.trace(temp.T.dot(temp))
#    import pdb; pdb.set_trace()
#    ###
#    
#    sig = Nsigma(chi2_red_ud,
#                 chi2_red_ud_bin,
#                 Ndof_ud)
#    
#    chi2s_ud += [chi2_red_ud]
#    chi2s_ud_bin += [chi2_red_ud_bin]
#    sigs += [sig]
#print('')
#chi2s_ud = np.array(chi2s_ud)
#np.save('chi2s_ud_full', chi2s_ud)
#chi2s_ud_bin = np.array(chi2s_ud_bin)
#np.save('chi2s_ud_bin_full', chi2s_ud_bin)
#sigs = np.array(sigs)
#np.save('sigs_full', sigs)
#import pdb; pdb.set_trace()

p0s_full = []
chi2s_ud_full = []
chi2s_ud_bin_full = []
sigs_full = []
dist_full = []
cons_full = []
for i in range(Nf):
    try:
        sys.stdout.write('\rFile %.0f of %.0f' % (i+1, Nf))
        sys.stdout.flush()
        hdul = pyfits.open(idir+fitsfiles[i])
        
        p0s_full += [hdul['p0'].data.copy()]
        pps = hdul['pps'].data
        chi2s_ud_full += [hdul['sig'].data.copy()[0]]
        sigs_full += [hdul['sig'].data.copy()[1]]
        
        hdul.close()
        
        ww = np.argmin(pps[:, -1])
        chi2s_ud_bin_full += [pps[ww, -1]]
        dist_full += [np.array([pps[ww, 1]-p0s_full[-1][1], pps[ww, 2]-p0s_full[-1][2]])]
        cons_full += [(pps[ww, 0]-p0s_full[-1][0])/p0s_full[-1][0]]
    except:
        continue
print('')
p0s_full = np.array(p0s_full)
chi2s_ud_full = np.array(chi2s_ud_full)
chi2s_ud_bin_full = np.array(chi2s_ud_bin_full)
sigs_full = np.array(sigs_full)
dist_full = np.array(dist_full)
cons_full = np.array(cons_full)

#chi2s_ud_diag = np.load('chi2s_ud_diag.npy')
#chi2s_ud_full = np.load('chi2s_ud_full.npy')
#chi2s_ud_bin_diag = np.load('chi2s_ud_bin_diag.npy')
#chi2s_ud_bin_full = np.load('chi2s_ud_bin_full.npy')
#sigs_diag = np.load('sigs_diag.npy')
#sigs_full = np.load('sigs_full.npy')


"""
Analysis.
"""

#cons = np.logspace(-4, -1.5, 11)
cons = np.logspace(-3, -0.5, 11)
Ncons = len(cons)
seps = np.linspace(4.99, 44.99, 9)
Nseps = len(seps)

res = 2.2e-6/(2.*130.)*180./np.pi*3600.*1000.
sig_threshold = 3.

dets_dist_diag = np.sqrt(np.sum(dist_diag**2, axis=1)) < res
dets_dist_full = np.sqrt(np.sum(dist_full**2, axis=1)) < res
dets_cons_diag = (cons_diag > -0.1) & (cons_diag < 0.1)
dets_cons_full = (cons_full > -0.1) & (cons_full < 0.1)
dets_diag = dets_dist_diag & dets_cons_diag
dets_full = dets_dist_full & dets_cons_full
    
dd_diag = np.sqrt(p0s_diag[:, 1]**2+p0s_diag[:, 2]**2)
dd_full = np.sqrt(p0s_full[:, 1]**2+p0s_full[:, 2]**2)

mask_diag = (dd_diag > seps[0]) & (dd_diag < seps[-1]) & (p0s_diag[:, 0] > cons[0]-1e-5) & (p0s_diag[:, 0] < cons[-1]+1e-5)
mask_full = (dd_full > seps[0]) & (dd_full < seps[-1]) & (p0s_full[:, 0] > cons[0]-1e-5) & (p0s_full[:, 0] < cons[-1]+1e-5)
dets_diag_mask = dets_diag[mask_diag]
dets_full_mask = dets_full[mask_full]
sigs_diag_mask = sigs_diag[mask_diag] > sig_threshold
sigs_full_mask = sigs_full[mask_full] > sig_threshold

conf_diag = np.zeros((2, 2))
conf_full = np.zeros((2, 2))
conf_diag[0, 0] = np.sum(dets_diag_mask & sigs_diag_mask)
conf_diag[0, 1] = np.sum(dets_diag_mask & np.logical_not(sigs_diag_mask))
conf_diag[1, 0] = np.sum(np.logical_not(dets_diag_mask) & sigs_diag_mask)
conf_diag[1, 1] = np.sum(np.logical_not(dets_diag_mask) & np.logical_not(sigs_diag_mask))
conf_full[0, 0] = np.sum(dets_full_mask & sigs_full_mask)
conf_full[0, 1] = np.sum(dets_full_mask & np.logical_not(sigs_full_mask))
conf_full[1, 0] = np.sum(np.logical_not(dets_full_mask) & sigs_full_mask)
conf_full[1, 1] = np.sum(np.logical_not(dets_full_mask) & np.logical_not(sigs_full_mask))

Ngood = np.zeros((Nseps-1, Ncons, 2))
Nfits = np.zeros((Nseps-1, Ncons, 2))
for i in range(Nseps-1):
    ww_seps = (dd_diag > seps[i]) & (dd_diag < seps[i+1])
    for j in range(Ncons):
        ww_cons = (p0s_diag[:, 0] > cons[j]-1e-5) & (p0s_diag[:, 0] < cons[j]+1e-5)
        Ngood[i, j, 0] = np.sum(dets_diag & ww_seps & ww_cons)
        Ngood[i, j, 1] = np.sum(dets_full & ww_seps & ww_cons)
        Nfits[i, j, 0] = np.sum(ww_seps & ww_cons)
        Nfits[i, j, 1] = np.sum(ww_seps & ww_cons)

yy_diag = np.sum(Ngood[:, :, 0], axis=0)/np.sum(Nfits[:, :, 0], axis=0)
yy_full = np.sum(Ngood[:, :, 1], axis=0)/np.sum(Nfits[:, :, 1], axis=0)
def logistic(x, k, x0):
    return 1./(1.+np.exp(-k*(x-x0)))
def logistic_inv(y, k, x0):
    return x0-np.log(1./y-1.)/k
def fit(params, x, y):
    return y-logistic(np.log10(x), params[0], params[1])
params = np.array([1., 1e-3])
pp_diag = leastsq(fit, params, args=(cons, yy_diag))
pp_full = leastsq(fit, params, args=(cons, yy_full))

fracs = np.linspace(0.01, 0.99, 100)
ww_diag = 10**logistic_inv(fracs, *pp_diag[0])
ww_full = 10**logistic_inv(fracs, *pp_full[0])

import pdb; pdb.set_trace()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#xx = np.logspace(-4, -1.5, 100)
xx = np.logspace(-3, -0.5, 100)
plt.figure()
ax1 = plt.gca()
l1 = ax1.plot(cons, yy_diag, color=colors[0], ls='None', marker='o', label='uncorrelated errors')
ax1.plot(xx, logistic(np.log10(xx), *pp_diag[0]), color=colors[0])
l2 = ax1.plot(cons, yy_full, color=colors[1], ls='None', marker='o', label='correlated errors')
ax1.plot(xx, logistic(np.log10(xx), *pp_full[0]), color=colors[1])
ax1.set_xscale('log')
#temp = (-1.5+4.)*0.1196821457656456/2.
temp = (-0.5+3)*0.1196821457656456/2.
#ax1.set_xlim([10**(-4.-temp), 10**(-1.5+temp)])
ax1.set_xlim([10**(-3.-temp), 10**(-0.5+temp)])
ax1.set_xlabel('Companion contrast')
temp = (1.-0.)*0.1196821457656456/2.
ax1.set_ylim([0.-temp, 1.+temp])
ax1.set_ylabel('Fraction of detections')
ax1.grid()
ax2 = ax1.twiny()
l3 = ax2.plot(ww_diag/ww_full, fracs, color='black', ls='--', label='ratio')
temp = (5.-1.)*0.1196821457656456/2.
ax2.set_xlim([1.-temp, 5.+temp])
ax2.set_xlabel('Ratio')
ls = l1+l2+l3
ls = l1+l2
la = [l.get_label() for l in ls]
#ax1.legend(ls, la, loc='upper left')
plt.tight_layout()
#plt.savefig('injection_recovery_sim.pdf')
plt.savefig('injection_recovery_real.pdf')
plt.show(block=True)

import pdb; pdb.set_trace()
