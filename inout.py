import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

def load_oifits(idir,
                fitsfiles,
                insname,
                Nbase=None,
                Ntria=None,
                check=True,
                visamp=False):
    """
    """
    
    wavel = []
    dwavel = []
    vis2 = []
    vis2_err = []
    vis2_u = []
    vis2_v = []
    vis2_sta = []
    t3 = []
    t3_err = []
    t3_sta = []
    for i in range(len(fitsfiles)):
        hdul = pyfits.open(idir+fitsfiles[i])
        for j in range(len(hdul)):
            try:
                if (hdul[j].header['INSNAME'] == insname):
                    if (hdul[j].header['EXTNAME'] == 'OI_WAVELENGTH'):
                        wavel += [hdul[j].data['EFF_WAVE']]
                        dwavel += [hdul[j].data['EFF_BAND']]
                    if (visamp == False):
                        if (hdul[j].header['EXTNAME'] == 'OI_VIS2'):
                            vis2 += [hdul[j].data['VIS2DATA']]
                            vis2[-1][hdul[j].data['FLAG']] = np.nan
                            vis2_err += [hdul[j].data['VIS2ERR']]
                            vis2_err[-1][hdul[j].data['FLAG']] = np.nan
                            vis2_u += [hdul[j].data['UCOORD']]
                            vis2_v += [hdul[j].data['VCOORD']]
                            vis2_sta += [hdul[j].data['STA_INDEX']]
                    else:
                        if (hdul[j].header['EXTNAME'] == 'OI_VIS'):
                            vis2 += [hdul[j].data['VISAMP']]
                            vis2[-1][hdul[j].data['FLAG']] = np.nan
                            vis2_err += [hdul[j].data['VISAMPERR']]
                            vis2_err[-1][hdul[j].data['FLAG']] = np.nan
                            vis2_u += [hdul[j].data['UCOORD']]
                            vis2_v += [hdul[j].data['VCOORD']]
                            vis2_sta += [hdul[j].data['STA_INDEX']]
                    if (hdul[j].header['EXTNAME'] == 'OI_T3'):
                        t3 += [np.radians(hdul[j].data['T3PHI'])]
                        t3[-1][hdul[j].data['FLAG']] = np.nan
                        t3_err += [np.radians(hdul[j].data['T3PHIERR'])]
                        t3_err[-1][hdul[j].data['FLAG']] = np.nan
                        t3_sta += [hdul[j].data['STA_INDEX']]
            except:
                pass
        hdul.close()
    if (Nbase is not None):
        Nbase = int(Nbase)
        vis2 = np.concatenate(vis2)
        vis2 = vis2.reshape((vis2.shape[0]/Nbase, Nbase, vis2.shape[1]))
        vis2_err = np.concatenate(vis2_err)
        vis2_err = vis2_err.reshape((vis2_err.shape[0]/Nbase, Nbase, vis2_err.shape[1]))
        vis2_u = np.concatenate(vis2_u)
        vis2_u = vis2_u.reshape((vis2_u.shape[0]/Nbase, Nbase))
        vis2_v = np.concatenate(vis2_v)
        vis2_v = vis2_v.reshape((vis2_v.shape[0]/Nbase, Nbase))
        vis2_sta = np.concatenate(vis2_sta)
        vis2_sta = vis2_sta.reshape((vis2_sta.shape[0]/Nbase, Nbase, vis2_sta.shape[1]))
    if (Ntria is not None):
        Ntria = int(Ntria)
        t3 = np.concatenate(t3)
        t3 = t3.reshape((t3.shape[0]/Ntria, Ntria, t3.shape[1]))
        t3_err = np.concatenate(t3_err)
        t3_err = t3_err.reshape((t3_err.shape[0]/Ntria, Ntria, t3_err.shape[1]))
        t3_sta = np.concatenate(t3_sta)
        t3_sta = t3_sta.reshape((t3_sta.shape[0]/Ntria, Ntria, t3_sta.shape[1]))
    wavel = np.array(wavel)
    dwavel = np.array(dwavel)
    vis2 = np.array(vis2)
    vis2_err = np.array(vis2_err)
    vis2_u = np.array(vis2_u)
    vis2_v = np.array(vis2_v)
    vis2_sta = np.array(vis2_sta)
    t3 = np.array(t3)
    t3_err = np.array(t3_err)
    t3_sta = np.array(t3_sta)
    if (wavel.shape[0] != vis2.shape[0]):
        wavel = np.repeat(wavel, vis2.shape[0], axis=0)
        dwavel = np.repeat(dwavel, vis2.shape[0], axis=0)
    
    if (check == True):
        sta = vis2_sta[0]
        for i in range(1, vis2_sta.shape[0]):
            if (np.array_equal(sta, vis2_sta[i]) == False):
                imap = np.zeros(sta.shape[0])
                for j in range(len(imap)):
                    imap[j] = np.bincount(np.where(vis2_sta[i] == sta[j])[0]).argmax()
                vis2[i, :, :] = vis2[i, imap.astype(int), :]
                vis2_err[i, :, :] = vis2_err[i, imap.astype(int), :]
                vis2_u[i, :] = vis2_u[i, imap.astype(int)]
                vis2_v[i, :] = vis2_v[i, imap.astype(int)]
                vis2_sta[i, :, :] = vis2_sta[i, imap.astype(int), :]
        sta = t3_sta[0]
        for i in range(1, t3_sta.shape[0]):
            if (np.array_equal(sta, t3_sta[i]) == False):
                imap = np.zeros(sta.shape[0])
                for j in range(len(imap)):
                    imap[j] = np.bincount(np.where(t3_sta[i] == sta[j])[0]).argmax()
                t3[i, :, :] = t3[i, imap.astype(int), :]
                t3_err[i, :, :] = t3_err[i, imap.astype(int), :]
                t3_sta[i, :, :] = t3_sta[i, imap.astype(int), :]
    
    return wavel, dwavel, vis2, vis2_err, vis2_u, vis2_v, vis2_sta, t3, t3_err, t3_sta

def load_p2vmred(idir,
                 fitsfiles,
                 insname):
    """
    """
    
    wavel = []
    vis = []
    vis_err = []
    vis_sta = []
    f1f2 = []
    for i in range(len(fitsfiles)):
        hdul = pyfits.open(idir+fitsfiles[i])
        for j in range(len(hdul)):
            try:
                if (hdul[j].header['INSNAME'] == insname):
                    if (hdul[j].header['EXTNAME'] == 'OI_WAVELENGTH'):
                        wavel += [hdul[j].data['EFF_WAVE']]
                    if (hdul[j].header['EXTNAME'] == 'OI_VIS'):
                        
                        vis += [hdul[j].data['VISDATA']]
                        vis[-1][hdul[j].data['FLAG']] = np.nan
                        vis_err += [hdul[j].data['VISERR']]
                        vis_err[-1][hdul[j].data['FLAG']] = np.nan
                        vis_sta += [hdul[j].data['STA_INDEX']]
                        f1f2 += [hdul[j].data['F1F2']]
                        f1f2[-1][hdul[j].data['FLAG']] = np.nan
            except:
                pass
        hdul.close()
    wavel = np.array(wavel)
    vis = np.array(vis)
    vis_err = np.array(vis_err)
    vis_sta = np.array(vis_sta)
    f1f2 = np.array(f1f2)
    
    return wavel, vis, vis_err, vis_sta, f1f2
