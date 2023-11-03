"""
    Script for reading in map of Br(theta, phi) on the solar surface, computing a
    PFSS extrapolation, and outputting to a netcdf file.
    
    Copyright (C) Anthony R. Yeates, Durham University May 2023

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np
import matplotlib.pyplot as plt
import ftplib
import astropy.units as units
from astropy.io import fits
import sunpy.map
from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time
from sunpy.coordinates.sun import L0
from scipy.interpolate import interp2d
from scipy.ndimage.filters import gaussian_filter as gauss
import drms
from datetime import timedelta, datetime
import urllib
import sys

def correct_flux_multiplicative(f):
    """
        Correct the flux balance in the map f (assumes that cells have equal area).
    """

    # Compute positive and negative fluxes:
    ipos = f > 0
    ineg = f < 0
    fluxp = np.abs(np.sum(f[ipos]))
    fluxn = np.abs(np.sum(f[ineg]))

    # Rescale both polarities to mean:
    fluxmn = 0.5 * (fluxn + fluxp)
    f1 = f.copy()
    f1[ineg] *= fluxmn / fluxn
    f1[ipos] *= fluxmn / fluxp

    return f1


def plgndr(m, x, lmax):
    """
        Evaluate associated Legendre polynomials P_lm(x) for given (positive)
        m, from l=0,lmax, with spherical harmonic normalization included.
        Only elements l=m:lmax are non-zero.
        
        Similar to scipy.special.lpmv except that function only works for 
        small l due to overflow, because it doesn't include the normalizationp.
    """

    nx = np.size(x)
    plm = np.zeros((nx, lmax + 1))
    pmm = 1
    if m > 0:
        somx2 = (1 - x) * (1 + x)
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= somx2 * fact / (fact + 1)
            fact += 2

    pmm = np.sqrt((m + 0.5) * pmm)
    pmm *= (-1) ** m
    plm[:, m] = pmm
    if m < lmax:
        pmmp1 = x * np.sqrt(2 * m + 3) * pmm
        plm[:, m + 1] = pmmp1
        if m < lmax - 1:
            for l in range(m + 2, lmax + 1):
                fact1 = np.sqrt(
                    ((l - 1.0) ** 2 - m ** 2) / (4.0 * (l - 1.0) ** 2 - 1.0)
                )
                fact = np.sqrt((4.0 * l ** 2 - 1.0) / (l ** 2 - m ** 2))
                pll = (x * pmmp1 - pmm * fact1) * fact
                pmm = pmmp1
                pmmp1 = pll
                plm[:, l] = pll
    return plm

# --------------------------------------------------------------------------------
def readcrmap_hmi(rot, ns, nph, smooth=0):
    """
        Reads the HMI synoptic map for Carrington rotation rot, map to the DuMFric grid, and correct the flux balance.
        Also reads in the neighbouring maps, and puts them together for smoothing.
        
        ARGUMENTS:
            rot is the number of the required Carrington rotation (e.g. 2190)
            ns and nph define the required grid (e.g. 180 and 360)
            smooth [optional] controls the strength of smoothing (default 0 is no smoothing)
    """

    # (1) READ IN DATA AND STITCH TOGETHER 3 ROTATIONS
    # ------------------------------------------------
    # Read in map and neighbours, and extract data arrays:
    # The seg='Mr_polfil' downloads the polar field corrected data --> without NaN values and errors due to projection effect
    # Read "Polar Field Correction for HMI Line-of-Sight Synoptic Data" by Xudong Sun, 2018 to know more
    # Link: https://arxiv.org/pdf/1801.04265.pdf
    # Data is stored in 2nd slot with No. 1 and not in the PRIMARY (No. 0), thus [1].data
    # typical file structure:
    # No.    Name      Ver    Type        Cards   Dimensions     Format
    # 0     PRIMARY    1   PrimaryHDU       6     ()
    # 1                1   CompImageHDU     13    (3600, 1440)   int32
    
    try:
        c = drms.Client()
        seg = c.query(("hmi.synoptic_mr_polfil_720s[%4.4i]" % rot), seg="Mr_polfil")
        with fits.open("http://jsoc.stanford.edu" + seg.Mr_polfil[0]) as fid:
            brm = fid[1].data
    except:
        print(
            "Error downloading HMI synoptic map for CR %4.4i"
            % (rot)
        )
        sys.exit(1)
    
    try:
        segl = c.query(
            ("hmi.synoptic_mr_polfil_720s[%4.4i]" % (rot + 1)), seg="Mr_polfil"
        )
        with fits.open("http://jsoc.stanford.edu" + segl.Mr_polfil[0]) as fid:
            brm_l = fid[1].data
    except:
        print(
            "Warning: error downloading HMI synoptic map for neighbouring rotation CR %4.4i."
            % (rot + 1)
        )
        brm_l = brm*0

    try:
        segr = c.query(
            ("hmi.synoptic_mr_polfil_720s[%4.4i]" % (rot - 1)), seg="Mr_polfil"
        )
        with fits.open("http://jsoc.stanford.edu" + segr.Mr_polfil[0]) as fid:
            brm_r = fid[1].data
    except:
        print(
            "Warning: error downloading HMI synoptic map for neighbouring rotation CR %4.4i."
            % (rot - 1)
        )
        brm_r = brm*0

    # Stitch together:
    brm3 = np.concatenate((brm_l, brm, brm_r), axis=1)
    del (brm, brm_l, brm_r)

    # Remove NaNs:
    brm3 = np.nan_to_num(brm3)

    # Coordinates of original map (pretend it goes only once around Sun in longitude!):
    nsm = np.size(brm3, axis=0)
    npm = np.size(brm3, axis=1)
    dsm = 2.0 / nsm
    dpm = 2 * np.pi / npm
    scm = np.linspace(-1 + 0.5 * dsm, 1 - 0.5 * dsm, nsm)
    pcm = np.linspace(0.5 * dpm, 2 * np.pi - 0.5 * dpm, npm)

    # (2) SMOOTH COMBINED MAP WITH SPHERICAL HARMONIC FILTER
    # ------------------------------------------------------
    if smooth > 0:
        # Azimuthal dependence by FFT:
        brm3 = np.fft.fft(brm3, axis=1)

        # Compute Legendre polynomials on equal (s, ph) grid,
        # with spherical harmonic normalisation:
        lmax = 2 * int((nph - 1) / 2)  # note - already lower resolution
        nm = 2 * lmax + 1  # only need to compute this many values
        plm = np.zeros((nsm, nm, lmax + 1))
        for m in range(lmax + 1):
            plm[:, m, :] = plgndr(m, scm, lmax)
        plm[:, nm - 1 : (nm - lmax - 1) : -1, :] = plm[:, 1 : lmax + 1, :]

        # Compute spherical harmonic coefficients:
        blm = np.zeros((nm, lmax + 1), dtype="complex")
        for l in range(lmax + 1):
            blm[: lmax + 1, l] = np.sum(
                plm[:, : lmax + 1, l] * brm3[:, : lmax + 1] * dsm, axis=0
            )
            blm[lmax + 1 :, l] = np.sum(
                plm[:, lmax + 1 :, l] * brm3[:, -lmax:] * dsm, axis=0
            )
            # Apply smoothing filter:
            blm[:, l] *= np.exp(-smooth * l * (l + 1))

        # Invert transform:
        brm3[:, :] = 0.0
        for j in range(nsm):
            brm3[j, : lmax + 1] = np.sum(
                blm[: lmax + 1, :] * plm[j, : lmax + 1, :], axis=1
            )
            brm3[j, -lmax:] = np.sum(blm[lmax + 1 :, :] * plm[j, lmax + 1 :, :], axis=1)

        brm3 = np.real(np.fft.ifft(brm3, axis=1))

    # (3) INTERPOLATE CENTRAL MAP TO COMPUTATIONAL GRID
    # -------------------------------------------------
    # Form computational grid arrays:
    ds = 2.0 / ns
    dph = 2 * np.pi / nph
    sc = np.linspace(-1 + 0.5 * ds, 1 - 0.5 * ds, ns)
    pc1 = np.linspace(0.5 * dph, 2 * np.pi - 0.5 * dph, nph)
    pc = pc1 / 3 + 2 * np.pi / 3  # coordinate on the stitched grid

    # Interpolate to the computational grid:
    bri = interp2d(
        pcm, scm, brm3, kind="cubic", copy=True, bounds_error=False, fill_value=0
    )
    br = np.zeros((ns, nph))
    for i in range(ns):
        br[i, :] = bri(pc, sc[i]).flatten()
    del (brm3, bri)

    # (4) CORRECT FLUX BALANCE
    # ------------------------
    br = correct_flux_multiplicative(br)

    return br

#--------------------------------------------------------------------------------
def readcrmap_gong(rot, ns, nph, smooth=0):
    """
        Download the GONG synoptic map for Carrington rotation rot, map to the DUMFRIC grid, and correct flux balance.
        Also read in the neighbouring maps, and put them together for smoothing.
        
        ARGUMENTS:
            rot is the number of the required Carrington rotation (e.g. 2190)
            ns and nph define the required grid (e.g. 180 and 360)
            smooth [optional] controls the strength of smoothing (default 0 is no smoothing)
        
        Sets br=0 if no synoptic map is found (for either the main rotation or the previous/next ones).
    """

    # (1) READ IN DATA FROM FTP AND STITCH TOGETHER 3 ROTATIONS
    # ---------------------------------------------------------
    ftp = ftplib.FTP('gong2.nso.edu')
    ftp.login()
    
    file_rot = get_gongcr_filename(rot, ftp)
    try:
        brm = (fits.open('ftp://gong2.nso.edu/'+file_rot))[0].data
        print('FOUND GONG CR MAP FOR CR%4.4i' % rot)
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % rot)
        brm = np.zeros((ns, nph))
    file_rot_l = get_gongcr_filename(rot+1, ftp)
    try:
        brm_l = (fits.open('ftp://gong2.nso.edu/'+file_rot_l))[0].data
        print('FOUND GONG CR MAP FOR CR%4.4i' % (rot+1))
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % (rot+1))
        brm_l = np.zeros((ns, nph))
    file_rot_r = get_gongcr_filename(rot-1, ftp)
    try:
        brm_r = (fits.open('ftp://gong2.nso.edu/'+file_rot_r))[0].data
        print('FOUND GONG CR MAP FOR CR%4.4i' % (rot-1))
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % (rot-1))
        brm_r = np.zeros((ns, nph))
        
    nsm = np.size(brm, axis=0)
    npm = np.size(brm, axis=1)
    dsm = 2.0/nsm
    dpm = 2*np.pi/npm
    scm = np.linspace(-1 + 0.5*dsm, 1 - 0.5*dsm, nsm)
    pcm = np.linspace(0.5*dpm, 2*np.pi - 0.5*dpm, npm)
    
    # Stitch together:
    brm3 = np.concatenate((brm_l, brm, brm_r), axis=1)
    del(brm, brm_l, brm_r)

    # Remove NaNs:
    brm3 = np.nan_to_num(brm3)

    # Coordinates of combined map (pretend it goes only once around Sun in longitude!):
    nsm = np.size(brm3, axis=0)
    npm = np.size(brm3, axis=1)
    dsm = 2.0/nsm
    dpm = 2*np.pi/npm
    scm = np.linspace(-1 + 0.5*dsm, 1 - 0.5*dsm, nsm)
    pcm = np.linspace(0.5*dpm, 2*np.pi - 0.5*dpm, npm)

    # (2) SMOOTH COMBINED MAP WITH SPHERICAL HARMONIC FILTER
    # ------------------------------------------------------
    if (smooth > 0):
        # Azimuthal dependence by FFT:
        brm3 = np.fft.fft(brm3, axis=1)

        # Choose suitable lmax based on smoothing filter coefficient:
        # -- such that exp[-smooth*lmax*(lmax+1)] < 0.05
        # -- purpose of this is to make sure high l's are suppressed, to avoid ringing
        lmax = 0.5*(-1 + np.sqrt(1-4*np.log(0.05)/smooth))
        print('lmax = %i' % lmax)

        # Compute Legendre polynomials on equal (s, ph) grid,
        # with spherical harmonic normalisation:
        lmax = 2*int((nph-1)/2)  # note - already lower resolution
        nm = 2*lmax+1  # only need to compute this many values
        plm = np.zeros((nsm, nm, lmax+1))
        for m in range(lmax+1):
            plm[:,m,:] = plgndr(m, scm, lmax)
        plm[:,nm-1:(nm-lmax-1):-1,:] = plm[:,1:lmax+1,:]

        # Compute spherical harmonic coefficients:
        blm = np.zeros((nm,lmax+1), dtype='complex')
        for l in range(lmax+1):
            blm[:lmax+1,l] = np.sum(plm[:,:lmax+1,l]*brm3[:,:lmax+1]*dsm, axis=0)
            blm[lmax+1:,l] = np.sum(plm[:,lmax+1:,l]*brm3[:,-lmax:]*dsm, axis=0)
            # Apply smoothing filter:
            blm[:,l] *= np.exp(-smooth*l*(l+1))

        # Invert transform:
        brm3[:,:] = 0.0
        for j in range(nsm):
            brm3[j,:lmax+1] = np.sum(blm[:lmax+1,:]*plm[j,:lmax+1,:], axis=1)
            brm3[j,-lmax:] = np.sum(blm[lmax+1:,:]*plm[j,lmax+1:,:], axis=1)

        brm3 = np.real(np.fft.ifft(brm3, axis=1))

    # (3) INTERPOLATE CENTRAL MAP TO COMPUTATIONAL GRID
    # -------------------------------------------------
    # Form computational grid arrays:
    ds = 2.0/ns
    dph = 2*np.pi/nph
    sc = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
    pc1 = np.linspace( 0.5*dph, 2*np.pi - 0.5*dph, nph)
    pc = pc1/3 + 2*np.pi/3  # coordinate on the stitched grid

    # Interpolate to the computational grid:
    bri = interp2d(pcm, scm, brm3, kind='cubic', copy=True, bounds_error=False, fill_value=0)
    br = np.zeros((ns, nph))
    for i in range(ns):
        br[i,:] = bri(pc, sc[i]).flatten()

    # (4) INTERPOLATE LEFT AND RIGHT MAPS TO COMPUTATIONAL GRID
    # ---------------------------------------------------------
    brl = np.zeros((ns, nph))
    brr = np.zeros((ns, nph))
    for i in range(ns):
        brl[i,:] = bri(pc - 2*np.pi/3, sc[i]).flatten()
        brr[i,:] = bri(pc + 2*np.pi/3, sc[i]).flatten()

    del(brm3, bri)

    # (5) CORRECT FLUX BALANCE
    # ------------------------
    br = correct_flux_multiplicative(br)
    
    return br

#--------------------------------------------------------------------------------
def get_gongcr_filename(rot, ftp):
    """
    Identify filename for a GONG map on FTP server, otherwise return ''.
    """
    # Identify date corresponding to 180 Carrington longitude (middle of map).
    # [This seems to be how GONG CR maps are labelled.]
    t0 = carrington_rotation_time(rot, longitude=180*units.deg)
    t0.format = 'datetime'
    
    # Get ftp subdirectory [YYYYmm/]:
    subdir = t0.strftime('QR/mqs/%Y%m/mrmqs%y%m%d/')
    try:
        ftp.cwd(subdir)
    except:
        return ''

    # List files in the directory:
    gongfiles = ftp.nlst()
    ftp.cwd('~/')

    for file in gongfiles:
        if file[17:21] == ('%4.4i' % rot):
            return subdir+file
    return ''
