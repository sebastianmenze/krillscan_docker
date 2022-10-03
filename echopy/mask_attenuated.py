#!/usr/bin/env python3
"""
Contains different modules for masking Attenuated Signal (AS).
    
Created on Fri Apr 27 14:18:05 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

import numpy as np
from echopy.transform import lin, log
from echopy.resample import twod, full
from skimage.measure import label

def ryan(Sv, r, r0, r1, n, thr, start=0):
    """
    Locate attenuated signal and create a mask following the attenuated signal 
    filter as in:
        
        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in 
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Scattering Layers (SLs) are continuous high signal-to-noise regions with 
    low inter-ping variability. But attenuated pings create gaps within SLs. 
                                                 
       attenuation                attenuation       ping evaluated
    ______ V _______________________ V ____________.....V.....____________
          | |   scattering layer    | |            .  block  .            |
    ______| |_______________________| |____________...........____________|
    
    The filter takes advantage of differences with preceding and subsequent 
    pings to detect and mask attenuation. A comparison is made ping by ping 
    with respect to a block of the reference layer. The entire ping is masked 
    if the ping median is less than the block median by a user-defined 
    threshold value.
    
    Args:
        Sv (float): 2D array with Sv data to be masked (dB). 
        r (float):  1D array with range data (m).
        r0 (int): upper limit of SL (m).
        r1 (int): lower limit of SL (m).
        n (int): number of preceding & subsequent pings defining the block.
        thr (int): user-defined threshold value (dB).
        start (int): ping index to start processing.
        
    Returns:
        list: 2D boolean array with AS mask and 2D boolean array with mask
              indicating where AS detection was unfeasible.
    """
    
     # raise errors if wrong arguments
    if r0>r1:
        raise Exception('Minimum range has to be shorter than maximum range')
    
    # return empty mask if searching range is outside the echosounder range
    if (r0>r[-1]) or (r1<r[0]):
        mask  = np.zeros_like(Sv, dtype=bool) 
        mask_ = np.zeros_like(Sv, dtype=bool) 
        return mask, mask_ 
    
    # turn layer boundaries into arrays with length = Sv.shape[1]
    r0 = np.ones(Sv.shape[1])*r0
    r1 = np.ones(Sv.shape[1])*r1
    
    # start masking process    
    mask_ = np.zeros(Sv.shape, dtype=bool)
    mask = np.zeros(Sv.shape, dtype=bool)    
    for j in range(start, len(Sv[0])):
        
        # find indexes for upper and lower SL limits
        up = np.argmin(abs(r - r0[j]))
        lw = np.argmin(abs(r - r1[j]))
            # TODO: now indexes are the same at every loop, but future 
            # versions will have layer boundaries with variable range
            # (need to implement mask_layer.py beforehand!)
        
        # mask where AS evaluation is unfeasible (e.g. edge issues, all-NANs)
        if (j-n<0) | (j+n>len(Sv[0])-1) | np.all(np.isnan(Sv[up:lw, j])):        
            mask_[:, j] = True
        
        # compare ping and block medians otherwise & mask ping if too different
        else:
            pingmedian  = log(np.nanmedian(lin(Sv[up:lw, j])))
            blockmedian = log(np.nanmedian(lin(Sv[up:lw, (j-n):(j+n)])))
            if (pingmedian-blockmedian)<thr:            
                mask[:, j] = True
         
    return [mask[:, start:], mask_[:, start:]]

def ariza_seabed(Sv, r, offset=20, thr=(-40,-35), m=20, n=50):
    """
    Mask attenuated pings by looking at seabed breaches.
    
    Ariza (in progress).
    """
    
    # get ping array
    p = np.arange(len(Sv[0]))
    
    # set to NaN shallow waters and data below the Sv threshold
    Sv_ = Sv.copy()
    Sv_[0:np.nanargmin(abs(r - offset)), :] = np.nan
    Sv_[Sv_<-thr[0]] = np.nan
    
    # bin Sv
    # TODO: update to 'twod' and 'full' funtions    
    Sv_bnd, r_bnd, p_bnd = bin2d(Sv_, r, p, m, n, operation='mean')[0:3]
    Sv_bnd = bin2dback(Sv_bnd, r_bnd, p_bnd, r, p)
    
    # label binned Sv data features
    Sv_lbl = label(~np.isnan(Sv_bnd))
    labels = np.unique(Sv_lbl)
    labels = np.delete(labels, np.where(labels==0))
    
    # list the median values for each Sv feature
    val = []
    for lbl in labels:
        val.append(log(np.nanmedian(lin(Sv_bnd[Sv_lbl==lbl]))))
    
    # keep the feature with a median above the Sv threshold (~seabed)
    # and set the rest of the array to NaN
    if val:
        if np.nanmax(val)>thr[1]:
            labels = labels[val!=np.nanmax(val)]
            for lbl in labels:
                Sv_bnd[Sv_lbl==lbl] = np.nan
        else:
            Sv_bnd[:] = np.nan
    else:
        Sv_bnd[:] = np.nan
        
    # remove everything in the original Sv array that is not seabed
    Sv_sb = Sv.copy()
    Sv_sb[np.isnan(Sv_bnd)] = np.nan
    
    # compute the percentile 90th for each ping, at the range at which 
    # the seabed is supposed to be.    
    seabed_percentile = log(np.nanpercentile(lin(Sv_sb), 95, axis=0))
    
    # get mask where this value falls bellow a Sv threshold (seabed breaches)
    mask = seabed_percentile<thr[0]
    mask = np.tile(mask, [len(Sv), 1])    
    
    return mask
    
def other():
    """
    Note to contributors:
        Other algorithms for masking attenuated signal must be named with the
        author or method name. If already published, the full citation must be
        provided. Please, add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check DESIGN.md to adhere to our coding style.
    """