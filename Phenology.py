###############################################################################
#
# PhenoPy is a Python 3.X library to process phenology indices derived from
# EarthObservation data.
#
###############################################################################

# libraries included in Python 3.X
from __future__ import division
import concurrent.futures
from functools import partial
import sys
import warnings

# common dependencies
import pandas as pd
import numpy as np
import dask as da
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from sklearn.metrics import mean_squared_error

# speciel dependencies
import xarray as xr                  # manipulate 3D time-series rasters
import shapely.geometry as geom      # create geographical points
import rioxarray as rio                     # manipulate GeoTIFF
from rasterstats import point_query  # extract raster values
from tqdm import tqdm                # progress bar

# builtin functions from Open Data Cube
sys.path.append('../Scripts')
from deafrica_datahandling import load_ard
from deafrica_bandindices import calculate_indices

# --------------------------------------------------------------------------- #
# ------------------------------- FUNCTIONS --------------------------------- #
# --------------------------------------------------------------------------- #

def PhenoShape(ds, collection, veg_proxy='NDVI', interpolType='linear', nan_replace=None,
               rollWindow=None, nGS=46):
    """
    Process phenological shape of remote sensing data by
    folding data to day-of-the-year. 
    
    Parameters
    ----------
    - ds: ds object
    - veg_proxy: String
        name of the vegetatoin index to be used i the analysis. Options are:
        'AWEI_ns' (Automated Water Extraction Index,
                  no shadows, Feyisa 2014)
        'AWEI_sh' (Automated Water Extraction Index,
                   shadows, Feyisa 2014)
        'BAEI' (Built-Up Area Extraction Index, Bouzekri et al. 2015) 
        'BAI' (Burn Area Index, Martin 1998)
        'BSI' (Bare Soil Index, Rikimaru et al. 2002)
        'BUI' (Built-Up Index, He et al. 2010)
        'CMR' (Clay Minerals Ratio, Drury 1987)
        'EVI' (Enhanced Vegetation Index, Huete 2002)
        'FMR' (Ferrous Minerals Ratio, Segal 1982)
        'IOR' (Iron Oxide Ratio, Segal 1982)  
        'LAI' (Leaf Area Index, Boegh 2002)
        'MNDWI' (Modified Normalised Difference Water Index, Xu 1996) 
        'MSAVI' (Modified Soil Adjusted Vegetation Index, 
                 Qi et al. 1994)              
        'NBI' (New Built-Up Index, Jieli et al. 2010)
        'NBR' (Normalised Burn Ratio, Lopez Garcia 1991)
        'NDBI' (Normalised Difference Built-Up Index, Zha 2003)
        'NDCI' (Normalised Difference Chlorophyll Index, 
                Mishra & Mishra, 2012)
        'NDMI' (Normalised Difference Moisture Index, Gao 1996)        
        'NDSI' (Normalised Difference Snow Index, Hall 1995)
        'NDVI' (Normalised Difference Vegetation Index, Rouse 1973)
        'NDWI' (Normalised Difference Water Index, McFeeters 1996)
        'SAVI' (Soil Adjusted Vegetation Index, Huete 1988)
        'TCB' (Tasseled Cap Brightness, Crist 1985)
        'TCG' (Tasseled Cap Greeness, Crist 1985)
        'TCW' (Tasseled Cap Wetness, Crist 1985)
        'WI' (Water Index, Fisher 2016) 
        default = "NDVI"
    - interpolType: String or Integer
        Type of interpolation to perform. Options include ‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘RBF‘, ‘previous’, ‘next’,
        where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
        interpolation of zeroth, first, second or third order; ‘previous’ and
        ‘next’ simply return the previous or next value of the point.
         Integer numbers specify the order of the spline interpolator to use.
         Default is ‘linear’.
    - doy: 1D vector with day of the year data
        Dates of the original timeseries data [dtype: datetime64[ns]]
    - nan_replace: Integer
        Value of the NaN data if there are any
    - rollWindow: Integer
        Value of avarage smoothing of linear trend [default None]
    - nGS: Integer
        Number of observations to predict the PhenoShape
        default is 46 [one per week]

    """
    # get vegatation index
    ds = calculate_indices(ds, index=veg_proxy, collection=collection)
    
    # rearrange DOY values according to phenology for the south hemisphere
    # start =  21 of July (DOY=202)
    south_doy = np.append(np.linspace(202,365,163, dtype='int16'), np.linspace(1,201,201,dtype='int16'))
    doy = ds.doy.values

    ## find location of real doy values in the south_doy data
    doy_where =[]
    for x in np.nditer(doy):
        doy_where.append( np.where(south_doy == x)[0][0] )
    doy_where = np.array(doy_where)

    # add south_doy to xarray and sort accordingly
    ds = ds.update(ds.assign(south_doy=doy_where))
    ds2 = ds.sortby(ds.south_doy)
 
    # get names for output bands
    xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype='int16') # weekly doy
    xnew2 =[]
    for x in np.nditer(xnew):
        xnew2.append( np.where(south_doy == x)[0][0] )
    xnew2 = np.array(xnew2)
    
    # call _getPheno2 function to lcoal
    phenoshape = _getPheno2(dstack=ds2[veg_proxy], doy=doy_where, interpolType=interpolType, 
                      nan_replace=nan_replace, rollWindow=rollWindow, nGS=nGS)
    
    return xr.DataArray(data=phenoshape, dims=["time","x", "y"], coords=[xnew2,ds2.x,ds2.y])


def PhenoLSP(inData, outData, doy, nGS=46, phentype=1, n_phen=10, n_jobs=4,
             chuckSize=256):
    """
    Obtain land surfurface phenology metrics for a PhenoShape product

    Parameters
    ----------
    - inData: String
        Absolute path to PhenoShape raster data
    - outData: String
        Absolute path for output land surface phenology raster
    - doy: 1D array
        Numpy array of the days of the year of the time series
    - nGS: Integer
        Number of observations to predict the PhenoShape
        default is 46; one per week
     - phenType: Type os estimation of SOS and EOS. 1 = median value between POS and start and end of season. 2 = using the knee inflexion method.
            default 1
    - n_phen: Integer
        Window size where to estimate SOS and EOS
    - n_jobs: Integer
        Number of parallel jorb to apply during modeling
    - chuckSize: Integer
        Size of raster chunks to be loaded during modeling
        Number must be multiple of 16 [GDAL specifications]
        default value is 256 [256 X 256 raster blocks]

    outputs
    -------
    Raster stack with the followingvariables:
        - SOS - DOY of Start of season
        - POS - DOY of Peak of season
        - POS - DOY of End of season
        - vSOS - Vaues at start os season
        - vPOS - Values at peak of season
        - vEOS - Values at end of season
        - LOS - Length of season
        - MSP - Mid spring (DOY)
        - MAU - Mid autum (DOY)
        - vMSP - Value at mid spring
        - vMAU - Value at mid autum
        - AOS - Amplitude of season
        - IOS - Integral of season [SOS-EOS]
        - ROG - Rate of greening [slope SOS-POS]
        - ROS - Rate of senescence [slope POS-EOS]
        - SW - Skewness of growing season [SOS-EOS]
    """

    # name of output bands
    bandNames = ['SOS - DOY of Start of season',
                 'POS - DOY of Peak of season',
                 'EOS - DOY of End of season',
                 'vSOS - Vaues at start os season',
                 'vPOS - Values at peak of season',
                 'vEOS - Values at end of season',
                 'LOS - Length of season',
                 'MSP - Mid spring (DOY)',
                 'MAU - Mid autum (DOY)',
                 'vMSP - Value at mean spring',
                 'vMAU - Value at mean autum',
                 'AOS - Amplitude of season',
                 'IOS - Integral of season [SOS-EOS]',
                 'ROG - Rate of greening [slope SOS-POS]',
                 'ROS - Rate of senescence [slope POS-EOS]',
                 'SW - Skewness of growing season [SOS-EOS]']

    # call _cal_LSP function to local
    do_work = partial(_cal_LSP, nGS=nGS, phentype=phentype, doy=doy,
                      n_phen=n_phen, num=len(bandNames))
    # apply PhenoLSP with parallel processing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _parallel_process(inData, outData, do_work, len(bandNames), n_jobs,
                              chuckSize, bandNames)
    except AttributeError:
        print('ERROR in parallel processing...')
        
        
        
        
# ----------------------------------
# Miscellaneous Functions
# ----------------------------------

def _getPheno(y, x, nGS, type):
    """
    Apply linear interpolation in the 'time' axis
    x: DOY values
    y: ndarray with VI values
    """
    inds = np.isnan(y)  # check if array has NaN values
    if np.sum(inds) == len(y):  # check is all values are NaN
        return y[0:nGS]
    else:
        try:
            xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
            if inds.any():  # if inds have at least one True
                y = _fillNaN(y)
            if type == 'linear':
                ynew = np.interp(xnew, x, y)
            elif type == 'RBF':
                _replaceElements(x)  # replace doy values when are the same
                f = Rbf(x, y, funciton='cubic')
                ynew = f(xnew)
            elif type == 'KDE':
                ynew = _KDE(x, y, nGS)
            else:
                _replaceElements(x)  # replace doy values when are the same
                f = interp1d(x, y, kind=type)
                ynew = f(xnew)
            """
            plt.plot(x, y, 'o')
            plt.plot(xnew, ynew)
            plt.plot(x[inds], y[inds], 'X')
            """

            return ynew

        except NotImplementedError:
            print("ERROR: Interpolation type must be ‘KDE’ ‘linear’, ‘nearest’,"
                  "‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘RBF‘, ‘previous’,"
                  "or ‘next’. Here, 'KSE' correspond to a non-parametric linear"
                  "regression using Kernel Density Estimators with Gaussian kernel."
                  "‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline "
                  "interpolation of zeroth, first, second or third order;"
                  "‘previous’ and ‘next’ simply return the previous or next value"
                  "of the point) or as an integer specifying the order of the"
                  "spline interpolator to use. Default is KDE.")


# ---------------------------------------------------------------------------#

def _getPheno0(y, doy, interpolType, nan_replace, rollWindow, nGS):
 
    # replace nan_relace values by NaN
    if nan_replace is not None:
        y = np.where(y == nan_replace, np.nan, y)
  
    # sort values by DOY  
    idx = doy.argsort()
    y = y[idx]     
    
    # get phenological shape
    phen = _getPheno(y, doy[idx], nGS, interpolType)
    
    # rolling average using moving window
    if rollWindow is not None:
        phen = _moving_average(phen, rollWindow)
    
    return phen

# ---------------------------------------------------------------------------#
def _getPheno2(dstack, doy, interpolType, nan_replace, rollWindow, nGS):
    """
    Obtain shape of phenological responses

    Parameters
    ----------
    - dstack: 3D arrays

    """    

    # get phenology shape accross the time axis
    return np.apply_along_axis(_getPheno0, 0, dstack, doy, interpolType, 
                               nan_replace, rollWindow, nGS)

# ---------------------------------------------------------------------------#

def LSPmetrics(phen, xnew, nGS, num, phentype):
    """
    Obtain land surfurface phenology metrics

    Parameters
    ----------
    - phen: 1D array
        PhenoShape data
    - xnew: 1D array
        DOY values for PhenoShape data
    - n_phen: Integer
        Window size where to estimate SOS and EOS
    - num: Integer
        Number of output variables

    Outputs
    -------
    - 2D array with the following variables:

        SOS = DOY of start of season
        POS = DOY of peak of season
        EOS = DOY of end of season
        vSOS = Value at start of season
        vPOS = Value at peak of season
        vEOS = Value at end of season
        LOS = Length of season (DOY)
        MSP = Mean spring (DOY)
        MAU = Mean autum (DOY)
        vMSP = Mean spring value
        vMAU = Mean autum value
        AOS = Amplitude of season (in value units)
        IOS = Integral of season (SOS-EOS)
        ROG = Rate of greening [slope SOS-POS]
        ROS = Rate of senescence [slope POS-EOS]
        SW = Skewness of growing season
    """
    inds = np.isnan(phen)  # check if array has NaN values
    if inds.any():  # check is all values are NaN
        return np.repeat(np.nan, num)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # basic variables
                vpos = np.max(phen)
                ipos = np.where(phen == vpos)[0]
                pos = xnew[ipos]
                trough = np.min(phen)
                ampl = vpos - trough

                # get position of seasonal peak and trough
                ipos = np.where(phen == vpos)[0]

                # scale annual time series to 0-1
                ratio = (phen - trough) / ampl

                # separate greening from senesence values
                dev = np.gradient(ratio)  # first derivative
                greenup = np.zeros([ratio.shape[0]],  dtype=bool)
                greenup[dev > 0] = True

                # select time where SOS and EOS are located (arround trs value)
                # KneeLocator looks for the inflection index in the curve
                try:
                    with warnings.catch_warnings():
                        # estimate SOS and EOS as median of the season
                        i = np.median(xnew[:ipos[0]][greenup[:ipos[0]]])
                        ii = np.median(xnew[ipos[0]:][~greenup[ipos[0]:]])
                        sos = xnew[(np.abs(xnew - i)).argmin()]
                        eos = xnew[(np.abs(xnew - ii)).argmin()]
                        isos = np.where(xnew == int(sos))[0]
                        ieos = np.where(xnew == eos)[0]
                    if sos is None:
                        isos = 0
                        sos = xnew[isos]
                    if eos is None:
                        ieos = len(xnew) - 1
                        eos = xnew[ieos]
                except ValueError:
                    sos = np.nan
                    isos = np.nan
                    eos = np.nan
                    ieos = np.nan
                except TypeError:
                    sos = np.nan
                    isos = np.nan
                    eos = np.nan
                    ieos = np.nan

                # los: length of season
                try:
                    los = eos - sos
                    if los < 0:
                        los[los < 0] = len(phen) + \
                            (eos[los < 0] - sos[los < 0])
                except ValueError:
                    los = np.nan
                except TypeError:
                    los = np.nan

                # get MSP, MAU (independent from SOS and EOS)
                # mean spring
                try:
                    idx = np.mean(xnew[(xnew > sos) & (xnew < pos[0])])
                    idx = (np.abs(xnew - idx)).argmin()  # indexing value
                    msp = xnew[idx]  # DOY of MGS
                    vmsp = phen[idx]  # mgs value

                except ValueError:
                    msp = np.nan
                    vmsp = np.nan
                except TypeError:
                    msp = np.nan
                    vmsp = np.nan
                # mean autum
                try:
                    idx = np.mean(xnew[(xnew < eos) & (xnew > pos[0])])
                    idx = (np.abs(xnew - idx)).argmin()  # indexing value
                    mau = xnew[idx]  # DOY of MGS
                    vmau = phen[idx]  # mgs value

                except ValueError:
                    mau = np.nan
                    vmau = np.nan
                except TypeError:
                    mau = np.nan
                    vmau = np.nan

                # doy of growing season
                try:
                    green = xnew[(xnew > sos) & (xnew < eos)]
                    id = []
                    for i in range(len(green)):
                        id.append((xnew == green[i]).nonzero()[0])
                    # index of growing season
                    id = np.array([item for sublist in id for item in sublist])
                except ValueError:
                    id = np.nan
                except TypeError:
                    id = np.nan

                # get intergral of green season
                try:
                    ios = trapz(phen[id], xnew[id])
                except ValueError:
                    ios = np.nan
                except TypeError:
                    ios = np.nan

                # rate of greening [slope SOS-POS]
                try:
                    rog = (vpos - phen[isos]) / (pos - sos)
                except ValueError:
                    rog = np.nan
                except TypeError:
                    rog = np.nan

                # rate of senescence [slope POS-EOS]
                try:
                    ros = (phen[ieos] - vpos) / (eos - pos)
                except ValueError:
                    ros = np.nan
                except TypeError:
                    ros = np.nan

                # skewness of growing season
                try:
                    sw = skew(phen[id])
                except ValueError:
                    sw = np.nan
                except TypeError:
                    sw = np.nan

                metrics = np.array((sos, pos[0], eos, phen[isos][0], vpos,
                                    phen[ieos][0], los, msp, mau, vmsp, vmau, ampl, ios, rog[0],
                                    ros[0], sw))

                return metrics

        except IndexError:
            return np.repeat(np.nan, num)
        except ValueError:
            return np.repeat(np.nan, num)
        except TypeError:
            return np.repeat(np.nan, num)

# ---------------------------------------------------------------------------#


def _cal_LSP(dstack, nGS, doy, n_phen, num, phentype):
    """
    Process the _LSP funciton into an 3D arrays

    Parameters
    ----------
    - dstack: 3D array
        PhenoShape data.
    - min_sep: integer
         Distance to consider betweem peaks and bottoms.
    - nGS: integer
        Number of DOY values
    - num: Integer
        Number of output variables

    Output
    ------
    - 3D arrays
        Stack with the LSP metrics descibes in _LSP
    """
    # prepare input data for _LSP
    xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype='int16')

    # estimate LSP metrics along the 0 axis
    return da.array.apply_along_axis(_getLSPmetrics, 0, dstack, xnew, nGS, num, phentype)

# ---------------------------------------------------------------------------#


def _RMSE(x, y, xnew, ynew):
    """
    Obtain RMSE values form 1D data inputs

    Parameters
    ----------
    - x, y: 1D array
        Values for DOY and time series data from the original dataset
    - xnew, ynew: 1D array
        Values of DOY and PhenoShape obtained by PhenoShape function
    """

    inds = np.isnan(ynew)  # check if array has NaN values
    inds2 = np.isnan(y)
    if inds.any():  # check is all values are NaN
        return np.nan
    else:
        if inds2.any():
            y = _fillNaN(y)
        ypred2 = np.interp(x, xnew, ynew)

        return np.sqrt(mean_squared_error(ypred2, y))

# ---------------------------------------------------------------------------#


def _RMSE2(phen, dstack, dates, nan_replace, nGS):
    """
    Apply _RMSE funciton to a spatial 3D arrays
    """
    # original data - dstack
    xarray = xr.DataArray(dstack)
    xarray.coords['dim_0'] = dates.dt.dayofyear
    # sort basds according to day-of-the-year
    xarray = xarray.sortby('dim_0')
    if nan_replace is not None:
        xarray = xarray.where(xarray.values != nan_replace)
    # xarray.values =  np.apply_along_axis(_fillNaN, 0, xarray.values)
    x = xarray.dim_0.values
    y = xarray.values

    xnew = np.linspace(np.min(x), np.max(x), nGS, dtype='int16')
    # change shape from 3D to 2D matrix
    y2 = y.reshape(y.shape[0], (y.shape[1] * y.shape[2]))
    ynew = phen.reshape(phen.shape[0], (y.shape[1] * y.shape[2]))

    rmse = np.zeros((y.shape[1] * y.shape[2]))
    for i in tqdm(range(y.shape[1] * y.shape[2])):
        # print(i)
        rmse[i] = _RMSE(x, y2[:, i], xnew, ynew[:, i])

    # reshape from 2D to 3D
    return rmse.reshape(1, phen.shape[1], y.shape[2])

# ---------------------------------------------------------------------------#


def _replaceElements(arr):
    '''
    Replace monotonic vector values to avoid
    interpolation errors
    '''
    s = []
    for i in range(len(arr)):
        # check whether the element
        # is repeated or not
        if arr[i] not in s:
            s.append(arr[i])
        else:
            # find the next greatest element
            for j in range(arr[i] + 1, sys.maxsize):
                if j not in s:
                    arr[i] = j
                    s.append(j)
                    break

# ---------------------------------------------------------------------------#


def _fillNaN(x):
    # Fill NaN data by linear interpolation
    mask = np.isnan(x) 
    x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x

# ---------------------------------------------------------------------------#
    
def _moving_average(a, n=3) :
    out = np.convolve(a, np.ones(n), 'valid') / n    
    return np.concatenate([ a[:np.int(n/2)], out, a[-np.int(n/2):] ]) # add values of tail

# ---------------------------------------------------------------------------#
