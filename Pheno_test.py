
from dask.distributed import Client

client = Client("tcp://10.0.41.65:36787")
client

# standard libraries
%matplotlib inline

import datacube
import matplotlib.pyplot as plt
import numpy as np
import dask as da
import pandas as pd
import sys
import xarray as xr
import rioxarray as rio
import datetime as dt
import os
# builtin funcitons from open data cube
sys.path.append('../Scripts')
from deafrica_datahandling import load_ard
from deafrica_bandindices import calculate_indices
from deafrica_plotting import display_map, rgb
import deafrica_temporal_statistics as ts
# builtin functions from UAI
import Phenology as phen


dc = datacube.Datacube(app="Pheno_test")

dc.list_products().name

# Set the vegetation proxy to use
veg_proxy = 'NDVI'
dates = ('2018-01', '2020-12')
inProduct = 'usgs_espa_ls8c1_sr'
collection = 'c1' # 'c1' (for USGS Collection 1);'c2' (for USGS Collection 2) and 's2' (for Sentinel-2)
resolution = 30
central_lat, central_lon = -35.979288, -72.598012
buffer = 0.005
# Phenopy parameters
nGS = 46
nan_replace = None
rollWindow=5
interpolType='linear'

# --------------------------------------------------------------------------------------
## get phenological plot using PhenoPy
#phen.PhenoPlot(dc=dc, X=central_lat, Y=central_lon, products=inProduct, dates=dates, 
#               veg_proxy=veg_proxy, resolution=resolution)

# Compute the bounding box for the study area
study_area_lat = (central_lat - buffer, central_lat + buffer)
study_area_lon = (central_lon - buffer, central_lon + buffer)

#display_map(x=study_area_lon, y=study_area_lat)

# datos de entrada
query = {
    #"measurements": ['B04','B08','SCL'],
    "x": study_area_lon,
    "y": study_area_lat,
    "time": dates,
    "output_crs": "EPSG:32719",
    "resolution": (-resolution, resolution),
    "dask_chunks": {"time": 1},
    "group_by":"solar_day",
}

# Show area in a map
#display_map(x=study_area_lon, y=study_area_lat)

# Load available data 
ds = load_ard(
    dc=dc,
    products=[inProduct],
    **query,
)
ds.update(ds.assign(doy=ds.time.dt.dayofyear,
                   year=ds.time.dt.year))
print(ds)

# -----------------------
### Phenology
# ----------------------

#-----------------------
# PhenoShapes
phenol = phen.PhenoShape(ds=ds, veg_proxy=veg_proxy, collection=collection, rollWindow=5)
print(phenol)

# plot RGB ant example of phenoshape
from rasterio.plot import show
plt.plot(phenol.isel(x=3, y=0))
image_norm = (phenol.values - phenol.values.min()) / (phenol.values.max() - phenol.values.min())
show([image_norm[[4,20,44]]])

# save to disk
import rasterio

profile = {}
profile.update(
    transform = ds.affine,
    crs = rasterio.crs.CRS.from_epsg(ds.spatial_ref.values),
    tiled = True,
    blockxsize = 256,
    blockysize = 256,
    driver = 'GTiff',
    width = ds.dims['x'],
    height = ds.dims['y'],
    dtype = rasterio.float64,
    count = phenol.shape[0],
    compress = 'lzw')

with rasterio.open('temp/test_phen_DC.tif', 'w', **profile) as dst_dataset:
     dst_dataset.write(phenol)

        
#-----------------------
# LSP



type(phenol)


def PhenoLSP(inData, outData, doy, nGS=46, phentype=1, n_phen=10, n_jobs=4,
             chuckSize=256):
    """
    Obtain land surfurface phenology metrics for a PhenoShape product

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

    _cal_LSP, nGS=nGS, phentype=phentype, doy=doy,
                      n_phen=n_phen, num=len(bandNames))
     





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
    return np.apply_along_axis(_getLSPmetrics, 0, dstack, xnew, nGS, num, phentype)






## get indices
ds = calculate_indices(ds, index=veg_proxy, collection=collection)
print(ds)

# plot example map and time series
ds[veg_proxy].isel(time=0).plot()
ds[veg_proxy].isel(x=3, y=0).plot()


# rearrange DOY values according to phenology for the south hemisphere
# start =  first of June (DOY=152)
south_doy = np.append(np.linspace(202,365,163, dtype='int16'), np.linspace(1,201,201,dtype='int16'))
doy_data = ds.doy.values

## find location of real doy values in the south_doy data
doy_where =[]
for x in np.nditer(doy_data):
    doy_where.append( np.where(south_doy == x)[0][0] )
doy_where = np.array(doy_where)

# add south_doy to xarray and sort accordingly
ds = ds.update(ds.assign(south_doy=doy_where))
ds2 = ds.sortby(ds.south_doy)

# get example data
vi1d = ds2[veg_proxy].isel(x=3, y=0)
print(vi1d)

# plot index
vi1d.plot.line('b-^', figsize=(11,4))

# get phenological shape
phenol = _getPheno0(y=y, doy=doy_where, interpolType='linear', nan_replace=None, rollWindow=5, nGS=nGS)

# doy of the predicted phenological shape
xnew = np.linspace(np.min(doy_data), np.max(doy_data), nGS, dtype='int16')

# plot
x = doy_where
y = vi1d.values
plt.scatter(x, y, marker='o', c=ds2.time.dt.year.values)
plt.plot(xnew, phenol)

TSS = pd.concat([pd.DataFrame(ds2.time.values),
                 pd.DataFrame(doy_where),
                 pd.DataFrame(ds2.time.dt.year),
                 pd.DataFrame(vi1d)], axis=1)
TSS.columns = ['dates', 'doy', 'year', 'VI']

groups = TSS.groupby('year')

rmse = _RMSE(TSS.doy.values, TSS.VI.values, xnew, phen)#.round(2)
minn = np.nanmin(valuesTSS)
maxx = np.nanmax(valuesTSS)
nRMSE = ((rmse/(maxx-minn))*100).round(2)

for name, group in groups:
    plt.plot(group.doy, group.VI, marker='o', linestyle='', ms=10, label=name)
plt.plot(xnew, phenol, '-', color='black')
plt.legend(prop={'size': legendsize})

