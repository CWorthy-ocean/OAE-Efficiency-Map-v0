import os
import shutil
from glob import glob
from subprocess import check_call

import textwrap

import cftime
import numpy as np
import xarray as xr

import pop_tools



def pop_add_cyclic(ds):
    
    nj = ds.TLAT.shape[0]
    ni = ds.TLONG.shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon)    
    lon  = np.concatenate((tlon, tlon + 360.), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

    TLAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    TLONG = xr.DataArray(lon, dims=('nlat', 'nlon'))
    
    dso = xr.Dataset({'TLAT': TLAT, 'TLONG': TLONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)       
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)


    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
                
            
    return dso

def add_coast_mask(grid):
    
    '''Add coastline mask to POP grid
    
    Find coastline coordinates in POP grid: look for sharp gradient of KMT, first make 0 KMT to be 0.0001
    
    Key: calculate gradients between lon = 319 and nlon = 0 
    '''
    
    ocean_mask = (grid.KMT > 0).values

    # Replace zeros with 1e-4 using np.where()
    kmt = grid.KMT.values
    kmt_ = np.where(kmt == 0, 1e5, kmt)

    # Calculate the gradients in both dimensions
    gradient_x, gradient_y = np.gradient(kmt_)

    # Compute the magnitude of the gradients
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # for nlon = 319, and nlon = 0, at the edge of POP grid
    grad_x, grad_y = np.gradient(kmt_[:, [0, 319]])
    gradient_edge = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Find the indices where the gradient magnitude exceeds the threshold
    coast_indices = np.where(gradient_magnitude > 40000)
    coast_indices_edge = np.where(gradient_edge > 40000)

    # conbime the 2 arrays
    coast_indices_edge_nlon = coast_indices_edge[1]
    coast_indices_edge_nlon[np.where(coast_indices_edge_nlon == 1)] = 319  # the 2nd column is 319
    whole_nlat = np.concatenate((coast_indices[0], coast_indices_edge[0]))
    whole_nlon = np.concatenate((coast_indices[1], coast_indices_edge_nlon))
    coast_indices = (whole_nlat, whole_nlon)

    # coastline mask in POP grid
    coast_mask = np.zeros(kmt.shape)
    for i in range(coast_indices[0].shape[0]):
        j = coast_indices[0][i] # nlat
        k = coast_indices[1][i] # nlon

        coast_mask[j,k] = 1

    coast_mask = coast_mask*ocean_mask
    grid['coast_mask'] = xr.DataArray(data=coast_mask, dims=['nlat', 'nlon'])
    
    return grid, coast_indices


def open_dataset(case, stream='pop.h', from_campaign=False):
    """access data from a case"""
    grid = pop_tools.get_grid('POP_gx1v7')


    #archive_root = f'{config.dir_scratch}/archive/{case}'
    
    
    rename_underscore2_vars = False
    if stream == 'pop.h':
        subdir = 'ocn/hist'
        datestr_glob = '????-??'
    elif stream == 'pop.h.ecosys.nday1':
        subdir = 'ocn/hist'
        datestr_glob = '????-??-??'
        rename_underscore2_vars = True
    else:
        raise ValueError(f'access to stream: "{stream}" not defined')
        
    # ### Added by Mengyang on Nov 1, 2023 to read files from run directories.
    # if from_run_dire == True:   
    #     archive_root = f'{config.dir_scratch}/{case}'
    #     subdir = 'run'

    ### Added by Mengyang on May 14, 2024 to read files from campaign directories.
    if from_campaign == True:   
        archive_root = f'/glade/campaign/cesm/development/bgcwg/projects/OAE-Global-Efficiency/Mengyang_Global_OAE_Experiments/archive/{case}'


    glob_str = f'{archive_root}/{subdir}/{case}.{stream}.{datestr_glob}.nc'
    files = sorted(glob(glob_str))
    assert files, f'no files found.\nglob string: {glob_str}'

    def preprocess(ds):
        return ds.set_coords(["KMT", "TAREA"]).reset_coords(["ULONG", "ULAT"], drop=True)

    ds = xr.open_mfdataset(
        files,
        coords="minimal",
        combine="by_coords",
        compat="override",
        preprocess=preprocess,
        decode_times=False,
        parallel=True,
        data_vars = ['time_bound','dz', 'REGION_MASK','UAREA','CO3', 'CO3_ALT_CO2','ECOSYS_XKW','pCO2SURF', 'FG_CO2','pCO2SURF_ALT_CO2','FG_ALT_CO2','ALK_FLUX','DIC','DIC_ALT_CO2','ALK','ALK_ALT_CO2',]
    )

    tb_var = ds.time.attrs["bounds"]
    time_units = ds.time.units
    calendar = ds.time.calendar

    ds['time'] = cftime.num2date(
        ds[tb_var].mean('d2'),
        units=time_units,
        calendar=calendar,
    )
    ds.time.encoding.update(dict(
        calendar=calendar,
        units=time_units,
    ))

    d2 = ds[tb_var].dims[-1]
    ds['time_delta'] = ds[tb_var].diff(d2).squeeze()
    ds = ds.set_coords('time_delta')

    ds['TLONG'] = grid.TLONG
    ds['TLAT'] = grid.TLAT
    ds['KMT'] = ds.KMT.fillna(0)

    if rename_underscore2_vars:
        rename_dict = dict()
        for v in ds.data_vars:
            if v[-2:] == '_2':
                rename_dict[v] = v[:-2]
        ds = ds.rename(rename_dict)

    return ds

def process(da):
    
    da['ALK_excess'] = da.ALK - da.ALK_ALT_CO2
    da['DIC_excess'] = da.DIC - da.DIC_ALT_CO2
    da['pCO2_SURF_excess'] = da.pCO2SURF - da.pCO2SURF_ALT_CO2
    da['DpCO2_excess'] = da.DpCO2 - da.DpCO2_ALT_CO2
    da['FG_CO2_excess'] = da.FG_CO2 - da.FG_ALT_CO2
    da['PH_excess'] = da.PH - da.PH_ALT_CO2
    
    
    return da