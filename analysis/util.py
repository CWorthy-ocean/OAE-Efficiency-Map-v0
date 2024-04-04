import numpy as np
import xarray as xr


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
