import numpy as np
import xarray as xr


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    lon1, lat1 := scalars
    lon2, lat2 := 1D arrays
    """

    Re = 6378.137

    # convert decimal degrees to radians
    deg2rad = np.pi / 180.
    lon1 = np.array(lon1) * deg2rad
    lat1 = np.array(lat1) * deg2rad
    lon2 = np.array(lon2) * deg2rad
    lat2 = np.array(lat2) * deg2rad

    if lon2.shape:
        N = lon2.shape[0]
        lon1 = np.repeat(lon1, N)
        lat1 = np.repeat(lat1, N)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.)**2. + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.)**2.
    c = 2. * np.arcsin(np.sqrt(a))
    km = Re * c
    return km


def pop_add_cyclic(ds):

    ni = ds.TLONG.shape[1]

    xL = int(ni / 2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data

    tlon = np.where(np.greater_equal(tlon, min(tlon[:, 0])), tlon - 360.0, tlon)
    lon = np.concatenate((tlon, tlon + 360.0), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.0
    lon = lon - 360.0

    lon = np.hstack((lon, lon[:, 0:1] + 360.0))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.0

    # -- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    # -- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:, 0:1]))

    TLAT = xr.DataArray(lat, dims=("nlat", "nlon"))
    TLONG = xr.DataArray(lon, dims=("nlat", "nlon"))

    dso = xr.Dataset({"TLAT": TLAT, "TLONG": TLONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ["TLAT", "TLONG"]]
    for v in varlist:
        v_dims = ds[v].dims
        if not ("nlat" in v_dims and "nlon" in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {"nlat", "nlon"}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index("nlon")
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)
            dso[v] = xr.DataArray(
                field, dims=other_dims + ("nlat", "nlon"), attrs=ds[v].attrs
            )

    # copy coords
    for v, da in ds.coords.items():
        if not ("nlat" in da.dims and "nlon" in da.dims):
            dso = dso.assign_coords(**{v: da})

    return dso
