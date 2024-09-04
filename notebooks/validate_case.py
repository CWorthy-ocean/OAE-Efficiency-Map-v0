import os
from glob import glob
import numpy as np
import xarray as xr

import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

import project

# paths
fpath_smyle = (
    "/global/cfs/projectdirs/m4746/Datasets/SMYLE-FOSI/ocn/proc/tseries/month_1"
)

dir_out = f"{project.dir_data}/Case-Validation"
os.makedirs(dir_out, exist_ok=True)

archive_root = f"{project.dir_data}/archive"


# start/end dates
start_date = "0347-01-01"
end_date = "0361-12-31"

variables = [
    "TEMP",
    "SALT",
    "UVEL",
    "VVEL",
    "WVEL",
    "PO4",
    "NO3",
    "SiO3",
    "NH4",
    "Fe",
    "Lig",
    "O2",
    "DIC",
    "DIC_ALT_CO2",
    "ALK",
    "ALK_ALT_CO2",
    "DOC",
    "DON",
    "DOP",
    "DOPr",
    "DONr",
    "DOCr",
    "zooC",
    "spChl",
    "spC",
    "spP",
    "spFe",
    "spCaCO3",
    "diatChl",
    "diatC",
    "diatP",
    "diatFe",
    "diatSi",
    "diazChl",
    "diazC",
    "diazP",
    "diazFe",
]


def zarr_validation_data(case):
    """return the filename of Zarr store with validation data"""
    return f"{dir_out}/{case}.diff.zarr"


def validator(case, is_oae_run):
    """compute RMSE difference and map of difference at last time index
    between OAE case and control"""

    variable_dict = dict()
    if is_oae_run:
        variable_dict["DIC_ALT_CO2"] = "DIC"
        variable_dict["ALK_ALT_CO2"] = "ALK"
    else:
        variable_dict = {v: v for v in variables}

    chunk_spec = {"nlat": -1, "nlon": -1, "z_t": 60}

    # load case data
    time_case = xr.cftime_range(start_date, end_date, freq="ME", calendar="noleap")
    files = sorted(
        glob(
            f"{archive_root}/{case}/ocn/hist/{case}.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"
        )
    )
    if not files:
        return 
    
    assert len(files) == len(
        time_case
    ), f"{len(files)} found -- expected {len(time_case)}"

    ds = xr.open_mfdataset(
        files,
        decode_times=False,
        combine="by_coords",
        coords="minimal",
        data_vars="minimal",
        compat="override",
        drop_variables=[
            "transport_regions",
            "transport_components",
            "moc_components",
        ],  # xarray can't merge these for some reason
        chunks=chunk_spec,
    )
    ds = ds.assign_coords({"time": time_case})
    
    # set time axis for control
    time_ctrl = xr.cftime_range(
        "0306-01-01", "0368-12-31", freq="ME", calendar="noleap"
    )
    ndx0 = np.where(time_case[0] == time_ctrl)[0].item()
    tndx = np.arange(ndx0, ndx0 + len(time_case), 1)

    # control details
    smyle_case = "g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005"
    stream = "pop.h"
    datestr = "030601-036812"

    # loop over variables and compute difference metrics
    ds_out = xr.Dataset()
    for v_case, v_ctrl in variable_dict.items():
        print(f"working on {v_case}...")

        # open control dataset
        file_in = f"{fpath_smyle}/{smyle_case}.{stream}.{v_ctrl}.{datestr}.nc"
        assert os.path.exists(file_in)
        
        ds_ctrl = xr.open_dataset(file_in, decode_times=False, chunks=chunk_spec)
        assert len(ds_ctrl.time) == len(time_ctrl), "mismatch in control run time axis"
        ds_ctrl = ds_ctrl.assign_coords({"time": time_ctrl})
        ds_ctrl = ds_ctrl.isel(time=tndx)
            
        # identify correct coordinates
        if "z_t" in ds_ctrl[v_ctrl].dims:
            isel_timeseries = dict(z_t=0, nlat=0, nlon=0)
            isel_slab = dict(z_t=0, time=-1)
            z_dim = "z_t"

        elif "z_w_top" in ds_ctrl[v_ctrl].dims:
            isel_timeseries = dict(z_w_top=9, nlat=0, nlon=0)
            isel_slab = dict(z_w_top=9, time=-1)
            z_dim = "z_w_top"

        elif "z_t_150m" in ds_ctrl[v_ctrl].dims:
            isel_timeseries = dict(z_t_150m=0, nlat=0, nlon=0)
            isel_slab = dict(z_t_150m=0, time=-1)
            z_dim = "z_t_150m"

        # initialize variables
        n = ds[v_case].isel(time=0).notnull().sum()
        ds_out[f"{v_case}_rmse"] = xr.full_like(
            ds[v_case].isel(**isel_timeseries), fill_value=np.nan
        )
        ds_out[f"{v_case}_diff"] = xr.full_like(
            ds[v_case].isel(**isel_slab), fill_value=np.nan
        )

        # compute metrics
        with xr.set_options(arithmetic_join="exact"):
            ds_out[f"{v_case}_rmse"].data = np.sqrt(
                ((ds[v_case] - ds_ctrl[v_ctrl]) ** 2 / n).sum(
                    [
                        z_dim,
                        "nlat",
                        "nlon",
                    ]
                )
            )
            ds_out[f"{v_case}_diff"].data = (ds[v_case] - ds_ctrl[v_ctrl]).isel(**isel_slab)

    print(f"writing {zarr_validation_data(case)}")
    ds_out.to_zarr(
        zarr_validation_data(case),
        mode="w",
        consolidated=True,
    )
