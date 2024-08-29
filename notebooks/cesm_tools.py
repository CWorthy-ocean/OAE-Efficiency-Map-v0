import os
import sys
import subprocess

import matplotlib.pyplot as plt

import numpy as np
from netCDF4 import default_fillvals


def to_netcdf_clean(dset, path, netcdf3=True, **kwargs):
    """wrap to_netcdf method to circumvent some xarray shortcomings"""

    dset = dset.copy()

    # ensure _FillValues are not added to coordinates
    for v in dset.coords:
        dset[v].encoding["_FillValue"] = None

    for v in dset.data_vars:

        # if dtype has been set explicitly in encoding, obey
        if "dtype" in dset[v].encoding:
            if dset[v].encoding["dtype"] == np.float64:
                fv = default_fillvals["f8"]
            elif dset[v].encoding["dtype"] == np.float32:
                fv = default_fillvals["f4"]
            elif dset[v].encoding["dtype"] in [np.int32]:
                fv = default_fillvals["i4"]

            if '_FillValue' not in dset[v].encoding:
                dset[v].encoding["_FillValue"] = fv

        # otherwise, default to single precision output
        elif dset[v].dtype in [np.float32, np.float64]:
            fv = default_fillvals["f4"]
            dset[v].encoding["dtype"] = np.float32

        elif dset[v].dtype in [np.int32, np.int64]:
            fv = default_fillvals["i4"]
            dset[v].encoding["dtype"] = np.int32

        elif dset[v].dtype == object:
            pass

        else:
            warnings.warn(f"warning: unrecognized dtype for {v}: {dset[v].dtype}")

        if "_FillValue" not in dset[v].encoding:
            dset[v].encoding["_FillValue"] = None

    sys.stderr.flush()

    print("-" * 30)
    print(f"Writing {path}")
    dset.to_netcdf(path, **kwargs)
    print("")
    if netcdf3:
        ncks_fl_fmt64bit(path)

    dumps = subprocess.check_output(["ncdump", "-h", path]).strip().decode("utf-8")
    print(dumps)
    dumps = subprocess.check_output(["ncdump", "-k", path]).strip().decode("utf-8")
    print(f"format: {dumps}")
    print("-" * 30)


def ncks_fl_fmt64bit(file_in, file_out=None):
    """
    Converts file to netCDF-3 64bit by calling:
      ncks --fl_fmt=64bit  file_in file_out

    Parameter
    ---------
    file : str
      The file to convert.
    """

    if file_out is None:
        file_out = file_in

    ncks_cmd = " ".join(["ncks", "-O", "--fl_fmt=64bit", file_in, file_out])
    cmd = " && ".join(["module load nco", ncks_cmd])

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(stdout.decode("UTF-8"))
        print(stderr.decode("UTF-8"))
        raise


def visualize_mask(mask3d):
    nregion = len(mask3d.region)

    # mask out land
    # mask3d = mask3d.where(KMT > 0)

    # visualize the regions
    ncol = int(np.sqrt(nregion))
    nrow = int(nregion / ncol) + min(1, nregion % ncol)

    fig, ax = plt.subplots(
        nrow, ncol, figsize=(4 * ncol, 3 * nrow), 
        constrained_layout=True,
    )

    for i, region in enumerate(mask3d.region.values):
        plt.axes(ax.ravel()[i])
        mask3d.sel(region=region).plot()

    # delete the unused axes
    for i in range(nregion, ncol * nrow):
        fig.delaxes(ax.ravel()[i])

    fig.suptitle(f'Mask name = {mask3d.mask_name}', fontsize=16)
    
    
    
def code_checkout(remote, coderoot, tag):
    """Checkout code for CESM
    If sandbox exists, check that the right tag has been checked-out.

    Otherwise, download the code, checkout the tag and run manage_externals.
    """

    sandbox = os.path.join(coderoot, tag)

    if os.path.exists(sandbox):
        print(f"Check for right tag: {sandbox}")
        p = subprocess.Popen("git status", shell=True, cwd=sandbox, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        stdout = stdout.decode("UTF-8")
        stderr = stderr.decode("UTF-8")
        print(stdout)
        print(stderr)
        if tag not in stdout.split("\n")[0]:
            raise ValueError("tag does not match")

    else:
        if not os.path.exists(coderoot):
            os.makedirs(coderoot)

        # clone the repo
        p = subprocess.Popen(
            f"git clone {remote} {tag}",
            shell=True,
            cwd=coderoot,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        if p.returncode != 0:
            raise Exception("git error")

        # check out the right tag
        assert os.path.exists(sandbox)
        p = subprocess.Popen(f"git checkout {tag}", shell=True, cwd=sandbox)
        stdout, stderr = p.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        if p.returncode != 0:
            raise Exception("git error")

        # check out externals
        p = subprocess.Popen("./manage_externals/checkout_externals -v", shell=True, cwd=sandbox)
        stdout, stderr = p.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        if p.returncode != 0:
            raise Exception("git error")

    return sandbox    
    
    
