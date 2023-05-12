import os
import shutil
from glob import glob
from subprocess import check_call

import textwrap

import cftime
import numpy as np
import xarray as xr

import pop_tools

import config

scriptroot = os.path.dirname(os.path.realpath(__file__))

caseroot_root = f"{config.dir_project_root}/cesm-cases"
os.makedirs(caseroot_root, exist_ok=True)



coderoot = "/glade/work/klindsay/cesm2_tags/cesm2.2.0"
refcase = "g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005"
refcase_root = "/glade/work/klindsay/cesm22_cases/SMYLE"
refcaserest_root = "/glade/work/mclong/SMYLE-FOSI/rest"


nmolcm2s_to_molm2yr = 1.0e-9 * 1.0e4 * 86400.0 * 365.0
nmolcm2_to_molm2 = 1.0e-9 * 1.0e4


def create_clone(
    case,
    alk_forcing_file,
    refdate="0347-01-01",
    stop_n=4,
    stop_option="nyear",
    job_queue="economy",
    wallclock="12:00:00",
    resubmit=None,
    clobber=False,
    submit=False,
):
    caseroot = f"{caseroot_root}/{case}"
    if os.path.exists(caseroot) and not clobber:
        print(f"Case {case} exists; skipping.")
        return

    rundir = f"{config.dir_scratch}/{case}"
    archive_root = f"{config.dir_scratch}/archive/{case}"

    check_call(["rm", "-fr", caseroot])
    check_call(["rm", "-fr", rundir])
    check_call(["rm", "-fr", archive_root])

    check_call(
        [
            "./create_clone",
            "--clone",
            f"{refcase_root}/{refcase}",
            "--case",
            caseroot,
            "--project",
            config.account,
            "--cime-output-root",
            config.dir_scratch,
        ],
        cwd=f"{coderoot}/cime/scripts",
    )

    # copy source mods
    src_pop_files = glob(f"{scriptroot}/SourceMods-OAE/src.pop/*")
    for src in src_pop_files:
        src_basename = os.path.basename(src)
        if src_basename == "diagnostics_latest.yaml":
            check_call(
                [
                    f"{coderoot}/components/pop/externals/MARBL/MARBL_tools/./yaml_to_json.py",
                    "-y",
                    f"{src}",
                    "-o",
                    f"{caseroot}/SourceMods/src.pop",
                ]
            )
        else:
            dst = f"{caseroot}/SourceMods/src.pop/{src_basename}"
            shutil.copyfile(src, dst)

    # xml settings
    def xmlchange(arg):
        """call xmlchange"""
        check_call(["./xmlchange", arg], cwd=caseroot)

    xmlchange("RUN_TYPE=branch")
    xmlchange(f"RUN_STARTDATE={refdate}")
    xmlchange(f"RUN_REFCASE={refcase}")
    xmlchange(f"RUN_REFDATE={refdate}")

    # copy restarts
    check_call(
        f"cp {refcaserest_root}/{refdate}-00000/* {config.dir_scratch}/{case}/run/.",
        shell=True,
    )

    # set ALT_CO2 tracers to CO2 tracers
    check_call(
        ["./set-alt-co2.sh", f"{rundir}/run/{refcase}.pop.r.{refdate}-00000.nc"],
        cwd=scriptroot,
    )

    check_call(["./case.setup"], cwd=caseroot)

    # namelist
    user_nl = dict()

    user_nl["pop"] = textwrap.dedent(
        f"""\
    lecosys_tavg_alt_co2 = .true.
    atm_alt_co2_opt = 'drv_diag'
    lalk_forcing_apply_file_flux = .true.
    alk_forcing_shr_stream_year_first = 1999
    alk_forcing_shr_stream_year_last = 2019
    alk_forcing_shr_stream_year_align = 347
    alk_forcing_shr_stream_file = '{alk_forcing_file}'
    alk_forcing_shr_stream_scale_factor = 1.0e5 ! convert from mol/m^2/s to nmol/cm^2/s
    """
    )

    for key, nl in user_nl.items():
        user_nl_file = f"{caseroot}/user_nl_{key}"
        with open(user_nl_file, "a") as fid:
            fid.write(user_nl[key])

    check_call(["./preview_namelists"], cwd=caseroot)

    check_call(["qcmd", "--", "./case.build"], cwd=caseroot)

    if resubmit is not None:
        xmlchange(f"RESUBMIT={resubmit}")
    
    if submit:
        xmlchange(f"STOP_N={stop_n}")
        xmlchange(f"JOB_QUEUE={job_queue}")
        xmlchange(f"STOP_OPTION={stop_option}")
        xmlchange(f"JOB_WALLCLOCK_TIME={wallclock}")
        check_call(["./case.submit"], cwd=caseroot)


def open_dataset(case, stream):
    """access data from a case"""
    grid = pop_tools.get_grid('POP_gx1v7')

    archive_root = f'{config.dir_scratch}/archive/{case}'

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


def compute_additional_CO2_flux(ds):
    """compute the additional CO2 flux"""
    with xr.set_options(keep_attrs=True):
        flux_effect = (-1.0) * (ds.FG_CO2 - ds.FG_ALT_CO2).where(ds.KMT > 0)
        flux_effect *= nmolcm2s_to_molm2yr
        flux_effect.attrs['units'] = 'mol m$^{-2}$ yr$^{-1}$'
        flux_effect.attrs['sign_convention'] = 'postive up'
        flux_effect['area_m2'] = ds.TAREA * 1e-4
        flux_effect = flux_effect.reset_coords('TAREA', drop=True)
    return flux_effect


def compute_time_cumulative_integral(da, convert_time=1.0):
    """integrate a DataArray in time"""
    with xr.set_options(keep_attrs=True):
        dao = da.weighted(da.time_delta * convert_time).sum('time')
    dao.attrs['units'] = dao.attrs['units'].replace('yr$^{-1}$', '')
    return dao


def compute_global_ts(da):
    """integrate DataArray globally"""
    with xr.set_options(keep_attrs=True):
        dao = (da * da.area_m2).sum(['nlat', 'nlon'])
    dao.attrs['units'] = dao.attrs['units'].replace('m$^{-2}$', '')
    return dao


def compute_additional_DIC_global_ts(ds):
    """return the globally-integrated, time-integrated flux"""
    
    add_co2_ts = compute_global_ts(
        compute_additional_CO2_flux(ds)
    )
    
    # compute cumulative integral in time
    dt = add_co2_ts.time_delta / 365 # time_delta in days, convert to years
    with xr.set_options(keep_attrs=True):
        dao = (-1.0) * (add_co2_ts * dt).cumsum('time')
    dao.attrs['long_name'] = 'Change in DIC inventory'
    dao.attrs['units'] = dao.attrs['units'].replace('yr$^{-1}$', '').strip()
    return dao

