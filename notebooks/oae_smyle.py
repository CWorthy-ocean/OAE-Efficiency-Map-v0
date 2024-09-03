import os
import shutil
from glob import glob
from subprocess import check_call

import click

import textwrap

import cftime
import numpy as np
import xarray as xr

import pop_tools
import cesm_tools
import project


scriptroot = os.path.dirname(os.path.realpath(__file__))

caseroot_root = f"{project.dir_project_root}/cesm-cases"
os.makedirs(caseroot_root, exist_ok=True)

"""
Details of g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005
FOSI (forced ocean--sea-ice) simulation. 

Details of this simulation are as follows:
    - CASE: g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005
    - CASEROOT: /glade/work/klindsay/cesm22_cases/SMYLE/$CASE
    - SRCROOT: /glade/work/klindsay/cesm2_tags/cesm2.2.0/
    - forcing:  JRA55-do v1.4, 1958-2018 (==> 61-year cycle)
    - spinup: 6 cycles (==> simyears 0001-0366)
    - years used for SMYLE ICs:   0306 (1958) - 0366 (2018)

* Modifications from CMIP6-OMIP2 were made to improve sea-ice and ocean BGC fields:
    -Use of full 1958-2018 (61-year) forcing cycle during spinup
    -Use of strong under-ice restoring to model prognostic freezing temperature (TFZ)
    -Reduced deep isopycnal mixing (kappa_isop_deep = 0.1, instead of CESM2-default of 0.2)
    -Enhanced sea ice albedoes:
            r_snw = 1.6
            dt_mlt = 0.5
            rsnw_mlt = 1000.
"""
refcase = "g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005"
refcaserest_root = f"{project.dir_project_root}/data/{refcase}/rest"
cesm_tag = "release-cesm2.1.5"
compset = "OMIP_DATM%JRA-1p4-2018_SLND_CICE_POP2%ECO_DROF%JRA-1p4-2018_SGLC_WW3_SIAC_SESP"
res = "TL319_g17"


nmolcm2s_to_molm2yr = 1.0e-9 * 1.0e4 * 86400.0 * 365.0
nmolcm2_to_molm2 = 1.0e-9 * 1.0e4


def create_oae_case(
    case,
    alk_forcing_file,
    refdate="0347-01-01",
    stop_n=4,
    stop_option="nyear",
    wallclock="12:00:00",
    resubmit=0,
    clobber=False,
    submit=False,
    curtail_output=True,
    queue="regular",
):
    caseroot = f"{project.dir_caseroot_root}/{case}"
    assert not os.path.exists(caseroot) or clobber, f"Case {case} exists; caseroot:\n{caseroot}\n"
        
    rundir = f"{project.dir_scratch}/{case}"
    archive_root = f"{project.dir_scratch}/archive/{case}"
    
    check_call(["rm", "-fr", caseroot])
    check_call(["rm", "-fr", rundir])
    check_call(["rm", "-fr", archive_root])

    check_call(
        " ".join([
            "module load python",
            "&&",      
            "./create_newcase",
            "--compset", compset,
            "--case", caseroot,
            "--res", res,
            "--machine", project.mach,
            "--compiler", "intel",            
            "--project", project.account,
            "--queue", queue,
            "--walltime", wallclock,
            "--run-unsupported"]),
        shell=True,
        cwd=f"{project.coderoot}/cime/scripts",
    )

    def xmlchange(arg, force=False):
        """call xmlchange"""
        check_call(f"module load python && ./xmlchange {arg}", cwd=caseroot, shell=True)
    
    xmlchange("MAX_TASKS_PER_NODE=128")
    xmlchange("MAX_MPITASKS_PER_NODE=128")

    xmlchange("NTASKS_ATM=72")
    xmlchange("NTASKS_CPL=72")
    xmlchange("NTASKS_WAV=72")
    xmlchange("NTASKS_GLC=72")
    xmlchange("NTASKS_ICE=72")
    xmlchange("NTASKS_ROF=72")
    xmlchange("NTASKS_LND=72")
    xmlchange("NTASKS_ESP=72")
    xmlchange("NTASKS_IAC=72")

    xmlchange("NTASKS_OCN=751")
    xmlchange("ROOTPE_OCN=72")

    xmlchange("CICE_BLCKX=16")
    xmlchange("CICE_BLCKY=16")
    xmlchange("CICE_MXBLCKS=7")
    xmlchange("CICE_DECOMPTYPE='sectrobin'")
    xmlchange("CICE_DECOMPSETTING='square-ice'")

    xmlchange("OCN_TRACER_MODULES='iage ecosys'")
    xmlchange("DATM_PRESAERO='clim_1850'")

    xmlchange("POP_AUTO_DECOMP=FALSE")
    xmlchange("POP_BLCKX=9")
    xmlchange("POP_BLCKY=16")
    xmlchange("POP_NX_BLOCKS=36")
    xmlchange("POP_NY_BLOCKS=24")
    xmlchange("POP_MXBLCKS=1")
    xmlchange("POP_DECOMPTYPE='spacecurve'")    
    
    # refcase SourceMods
    check_call(
        f"cp -vr {scriptroot}/CESM-RefCase/{refcase}/SourceMods/* {caseroot}/SourceMods",
        shell=True,
    )
    
    # list SourceMod files
    src_pop_files = glob(f"{scriptroot}/SourceMods-OAE/src.pop/*")
    if curtail_output:
        src_pop_files.extend(
            glob(f"{scriptroot}/SourceMods-OAE/src.pop.curtail-output/*")
        )
    else:
        src_pop_files.extend(
            glob(f"{scriptroot}/SourceMods-OAE/src.pop.full-output/*")
        )

    # copy SourceMod files
    for src in src_pop_files:
        src_basename = os.path.basename(src)
        if src_basename == "diagnostics_latest.yaml":
            check_call(
                " ".join([
                    "module load python", 
                    "&&",
                    f"{project.coderoot}/components/pop/externals/MARBL/MARBL_tools/./yaml_to_json.py",
                    "-y",
                    f"{src}",
                    "-o",
                    f"{caseroot}/SourceMods/src.pop",
                ]),
                shell=True
            )
        else:
            dst = f"{caseroot}/SourceMods/src.pop/{src_basename}"
            shutil.copyfile(src, dst)
            if '.csh' in src_basename: 
                check_call(['chmod', '+x', dst])
            
    xmlchange(f"DIN_LOC_ROOT={project.cesm_inputdata}")
    xmlchange(f"DOUT_S_ROOT='{project.dir_data}/archive/$CASE'")
        
    xmlchange(f"RUN_TYPE=branch")
    xmlchange(f"RUN_STARTDATE={refdate}")
    xmlchange(f"RUN_REFCASE={refcase}")
    xmlchange(f"RUN_REFDATE={refdate}")

    xmlchange(f"STOP_N={stop_n}")
    xmlchange(f"STOP_OPTION={stop_option}")
    xmlchange(f"REST_N={stop_n}")
    xmlchange(f"REST_OPTION={stop_option}")
    xmlchange(f"RESUBMIT={resubmit}")
    xmlchange(f"JOB_WALLCLOCK_TIME={wallclock}")

    xmlchange(f"CHARGE_ACCOUNT={project.account}")
    xmlchange(f"PROJECT={project.account}")
    xmlchange(f"JOB_QUEUE={queue}")
    
    # copy restarts
    os.makedirs(f"{project.dir_scratch}/{case}/run", exist_ok=True)
    check_call(
        f"cp {refcaserest_root}/{refdate}-00000/* {project.dir_scratch}/{case}/run/.",
        shell=True,
    )

    check_call(
        "module load python && ./case.setup",
        cwd=caseroot,
        shell=True,
    )

    # handle the myserious failures with ESP
    os.makedirs(f"{caseroot}/SourceMods/src.desp", exist_ok=True)
    with open(f"{project.dir_scratch}/{case}/run/rpointer.esp", "w") as fid:
        fid.write("\n")

    # copy RefCase user_nl files    
    user_nl_files = glob(f"{scriptroot}/CESM-RefCase/{refcase}/user_nl*")
    for file in user_nl_files:
        file_out = os.path.join(caseroot, os.path.basename(file))
        print(f"{file} -> {file_out}")
        with open(file, "r") as fid:
            file_str = fid.read().replace("/glade/p/cesmdata/cseg/inputdata", project.cesm_inputdata)
        with open(file_out, "w") as fid:
            fid.write(file_str)        

    # user_datm files        
    user_datm_files = glob(f"{scriptroot}/CESM-RefCase/{refcase}/user_datm.*")
    for file in user_datm_files:
        file_out = os.path.join(caseroot, os.path.basename(file))
        print(f"{file} -> {file_out}")
        with open(file, "r") as fid:
            file_str = fid.read().replace("/glade/p/cesmdata/cseg/inputdata", project.cesm_inputdata)
        with open(file_out, "w") as fid:
            fid.write(file_str)
        
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

    if curtail_output:
        user_nl["pop"] += textwrap.dedent(
            f"""\

        ! curtail output
        ldiag_bsf = .false.    
        diag_gm_bolus = .false.
        moc_requested = .false.
        n_heat_trans_requested = .false.
        n_salt_trans_requested = .false.
        ldiag_global_tracer_budgets = .false.
        """
        )

    for key, nl in user_nl.items():
        user_nl_file = f"{caseroot}/user_nl_{key}"
        with open(user_nl_file, "a") as fid:
            fid.write(user_nl[key])

    # check_call(
    #     "module load python && ./preview_namelists", 
    #     cwd=caseroot,
    #     shell=True,
    # )

    
    check_call(
        "module load python && ./case.build --skip-provenance-check",
        cwd=caseroot,
        shell=True,
    )

    # set ALT_CO2 tracers to CO2 tracers
    check_call(
        ["./set-alt-co2.sh", f"{rundir}/run/{refcase}.pop.r.{refdate}-00000.nc"],
        cwd=scriptroot,
    )    
    
    #if submit:
    #    check_call(
    #        "module load python && ./case.submit",
    #        cwd=caseroot,
    #        shell=True,
    #    )


def open_dataset(case, stream='pop.h'):
    """access data from a case"""
    grid = pop_tools.get_grid('POP_gx1v7')

    archive_root = f'{project.dir_scratch}/archive/{case}'

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


@click.command()
@click.option('--case')
@click.option('--alk-forcing-file')
@click.option('--refdate')
def main(case, alk_forcing_file, refdate):
    print(case)
    print(alk_forcing_file)
    print(refdate)
    print("=" * 80)
    
    create_oae_case(
        case,
        alk_forcing_file,
        refdate=refdate,
        stop_n=15,
        stop_option="nyear",
        wallclock="12:00:00",
        resubmit=0,
        clobber=False,
        submit=False,
        curtail_output=True,
        queue="regular",
    )

if __name__ == "__main__":
    main()
