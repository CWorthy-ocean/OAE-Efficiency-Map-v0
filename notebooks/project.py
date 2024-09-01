import os

scriptroot = os.path.dirname(os.path.realpath(__file__))

USER = os.environ["USER"]


project_name = "OAE-Efficiency-Map"
mach = "perlmutter"

account = "m4746"
dir_scratch=f"/pscratch/sd/{USER[0]}/{USER}"

dir_project_root = f"/global/cfs/projectdirs/m4746/Projects/{project_name}"
os.makedirs(dir_project_root, exist_ok=True)

dir_data = f"{dir_project_root}/data"
os.makedirs(dir_data, exist_ok=True)

dir_codes = f"{dir_project_root}/codes"
os.makedirs(dir_codes, exist_ok=True)

dir_caseroot_root=f"{dir_project_root}/cesm-cases"
os.makedirs(dir_caseroot_root, exist_ok=True)

cesm_inputdata = f"/global/cfs/projectdirs/m4746/Datasets/cesm-inputdata"
os.makedirs(cesm_inputdata, exist_ok=True)

coderoot = f"{dir_codes}/cesm2.2.0"
assert os.path.exists(coderoot)