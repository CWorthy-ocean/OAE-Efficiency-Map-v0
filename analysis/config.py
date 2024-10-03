import os

USER = os.environ['USER']
project_name = 'oae-dor-global-efficiency'

#account = "P93300670"
account = "UCNN0042" 
#account = "NCGD0011"
#account = "NCGD0013"


dir_project_root = f'/glade/work/{USER}/{project_name}'
os.makedirs(dir_project_root, exist_ok=True)

dir_data = f'{dir_project_root}/data'
os.makedirs(dir_data, exist_ok=True)

dir_scratch = f'/glade/scratch/{USER}'

#### Added on May 28, 2023, by Mengyang
# if saving cesm-cases and archive to sracth
dir_project_root_scratch = f'/glade/scratch/{USER}/{project_name}'
os.makedirs(dir_project_root_scratch, exist_ok=True)

# if reading forcing from sracth
dir_data_scratch = f'{dir_project_root_scratch}/data'
os.makedirs(dir_data_scratch, exist_ok=True)

# ### October 28, 2023,  try to save to Matt's scratch space
# dir_scratch = f'/glade/scratch/mclong/OAE-Global-Efficiency'
# dir_project_root_scratch = f'/glade/scratch/mclong/OAE-Global-Efficiency/{project_name}'
# os.makedirs(dir_project_root_scratch, exist_ok=True)

