import os

USER = os.environ['USER']
project_name = 'oae-dor-global-efficiency'

account = "P93300670"

dir_project_root = f'/glade/work/{USER}/{project_name}'
os.makedirs(dir_project_root, exist_ok=True)

dir_data = f'{dir_project_root}/data'
os.makedirs(dir_data, exist_ok=True)

dir_scratch = f'/glade/derecho/scratch/{USER}'