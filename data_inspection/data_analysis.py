# This script implemented by Ilaria Carannante (ilariac@kth.se) is used to:
# 1) plot the NMDA and AMPA currents
# 2) extract the NMDA-AMPA ratio
# 3) compute the average trace

import json
from utils import *

import argparse
parser = argparse.ArgumentParser(description="Plot NMDA and AMPA currents, extract ratio, compute average trace")    
parser.add_argument("--region", help="select input region",
                  choices=["M1-contra","M1-ipsi", "S1", "PF"])
parser.add_argument("--type", help="Cell types",
                  choices=["dSPN","iSPN","FS","LTS","ChIN"])
args = parser.parse_args()

input_region = args.region
cell_type = args.type

print("Selected input region : " + args.region)
print("Selected cell type : " + args.type)
path = '../data/'+str(input_region)+'/'+str(cell_type)
    
    
# creating a dictionary containing trace names and values    
file_dict=create_file_dict(path, input_region, cell_type)

# plotting single traces
plot_NMDA_AMPA_traces(file_dict, input_region, cell_type, total_duration=False, save_plots=True, show_plots=False)

# plotting normalized traces
plot_NMDA_AMPA_norm_traces(file_dict, input_region, cell_type, total_duration=True, save_plots=True, show_plots=False)

# finding NMDA/AMPA ratio and average traces
ratio_values_avg, ratio_values_std, ampa_avg_trace, ampa_sem_trace, nmda_avg_trace, nmda_sem_trace = ratio_average_trace(file_dict, input_region, cell_type, save_traces=True)

print('NMDA to AMPA ratio is %.3f and its std is %.3f' %(ratio_values_avg, ratio_values_std))
"%s - %s - NMDA-AMPA summary" %(str(input_region),str(cell_type))
# plotting NMDA to AMPA ratio and average
plot_ratio_average_trace(file_dict, input_region, cell_type, save_plots=True, show_plots=False)




