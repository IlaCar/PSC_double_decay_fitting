# This script, implemented by Ilaria Carannante (ilariac@kth.se), plots the average traces
import json
from utils import *

import argparse
parser = argparse.ArgumentParser(description="Plot NMDA and AMPA currents of a selected input region")    
parser.add_argument("--region", help="select input region",
                  choices=["M1-contra","M1-ipsi", "S1", "PF"])
args = parser.parse_args()

input_region = args.region

print("Selected input region : " + args.region)

plot_summary(input_region,save_plots=True, show_plots=True)
plot_summary_AMPA(input_region,save_plots=True, show_plots=True)
plot_summary_NMDA(input_region,save_plots=True, show_plots=True)


