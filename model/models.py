import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import json
from Newton import *
from functions_plots import *


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


# Newton's method parameters
t0        = 0
t_in_ampa = 5
t_in_nmda = 15
TOL       = 1.e-10
max_iter  = 8


ampa_avg_raw=np.loadtxt('../data_inspection/avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_avg_AMPA.txt', unpack= True)
nmda_avg_raw=np.loadtxt('../data_inspection/avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_avg_NMDA.txt', unpack= True)

nmda_ratio = np.genfromtxt('../data_inspection/avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_ratio.txt', delimiter=' ', dtype=['O',float]).tolist()[0][1]

#plotting raw traces
plotting_raw_traces(input_region, cell_type, ampa_avg_raw, nmda_avg_raw)

#plotting normalized raw traces
norm_ampa_avg_raw, norm_nmda_avg_raw = plotting_normalized_raw_traces(input_region, cell_type, ampa_avg_raw,  nmda_avg_raw)

#shifting data
time, index_min_avg, norm_ampa_avg, index_max_avg, norm_nmda_avg = shifting_data(input_region, cell_type, norm_ampa_avg_raw, norm_nmda_avg_raw)
#plt.show()

######################################################################################################
#fitting ampa tau rise 
print('... fitting tau rise - AMPA ...')
tau_1_ampa = fitting_rise(input_region, cell_type, 'AMPA', time, norm_ampa_avg, index_min_avg)
#plt.show()

#fitting ampa tau decay
print('... fitting tau decay with a single exponential - AMPA ...')
I_sing_ampa, tau_d_sing_ampa = fitting_mono_decay(input_region, cell_type, 'AMPA', time, norm_ampa_avg, index_min_avg)

tp_single_ampa = t_peak_single(tau_1_ampa, tau_d_sing_ampa)
fact_single_ampa = fact_single(tp_single_ampa, tau_1_ampa, tau_d_sing_ampa)
print('tp_single_ampa:  ',tp_single_ampa)
print('fact_single_ampa:   ',fact_single_ampa)
#plt.show()

print('... fitting tau decay with a double exponential - AMPA ...')
params, tau_w_ampa = \
fitting_double_decay(input_region, cell_type, 'AMPA', time, norm_ampa_avg, index_min_avg)
I2_double_ampa = params[0]
tau_2_double_ampa = params[1]
I3_double_ampa = params[2]
tau_3_double_ampa = params[3]
print('tau double decays:',tau_2_double_ampa,tau_3_double_ampa )
#plt.show()

print('... fitting tau decay with a forced single exponential - AMPA ...')
I_w_ampa,tau_d_w_ampa =fitting_forced_mono_decay(input_region, cell_type, 'AMPA', time, norm_ampa_avg, index_min_avg, tau_w_ampa)
tp_w_ampa = t_peak_single(tau_1_ampa, tau_w_ampa)
fact_w_ampa = fact_single(tp_w_ampa, tau_1_ampa, tau_w_ampa)
print('tp_w_ampa:  ', tp_w_ampa)
print('fact_w_ampa:   ',fact_w_ampa)
#plt.show()

print("... computing Newton's method - AMPA ...")
tp_double_ampa = my_newton2(I2_double_ampa+I3_double_ampa,tau_1_ampa,I2_double_ampa,tau_2_double_ampa,I3_double_ampa,tau_3_double_ampa,t0,t_in_ampa,TOL,max_iter)
fact_double_ampa = fact_double(tp_double_ampa, tau_1_ampa, I2_double_ampa, tau_2_double_ampa, I3_double_ampa, tau_3_double_ampa)
print('tp_double_ampa:  ', tp_double_ampa)
print('fact_double_ampa:   ', fact_double_ampa)

plotting_double_ex_components(input_region, cell_type, 'AMPA', time, norm_ampa_avg , index_min_avg , I2_double_ampa,tau_2_double_ampa,I3_double_ampa,tau_3_double_ampa)

print("... plotting different models - AMPA ...")

plotting_models(input_region, cell_type, 'AMPA', time, norm_ampa_avg, tau_1_ampa, tau_d_sing_ampa,fact_single_ampa, tau_w_ampa, fact_w_ampa, I2_double_ampa, tau_2_double_ampa, I3_double_ampa, tau_3_double_ampa, fact_double_ampa)
#plt.show()

# statistical test 
test_my_fitting_ampa, test_single_fitting_ampa, test_w_fitting_ampa = RMSE_fitting( \
                time, norm_ampa_avg, index_min_avg, \
                I2_double_ampa, tau_2_double_ampa, I3_double_ampa, tau_3_double_ampa, \
                I_sing_ampa, tau_d_sing_ampa, \
                I_w_ampa,tau_d_w_ampa)
                                                           
#plt.show()

print('test_my_fitting_ampa',test_my_fitting_ampa)
print('test_single_fitting_ampa',test_single_fitting_ampa)
print('test_w_fitting_ampa',test_w_fitting_ampa)

print('#############################################################')
print('test ratio fitting single',test_single_fitting_ampa/test_my_fitting_ampa )
print('test ratio fitting w',test_w_fitting_ampa/test_my_fitting_ampa )

######################################################################################################
if norm_nmda_avg_raw.all() !=0 :
    #fitting nmda tau rise
    print('... fitting tau rise - NMDA ...')
    tau_1_nmda = fitting_rise(input_region, cell_type, 'NMDA', time, norm_nmda_avg, index_max_avg)
    #plt.show()

    #fitting nmda tau decay
    print('... fitting tau decay with a single exponential - NMDA ...')

    I_sing_nmda, tau_d_sing_nmda = fitting_mono_decay(input_region, cell_type, 'NMDA', time, norm_nmda_avg, index_max_avg)

    tp_single_nmda = t_peak_single(tau_1_nmda, tau_d_sing_nmda)
    fact_single_nmda = fact_single(tp_single_nmda, tau_1_nmda, tau_d_sing_nmda)
    print('tp_single_nmda:  ',tp_single_nmda)
    print('fact_single_nmda:   ',fact_single_nmda)

    print('... fitting tau decay with a double exponential - NMDA ...')
    params, tau_w_nmda = \
    fitting_double_decay(input_region, cell_type, 'NMDA', time, norm_nmda_avg, index_max_avg)
    I2_double_nmda = params[0]
    tau_2_double_nmda = params[1]
    I3_double_nmda = params[2]
    tau_3_double_nmda = params[3]
    print('tau double decays:',tau_2_double_nmda,tau_3_double_nmda )

    print('... fitting tau decay with a forced single exponential - AMPA ...')
    I_w_nmda,tau_d_w_nmda = fitting_forced_mono_decay(input_region, cell_type, 'NMDA', time, norm_nmda_avg, index_max_avg, tau_w_nmda)

    tp_w_nmda = t_peak_single(tau_1_nmda, tau_w_nmda)
    fact_w_nmda = fact_single(tp_w_nmda, tau_1_nmda, tau_w_nmda)
    print('tp_w_nmda:  ', tp_w_nmda)
    print('fact_w_nmda:   ',fact_w_nmda)


    print("... computing Newton's method - NMDA ...")

    tp_double_nmda = my_newton2(I2_double_nmda+I3_double_nmda,tau_1_nmda,I2_double_nmda,tau_2_double_nmda,I3_double_nmda,tau_3_double_nmda,t0,t_in_nmda,TOL,max_iter)
    fact_double_nmda = fact_double(tp_double_nmda, tau_1_nmda, I2_double_nmda, tau_2_double_nmda, I3_double_nmda, tau_3_double_nmda)
    print('tp_double_nmda:  ', tp_double_nmda)
    print('fact_double_nmda:   ', fact_double_nmda)

    #plotting_double_ex_components(input_region, cell_type, 'NMDA', time, norm_nmda_avg , index_max_avg , I2_double_nmda,tau_2_double_nmda,I3_double_nmda,tau_3_double_nmda)

    print("... plotting different models - NMDA ...")

    plotting_models(input_region, cell_type, 'NMDA', time, norm_nmda_avg, tau_1_nmda, tau_d_sing_nmda,fact_single_nmda, tau_w_nmda, fact_w_nmda, I2_double_nmda, tau_2_double_nmda, I3_double_nmda, tau_3_double_nmda, fact_double_nmda)

    #plotting_models_not_normalized(input_region, cell_type, 'NMDA', time, nmda_avg, max_nmda_avg, tau_1_nmda, tau_d_sing_nmda,fact_single_nmda, tau_w_nmda, fact_w_nmda, I2_double_nmda, tau_2_double_nmda, I3_double_nmda, tau_3_double_nmda, fact_double_nmda)

    # statistical test 
    test_my_fitting_nmda, test_single_fitting_nmda, test_w_fitting_nmda = RMSE_fitting( \
                    time, norm_nmda_avg, index_max_avg, \
                    I2_double_nmda, tau_2_double_nmda, I3_double_nmda, tau_3_double_nmda, \
                    I_sing_nmda, tau_d_sing_nmda, \
                    I_w_nmda,tau_d_w_nmda)
              
                    
    print('test_my_fitting_nmda',test_my_fitting_nmda)
    print('test_single_fitting_nmda',test_single_fitting_nmda)
    print('test_w_fitting_nmda',test_w_fitting_nmda)

    print('#############################################################')
    print('test ratio fitting single',test_single_fitting_nmda/test_my_fitting_nmda )
    print('test ratio fitting w',test_w_fitting_nmda/test_my_fitting_nmda )

else:
    print('no NMDA trace')

plt.show()

#saving parameters
print('saving parameters')
if norm_nmda_avg_raw.all() == 0 :
    tau_1_nmda = 0
    tau_2_double_nmda = 0
    tau_3_double_nmda = 0
    I2_double_nmda = 0
    I3_double_nmda = 0
    tp_double_nmda = 0
    fact_double_nmda = 0  
    nmda_ratio = 0

info={  
    'metadata': {
        'input_region'      : input_region,
        'cell_type'         : cell_type
        },
    'data':{
        'tau1_ampa' : tau_1_ampa,
        'tau2_ampa' : tau_2_double_ampa,
        'tau3_ampa' : tau_3_double_ampa,
        'I2_ampa'   : I2_double_ampa,
        'I3_ampa'   : I3_double_ampa,
        'tpeak_ampa' : tp_double_ampa,    
        'factor_ampa': fact_double_ampa,   
        'tau1_nmda' : tau_1_nmda,
        'tau2_nmda' : tau_2_double_nmda,
        'tau3_nmda' : tau_3_double_nmda,
        'I2_nmda'   : I2_double_nmda,
        'I3_nmda'   : I3_double_nmda,
        'tpeak_nmda' : tp_double_nmda,
        'factor_nmda': fact_double_nmda,   
        'nmda_ratio' : nmda_ratio
        }
    }

#saving info in json
with open('json_files/'+str(info['metadata']['input_region'])+'_'+str(info['metadata']['cell_type'])+'.json', 'w') as fp:
        json.dump(info, fp, indent = 2)

if norm_nmda_avg_raw.all() == 0 :
    info={  
        'metadata': {
            'input_region'      : input_region,
            'cell_type'         : cell_type
            },
        'data':{
            'test_my_fitting_ampa' : test_my_fitting_ampa,
            'test_single_fitting_ampa' : test_single_fitting_ampa,
            'test_w_fitting_ampa' : test_w_fitting_ampa,
            }
        }
else:
    info={  
        'metadata': {
            'input_region'      : input_region,
            'cell_type'         : cell_type
            },
        'data':{
            'test_my_fitting_ampa' : test_my_fitting_ampa,
            'test_single_fitting_ampa' : test_single_fitting_ampa,
            'test_w_fitting_ampa' : test_w_fitting_ampa,
            'test_my_fitting_nmda' : test_my_fitting_nmda,
            'test_single_fitting_nmda'   : test_single_fitting_nmda,
            'test_w_fitting_nmda' : test_w_fitting_nmda,
            }
        }

with open('json_files/'+str(info['metadata']['input_region'])+'_'+str(info['metadata']['cell_type'])+'_fitting.json', 'w') as fp:
        json.dump(info, fp, indent = 2)

if norm_nmda_avg_raw.all() ==0 :
    tau_1_nmda = 0
    tau_d_sing_nmda = 0
    tp_single_nmda = 0 
    fact_single_nmda = 0
    tau_d_w_nmda =  0
    tp_w_nmda = 0 
    fact_w_nmda = 0
    nmda_ratio= 0

info={  
    'metadata': {
        'input_region'      : input_region,
        'cell_type'         : cell_type
        },
    'data':{
        'tau1_ampa' : tau_1_ampa,
        'tau_d_sing_ampa' : tau_d_sing_ampa,
        'tp_single_ampa' : tp_single_ampa, 
        'fact_single_ampa' : fact_single_ampa,
        'tau_d_w_ampa' : tau_d_w_ampa,
        'tp_w_ampa' : tp_w_ampa, 
        'fact_w_ampa' : fact_w_ampa,
        'tau1_nmda' : tau_1_nmda,
        'tau_d_sing_nmda' : tau_d_sing_nmda,
        'tp_single_nmda' : tp_single_nmda, 
        'fact_single_nmda' : fact_single_nmda,
        'tau_d_w_nmda' : tau_d_w_nmda,
        'tp_w_nmda' : tp_w_nmda, 
        'fact_w_nmda' : fact_w_nmda,
        'nmda_ratio' : nmda_ratio
        }
    }

with open('json_files/'+str(info['metadata']['input_region'])+'_'+str(info['metadata']['cell_type'])+'_single_w.json', 'w') as fp:
        json.dump(info, fp, indent = 2)




























