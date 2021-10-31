import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import json
from Newton import *

# fitting functions
def single_exp_decay(x, a, b):
    return a*np.exp(-x/b)

def double_exp_decay(x, I1, tau_d1, I2, tau_d2):
    return I1*np.exp(-x/tau_d1) + I2*np.exp(-x/tau_d2)


# functions
def tau_w (I1,tau_d1,I2,tau_d2):
    return (I1/(I1+I2)*tau_d1 + I2/(I1+I2)*tau_d2)

def t_peak_single(tau_r, tau_d):
    return (tau_r*tau_d)/(tau_d-tau_r) * np.log(tau_d/tau_r)

def fact_single(t_peak, tau_r, tau_d):
    return (-np.exp(-t_peak/tau_r) + np.exp(-t_peak/tau_d))**(-1)

def fact_double(tpeak, tau_r, I1, tau_d1, I2, tau_d2):
    return ((I1*np.exp(-tpeak/tau_d1) + I2*np.exp(-tpeak/tau_d2) - (I1+I2)*np.exp(-tpeak/tau_r))**(-1))

def model(x,tau_r,tau_d):
    return (-np.exp(-x/tau_r)+np.exp(-x/tau_d))

def my_model(x, tau_r, I1, tau_d1, I2, tau_d2):
    return (I1*np.exp(-x/tau_d1) + I2*np.exp(-x/tau_d2) - (I1+I2) * np.exp(-x/tau_r))

######################################################################################################
def plotting_raw_traces(input_region, cell_type, ampa_avg_raw, nmda_avg_raw):
    # defining time vector
    time_raw=np.linspace(0,1000,5000)
    plt.figure()
    plt.plot(time_raw,ampa_avg_raw, label='AMPA')
    if nmda_avg_raw.all()!=0:
        plt.plot(time_raw,nmda_avg_raw, label='NMDA')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')
    if nmda_avg_raw.all()!=0:
        plt.title(str(input_region)+'  '+ str(cell_type)+'  NMDA-AMPA')
    else:
        plt.title(str(input_region)+'  '+ str(cell_type)+'  AMPA')
        
    plt.legend()


######################################################################################################
def plotting_normalized_raw_traces(input_region, cell_type, ampa_avg_raw, nmda_avg_raw):
    # defining time vector
    time_raw=np.linspace(0,1000,5000)

    plt.figure()
    # normalize between -1 and ~0 (the mean of the first 100ms is 0)
    norm_ampa_avg_raw=ampa_avg_raw/(np.abs(np.min(ampa_avg_raw)))
    plt.plot(time_raw,norm_ampa_avg_raw, label='normalized AMPA')


    if nmda_avg_raw.all()!=0:
        # normalizing between ~0 and 1 (the mean of the first 100ms is 0)
        norm_nmda_avg_raw=nmda_avg_raw/(np.max(nmda_avg_raw))
        plt.plot(time_raw,norm_nmda_avg_raw, label='normalized NMDA')
        plt.title(str(input_region)+'  '+ str(cell_type)+'  normalized NMDA-AMPA')

    plt.title(str(input_region)+'  '+ str(cell_type)+'  normalized AMPA')
    plt.xlabel('Time (ms)')
    plt.ylabel('Normalized Current')
    plt.legend()

    if nmda_avg_raw.all()!=0:
        return norm_ampa_avg_raw, norm_nmda_avg_raw
    else:
        norm_nmda_avg_raw = nmda_avg_raw
        return norm_ampa_avg_raw, norm_nmda_avg_raw

######################################################################################################
def shifting_data(input_region, cell_type, norm_ampa_avg_raw, norm_nmda_avg_raw):

    plt.figure()

    #shifting data
    if str(input_region)=='M1-contra':
        if str(cell_type)=='FS':
            tt=512
        else:
            tt=510             
    if str(input_region)=='M1-ipsi':
        if str(cell_type)=='dSPN':
            tt=511
        if str(cell_type)=='iSPN':
            tt=511 
        if str(cell_type)=='FS':
            tt=510 
        if str(cell_type)=='ChIN':
            tt=515 
        if str(cell_type)=='LTS':
            tt=519                          
    if str(input_region)=='S1':
        if str(cell_type)=='dSPN':
            tt=510
        if str(cell_type)=='iSPN':
            tt=509 
        if str(cell_type)=='FS':
            tt=510 
        if str(cell_type)=='ChIN':
            tt=514 
    if str(input_region)=='PF':
        if str(cell_type)=='dSPN':
            tt=511
        if str(cell_type)=='iSPN':
            tt=512 
        if str(cell_type)=='FS':
            tt=508
        if str(cell_type)=='ChIN':
            tt=513                           

    # defining time vector
    time=np.linspace(0,(5000-tt)/5,5000-tt)
        
    norm_ampa_avg=norm_ampa_avg_raw[tt:]
    index_min_avg=norm_ampa_avg.argmin()
    plt.plot(time,norm_ampa_avg,label='shifted normalized raw data AMPA')

    if norm_nmda_avg_raw.all()!=0:
        norm_nmda_avg=norm_nmda_avg_raw[tt:]
        index_max_avg=norm_nmda_avg.argmax()
        plt.plot(time,norm_nmda_avg,label='shifted normalized raw data NMDA')  
        plt.title(str(input_region)+'  '+ str(cell_type)+'  shifted normalized NMDA-AMPA') 
    else:
        plt.title(str(input_region)+'  '+ str(cell_type)+'  shifted normalized AMPA')

    plt.xlabel('Time (ms)')
    plt.ylabel('Normalized Current')

    plt.legend()
    if norm_nmda_avg_raw.all() !=0 :
        return time, index_min_avg, norm_ampa_avg, index_max_avg, norm_nmda_avg
    else:
        index_max_avg=-1
        norm_nmda_avg=norm_nmda_avg_raw
        return time, index_min_avg, norm_ampa_avg, index_max_avg, norm_nmda_avg


######################################################################################################
def plotting_ratio(input_region, cell_type, time, nmda_avg_raw, index_max_avg_raw, ampa_avg_raw, nmda_ratio):
    plt.figure()
    plt.plot(time, nmda_avg_raw, label='NMDA avg raw data')
    plt.plot(time[index_max_avg_raw+250], nmda_avg_raw[index_max_avg_raw+250],'*r', markersize=7, label='~50 ms after peak')
    plt.plot(time, ampa_avg_raw, label='AMPA avg raw data')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')
    plt.title(str(input_region)+' _ '+ str(cell_type) + ' _ ratio: ' + str(nmda_ratio)[:5])
    plt.legend()

######################################################################################################
def fitting_rise(input_region, cell_type, receptor, time, trace, index):

    if str(input_region)=='M1-contra':
        if str(receptor)=='AMPA':
            if str(cell_type)=='dSPN':
                index_start = 2  
                index_stop = 11
            if str(cell_type)=='iSPN':
                index_start = 2  
                index_stop = 11  
            if str(cell_type)=='FS':
                index_start = 4  
                index_stop = 15        
        else: 
            if str(cell_type)=='dSPN':
                index_start = 3  
                index_stop = 20
            if str(cell_type)=='iSPN':
                index_start = 3  
                index_stop = 20    
                
    if str(input_region)=='M1-ipsi':
        if str(receptor)=='AMPA':
            if str(cell_type)=='dSPN':
                index_start = 3  
                index_stop = 12
            if str(cell_type)=='iSPN':
                index_start = 2  
                index_stop = 11  
            if str(cell_type)=='FS':
                index_start = 4  
                index_stop = 12  
            if str(cell_type)=='ChIN':
                index_start = 5  
                index_stop = 12      
            if str(cell_type)=='LTS':
                index_start = 4  
                index_stop = 25                    
        else: 
            if str(cell_type)=='dSPN':
                index_start = 4  
                index_stop = 23
            if str(cell_type)=='iSPN':
                index_start = 3  
                index_stop = 19 
            if str(cell_type)=='FS':
                index_start = 1  
                index_stop = 12   
            if str(cell_type)=='ChIN':
                index_start = 7 
                index_stop = 58  
            if str(cell_type)=='LTS':
                index_start = 3  
                index_stop = 25                                  

    if str(input_region)=='S1':
        if str(receptor)=='AMPA':
            if str(cell_type)=='dSPN':
                index_start = 3  
                index_stop = 14
            if str(cell_type)=='iSPN':
                index_start = 2  
                index_stop = 16  
            if str(cell_type)=='FS':
                index_start = 2  
                index_stop = 10  
            if str(cell_type)=='ChIN':
                index_start = 2  
                index_stop = 11      
                 
        else: 
            if str(cell_type)=='dSPN':
                index_start = 4  
                index_stop = 20
            if str(cell_type)=='iSPN':
                index_start = 5  
                index_stop = 27 
            if str(cell_type)=='FS':
                index_start = 4  
                index_stop = 18   
            if str(cell_type)=='ChIN':
                index_start = 5 
                index_stop = 45  
 
    if str(input_region)=='PF':
        if str(receptor)=='AMPA':
            if str(cell_type)=='dSPN':
                index_start = 3  
                index_stop = 21
            if str(cell_type)=='iSPN':
                index_start = 6  
                index_stop = 23 
            if str(cell_type)=='FS':
                index_start = 4  
                index_stop = 14
            if str(cell_type)=='ChIN':
                index_start = 2  
                index_stop = 14      
                 
        else: 
            if str(cell_type)=='dSPN':
                index_start = 5  
                index_stop = 24
            if str(cell_type)=='iSPN':
                index_start = 7  
                index_stop = 32 
            if str(cell_type)=='FS':
                index_start = 4  
                index_stop = 18   
            if str(cell_type)=='ChIN':
                index_start = 5 
                index_stop = 40  
 
                
    plt.figure()
    trace_10=trace[index_start]
    trace_90=trace[index_stop]
    time_rise=time[index_stop]-time[index_start]
    tau_1=time_rise/np.log(9)
    plt.plot(time[:index],trace[:index],'o-',color='#e41a1c',markersize=0.5, label = 'normalized data')
    plt.xlabel('Time (ms)')
    plt.ylabel('Normalized Current')

    if receptor == 'NMDA':
        plt.xlim(0,8)
        plt.plot(time[index_start],trace_10,'*b',markersize=10,label="~10%")
        plt.plot(time[index_stop],trace_90,'*g',markersize=10,label="~90%")
        if cell_type=="ChIN":
            plt.xlim(0,15)
            plt.text(5.5, 0.25, r'time rise=$t_{90}-t_{10}$=' +'%5.3f'% time_rise+'\n' +r'$\tau_{rise}$= time rise / ln(9)=' + '%5.3f'% tau_1 , {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"))

        else:
            plt.text(4.5, 0.25, r'time rise=$t_{90}-t_{10}$=' +'%5.3f'% time_rise+'\n' +r'$\tau_{rise}$= time rise / ln(9)=' + '%5.3f'% tau_1 , {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"))
        plt.legend(loc=7,frameon=True,edgecolor='black',fontsize='medium')
    else:
        plt.xlim(0,5)
        plt.plot(time[index_start],trace_10,'*b',markersize=10,label="~10%")
        plt.plot(time[index_stop],trace_90,'*g',markersize=10,label="~90%")
        plt.text(2.75, -0.15, r'time rise=$t_{90}-t_{10}$=' +'%5.3f'% time_rise+'\n' +r'$\tau_{rise}$= time rise / ln(9)=' + '%5.3f'% tau_1 , {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"))
        plt.legend(loc=7,frameon=True,edgecolor='black',fontsize='medium')
        if cell_type=='LTS':
            plt.xlim(0,8)
    plt.title(str(input_region) +'    ' + str(cell_type)+'    ' + str(receptor) +'    ' + r'$\tau_{rise}$' + ' fitting')
    print('tau_1:', tau_1)
    return tau_1

######################################################################################################
def fitting_mono_decay(input_region, cell_type, receptor, time, trace, index_stationary):
    fig, ax = plt.subplots()
    import matplotlib as mpl
    label_size = 12
    mpl.rcParams['xtick.labelsize'] = label_size    
    mpl.rcParams['ytick.labelsize'] = label_size     


    if receptor == 'NMDA':
        params, params_covariance = optimize.curve_fit(single_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([0.,0.1],[10.,250]))

    else:
        params, params_covariance = optimize.curve_fit(single_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([-10.,0.1],[0.,50]))    

    I_sing = params[0]
    tau_d_sing = params[1]
    plt.plot(time[index_stationary:],trace[index_stationary:],'o-',color='#e41a1c',markersize=0.5, label = 'normalized data')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Current')
    ax.plot(time[index_stationary:],single_exp_decay(time[index_stationary:],I_sing,tau_d_sing), '#4daf4a', label=r'Fitting: $A \, e^{-x/B}$',linewidth=3)

    if receptor == 'NMDA':
        ax.text(600, 0.5, r'$A=$'+ '%5.3f'% I_sing + r'$,  B=$'+ '%5.3f'% tau_d_sing, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"), fontsize='11')
    else:
        ax.text(600, -0.5, r'$A=$'+ '%5.3f'% I_sing + r'$,  B=$'+ '%5.3f'% tau_d_sing, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"), fontsize='11')

    ax.legend(loc='best',frameon=True,edgecolor='black',fontsize='large')
    fig.suptitle(str(input_region) +'    ' + str(cell_type)+'    ' + str(receptor) +'    ' + r'$\tau_{decay}$' +  '\n'+ ' Single exponential fitting', fontsize='15')
    axes = plt.gca()
    axes.xaxis.label.set_size(13)
    axes.yaxis.label.set_size(13)

    return params

######################################################################################################
def fitting_double_decay(input_region, cell_type, receptor, time, trace, index_stationary):
    fig, ax = plt.subplots()

    if receptor == 'NMDA':
        if cell_type=='ChAT':
            params, params_covariance = optimize.curve_fit(double_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([0.,1.,0.,1.],[100.,100.,100.,650.]))
        else:
            params, params_covariance = optimize.curve_fit(double_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([0.,1.,0.,1.],[100.,100.,100.,450.]))
    else:
        if cell_type=='ChAT':
            params, params_covariance = optimize.curve_fit(double_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([-100.,1.,-100.,1.],[0.,100.,0.,1250.]))
        else:
            params, params_covariance = optimize.curve_fit(double_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([-100.,1.,-100.,1.],[0.,100.,0.,850.]))

    a1 = params[0] 
    b1 = params[1]
    c1 = params[2]
    d1 = params[3] 
    tau_w_nmda=tau_w (a1,b1,c1,d1)
    plt.plot(time[index_stationary:],trace[index_stationary:],'o-',color='#e41a1c',markersize=0.5, label = 'normalized data')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Current')
    ax.plot(time[index_stationary:],double_exp_decay(time[index_stationary:],a1,b1,c1,d1), color='#377eb8', label=r'Fitting: $I_f \, e^{-x/\tau_f}+I_s \, e^{-x/\tau_s}$',linewidth=3)

    if receptor == 'NMDA':
        ax.text(500, 0.5, r'$I_f=$'+ '%5.3f'% a1+'\n' +r'$\tau_f=$'+ '%5.3f'% b1+'\n'+r'$I_s=$'+ '%5.3f'% c1+'\n'+r'$\tau_s=$'+ '%5.3f'% d1, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"), fontsize='11')
        ax.text(500, 0.35, r'$ \tau_\omega= \frac{I_f}{I_f+I_s}\tau_f + \frac{I_s}{I_f+I_s}\tau_s =$' +'%5.3f'% tau_w_nmda, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"), fontsize='11')
    else:
        ax.text(500, -0.5, r'$I_f=$'+ '%5.3f'% a1+'\n' +r'$\tau_f=$'+ '%5.3f'% b1+'\n'+r'$I_s=$'+ '%5.3f'% c1+'\n'+r'$\tau_s=$'+ '%5.3f'% d1, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"))
        plt.text(500, -0.7, r'$ \tau_\omega= \frac{I_f}{I_f+I_s}\tau_f + \frac{I_s}{I_f+I_s}\tau_s =$' +'%5.3f'% tau_w_nmda, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"), fontsize='11')

    ax.legend(loc='best',frameon=True,edgecolor='black',fontsize='large')
    ax.legend(loc='best',frameon=True,edgecolor='black',fontsize='large')
    fig.suptitle(str(input_region) +'    ' + str(cell_type)+'    ' + str(receptor) +'    ' + r'$\tau_{decay}$' + ' fitting' + '\n' + 'Double-exponential fitting', fontsize='15')
    axes = plt.gca()
    axes.xaxis.label.set_size(13)
    axes.yaxis.label.set_size(13)
    return params, tau_w_nmda

######################################################################################################
def fitting_forced_mono_decay(input_region, cell_type, receptor, time, trace, index_stationary, tau_w):
    fig, ax = plt.subplots()


    if receptor == 'NMDA':
        params, params_covariance = optimize.curve_fit(single_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([0.,tau_w],[10.,tau_w+10**(-6)]))
    else:
        params, params_covariance = optimize.curve_fit(single_exp_decay, time[index_stationary:], trace[index_stationary:], bounds=([-10.,tau_w],[0.,tau_w+10**(-6)]))    

    I_w = params[0]
    tau_d_w = params[1]
    plt.plot(time[index_stationary:],trace[index_stationary:],'o-',color='#e41a1c',markersize=0.5, label = 'normalized data')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Current')
    plt.plot(time[index_stationary:],single_exp_decay(time[index_stationary:],I_w,tau_d_w), '#984ea3', label=r'Fitting: $A \, e^{-x/\tau_w}$',linewidth=3)

    if receptor == 'NMDA':
        ax.text(550, 0.5, r'$A=$'+ '%5.3f'% I_w + r'$,  \tau_w=$'+ '%5.3f'% tau_d_w, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"), fontsize='11')
    else:
        ax.text(550, -0.5, r'$A=$'+ '%5.3f'% I_w + r'$,  \tau_w=$'+ '%5.3f'% tau_d_w, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"), fontsize='11')

    ax.legend(loc='best',frameon=True,edgecolor='black',fontsize='large')
    fig.suptitle(str(input_region) +'    ' + str(cell_type)+'    ' + str(receptor) +'    ' + r'$\tau_w}$' +  '\n'+ ' Forced fitting', fontsize='15')

    axes = plt.gca()
    axes.xaxis.label.set_size(13)
    axes.yaxis.label.set_size(13)

    return params


######################################################################################################
def plotting_double_ex_components(input_region, cell_type, receptor, time, trace, index_stationary, a1,b1,c1,d1):
    plt.figure()
    plt.plot(time[index_stationary:],trace[index_stationary:],'o-',color='#e41a1c',markersize=0.5, label = 'original data')
    plt.xlabel('time (ms)')
    plt.ylabel('Normalized Current')
    plt.plot(time[index_stationary:],double_exp_decay(time[index_stationary:],a1,b1,c1,d1), color='#377eb8', label=r'Fitting: $I_f \, e^{-x/\tau_f}+I_s \, e^{-x/\tau_s}$',linewidth=3)
    plt.plot(time[index_stationary:],single_exp_decay(time[index_stationary:],a1,b1),'--',color='cyan',label=r'Fitting: $I_f \, e^{-x/\tau_f}$',linewidth=2)
    plt.plot(time[index_stationary:],single_exp_decay(time[index_stationary:],c1,d1),'--',color='cornflowerblue',label=r'Fitting: $I_s \, e^{-x/\tau_s}$',linewidth=2)

    if receptor == 'NMDA':
        plt.text(550, 0.2, r'$I_f=$'+ '%5.3f'% a1+'\n' +r'$\tau_f=$'+ '%5.3f'% b1+'\n'+r'$I_s=$'+ '%5.3f'% c1+'\n'+r'$\tau_s=$'+ '%5.3f'% d1, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"))  
    else:  
        plt.text(550, -0.5, r'$I_f=$'+ '%5.3f'% a1+'\n' +r'$\tau_f=$'+ '%5.3f'% b1+'\n'+r'$I_s=$'+ '%5.3f'% c1+'\n'+r'$\tau_s=$'+ '%5.3f'% d1, {'color': 'k', 'fontsize': 10},bbox=dict(fc="none"))
    plt.legend(loc='best',frameon=True,edgecolor='black',fontsize='large')
    plt.title(str(input_region) +'    ' + str(cell_type)+'    ' + str(receptor) + '\n' +  'Double exponential components')





######################################################################################################
def plotting_models(input_region, cell_type, receptor, time, trace, tau_1, tau_d_sing,fact_single, tau_w, fact_w, I2_double, tau_2_double, I3_double, tau_3_double, fact_double):
    fig, ax = plt.subplots()
    #plt.title('Models')
    plt.plot(time,trace,'o-',color='#e41a1c',markersize=0.5, label = 'normalized data')
    
    if receptor == 'NMDA':
        plt.plot(time,model(time,tau_1, tau_d_sing)*fact_single,color='#4daf4a', label='single decay constant',linewidth=2)
        plt.plot(time,model(time,tau_1, tau_w)*fact_w, color='#984ea3', label='weighted decay time constant',linewidth=2)
        plt.plot(time,my_model(time, tau_1, I2_double, tau_2_double, I3_double, tau_3_double)*fact_double, color='#377eb8', label='double decay time constant',linewidth=2)
    else:
        ax.plot(time,-model(time,tau_1, tau_d_sing)*fact_single, color='#4daf4a', label='single decay time constant',linewidth=2)
        ax.plot(time,-model(time,tau_1, tau_w)*fact_w, color='#984ea3', label='weighted decay time constant',linewidth=2)
        ax.plot(time,-my_model(time, tau_1, I2_double, tau_2_double, I3_double, tau_3_double)*fact_double, color='#377eb8', label='double decay time constant',linewidth=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Current')
    fig.suptitle(str(input_region) +'    ' + str(cell_type)+'    ' + str(receptor) +  '\n'+'Model comparison', fontsize='15')
    ax.legend()

    axes = plt.gca()
    axes.xaxis.label.set_size(13)
    axes.yaxis.label.set_size(13)    
    
    fig.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_'+str(receptor)+'_models.png')
    #fig.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_'+str(receptor)+'_models.pdf')

######################################################################################################
def plotting_models_not_normalized(input_region, cell_type, receptor, time, trace, stationary_point, tau_1, tau_d_sing,fact_single, tau_w, fact_w, I2_double, tau_2_double, I3_double, tau_3_double, fact_double):
    plt.figure()
    plt.plot(time,trace,'o-',color='#e41a1c',markersize=2, label = 'data')
    
    if receptor == 'NMDA':
        plt.plot(time,stationary_point*model(time,tau_1, tau_d_sing)*fact_single, color='#4daf4a', label='single time constant')
        plt.plot(time,stationary_point*model(time,tau_1, tau_w)*fact_w, color='#984ea3', label='weighted time constant')
        plt.plot(time,stationary_point*my_model(time, tau_1, I2_double, tau_2_double, I3_double, tau_3_double)*fact_double, color='#377eb8', label='my double')
    else:
        plt.plot(time,stationary_point*model(time,tau_1, tau_d_sing)*fact_single, color='#4daf4a', label='single time constant')
        plt.plot(time,stationary_point*model(time,tau_1, tau_w)*fact_w, color='#984ea3', label='weighted time constant')
        plt.plot(time,stationary_point*my_model(time, tau_1, I2_double, tau_2_double, I3_double, tau_3_double)*fact_double, color='#377eb8', label='my double')

    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')
    plt.title(str(input_region) +'    ' + str(cell_type)+'    ' + str(receptor) +'  -  Models not normalized')
    plt.legend()

######################################################################################################
def RMSE_fitting(time, norm_avg, index_avg, \
                I2_double, tau_2_double, I3_double, tau_3_double, \
                I_sing, tau_d_sing, \
                I_w,tau_d_w):
    test_my_fitting=0
    test_single_fitting=0
    test_w_fitting=0

    for i in np.arange(index_avg,len(time)):
        test_my_fitting=test_my_fitting + (double_exp_decay(time[i], I2_double, tau_2_double, I3_double, tau_3_double)-norm_avg[i])**2
        test_single_fitting=test_single_fitting + (single_exp_decay(time[i],I_sing, tau_d_sing)-norm_avg[i])**2
        test_w_fitting=test_w_fitting + (single_exp_decay(time[i],I_w,tau_d_w)-norm_avg[i])**2

    test_my_fitting=test_my_fitting**(0.5)/(len(np.arange(index_avg,len(time)))**(0.5))
    test_single_fitting=test_single_fitting**(0.5)/(len(np.arange(index_avg,len(time)))**(0.5))
    test_w_fitting=test_w_fitting**(0.5)/(len(np.arange(index_avg,len(time)))**(0.5))

    return test_my_fitting, test_single_fitting, test_w_fitting







