# This script implemented by Ilaria Carannante (ilariac@kth.se)
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

#####################################################################################################
def create_file_dict(path, input_region, cell_type):
    file_dict=dict() 

    #reading single traces
    idx=0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(filename, 'r') as f: 
            if str('M1') in input_region:
                fname=os.path.basename(filename)[5:10]+os.path.basename(filename)[-8:-4]            
                if str(cell_type)=='LTS':
                    fname=os.path.basename(filename)[:8]
            else:      
                fname=os.path.basename(filename)[3:8]+os.path.basename(filename)[-8:-4]
            fname_value=np.loadtxt(filename)
            fname_value=fname_value-np.mean(fname_value[0:500]) # subtracting the average value between 0 and 100 ms
            fname_norm=os.path.basename(fname)+'_norm'
              
            if str('AMPA') in fname:
              fname_norm_value=fname_value/(np.abs(np.min(fname_value))) # normalizing traces
            if str('NMDA') in fname:
              fname_norm_value=fname_value/np.max(fname_value) # normalizing traces
                
            file_dict[fname]=(fname_value)
            file_dict[fname_norm]=(fname_norm_value)
            idx+=1
    print('Total imported files:', idx)
    return file_dict
    
#####################################################################################################    
def plot_NMDA_AMPA_traces(file_dict, input_region,cell_type, total_duration, save_plots, show_plots):

    # defining time vector
    time=np.linspace(0,1000,5000)
    #plotting single traces
    for key, value in sorted(file_dict.items()):
        if str('AMPA') in key and str('norm') not in key:
            plt.figure()
            if total_duration == True:
                plt.plot(time, value, label=key)
            else:
                plt.plot(time[450:800], value[450:800], label=key)
            plt.axvline(x=time[value[:2000].argmin()], color='r', linestyle='--')                

        if str('NMDA') in key and str('norm') not in key:
            if total_duration == True:
                plt.plot(time, value, label=key)
            else:
                plt.plot(time[450:800], value[450:800], label=key)
            plt.xlabel('Time (ms)')
            plt.ylabel('Current (A)')
            plt.title(str(input_region)+'_'+str(cell_type)+'_NMDA_AMPA')
            plt.legend()
            if save_plots==True and total_duration==True:
                #plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_'+str(key[:4])+'.pdf', dpi=300)
                plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_'+str(key[:4])+'.png')
            elif save_plots==True and total_duration==False:     
                #plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_zoom_'+str(key[:4])+'.pdf', dpi=300) 
                plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_zoom_'+str(key[:4])+'.png')                
    if show_plots == True:
        plt.show()

#####################################################################################################
def plot_NMDA_AMPA_norm_traces(file_dict, input_region,cell_type, total_duration, save_plots, show_plots):

    # defining time vector
    time=np.linspace(0,1000,5000)
    #plotting single traces
    plt.figure(figsize=(14,7))
    for key, value in sorted(file_dict.items()):
        if str('AMPA_norm') in key:
            if total_duration == True:
                plt.plot(time, value, label=key)
            else:
                plt.plot(time[300:1250], value[300:1250], label=key)             

        if str('NMDA_norm') in key:
            if total_duration == True:
                plt.plot(time, value, label=key)
            else:
                plt.plot(time[300:1250], value[300:1250], label=key)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')
    plt.title(str(input_region)+'_'+str(cell_type)+'_normalized NMDA-AMPA traces')
    plt.legend(loc='upper right')
    if save_plots==True:
        #plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_norm.pdf', dpi=300)
        plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_norm.png')
    if show_plots == True:
        plt.show()
        
#####################################################################################################        
def ratio_average_trace(file_dict, input_region, cell_type, save_traces):
    ampa_list_traces=[]
    ampa_traces=[]
    ampa_peaks=[]
    nmda_list_traces=[]
    nmda_traces=[]
    nmda_peaks=[]
    for key, value in sorted(file_dict.items()):
        if str('norm') not in key:
            if str('AMPA') in key:
                ampa_list_traces = np.append(ampa_list_traces, key)
                ampa_traces = np.append(ampa_traces, value)
                #AMPA peaks correspond to the min value
                ampa_peaks = np.append(ampa_peaks,np.min(value))   
                
            if str('NMDA') in key: 
                nmda_list_traces = np.append(nmda_list_traces, key)
                nmda_traces = np.append(nmda_traces, value)
                #NMDA peaks correspond to the mean between 150 and 160 ms
                nmda_peaks = np.append(nmda_peaks,np.mean(value[750:800]))   

    if len(ampa_list_traces)==len(nmda_list_traces): #checking the lists
        #calculate the NMDA/AMPA ratio
        if str(input_region)==('M1-contra') and str(cell_type)=='FS':
            print('NMDA/AMPA ratio is zero')
            ratio_values=0
        elif str(input_region)==('M1-ipsi') and str(cell_type)=='FS':
            index_ratio_cells=[1,8,11] #corresponging to 'i040', 'i085' and 'i100'
            ratio_values=[]
            for i in index_ratio_cells:
                ratio_values=np.append(ratio_values,np.abs(nmda_peaks[i]/ampa_peaks[i])) 
        elif str(input_region)==('S1') and str(cell_type)=='FS':
            index_ratio_cells=[2,3] #corresponging to 'i152' and 'i155'
            ratio_values=[]
            for i in index_ratio_cells:
                ratio_values=np.append(ratio_values,np.abs(nmda_peaks[i]/ampa_peaks[i]))         
        
        else:         
            ratio_values = np.abs(nmda_peaks/ampa_peaks) #ratio cell by cell           
        ratio_values_avg = np.mean(ratio_values)     #average ratio
        ratio_values_std = np.std(ratio_values)      #std ratio 
    else:
        print('#ampa traces is different from #nmda trace - please check') 
        import pdb
        pdb.set_trace() 

    matrix_ampa=np.reshape(ampa_traces,(len(ampa_list_traces),5000))
        
    if str(input_region)==('M1-ipsi') and str(cell_type)=='ChIN':
        small_matrix_ampa=[matrix_ampa[0],matrix_ampa[3]] #corresponding to 'i122' and 'i134'
        ampa_avg_trace=np.mean(small_matrix_ampa,0)
        ampa_sem_trace=stats.sem(small_matrix_ampa)    
    
    else:
        ampa_avg_trace=np.mean(matrix_ampa,0)
        ampa_sem_trace=stats.sem(matrix_ampa)

    matrix_nmda=np.reshape(nmda_traces,(len(nmda_list_traces),5000))

    if str(input_region)==('M1-contra') and str(cell_type)=='FS':
        print('FS cells do not express NMDA')
        nmda_avg_trace=np.zeros(5000)
        nmda_sem_trace=np.zeros(5000)      

    elif str(input_region)==('M1-ipsi') and str(cell_type)=='FS':
        small_matrix_nmda=[matrix_nmda[1],matrix_nmda[8],matrix_nmda[10]]
        nmda_avg_trace=np.mean(small_matrix_nmda,0)
        nmda_sem_trace=stats.sem(small_matrix_nmda)     
    elif str(input_region)==('S1') and str(cell_type)=='FS':
        small_matrix_nmda=[matrix_nmda[2],matrix_nmda[3]]
        nmda_avg_trace=np.mean(small_matrix_nmda,0)
        nmda_sem_trace=stats.sem(small_matrix_nmda) 
    elif str(input_region)==('PF') and str(cell_type)=='FS':
        print('FS cells do not express NMDA')
        nmda_avg_trace=np.zeros(5000)
        nmda_sem_trace=np.zeros(5000)         

    else:
        nmda_avg_trace=np.mean(matrix_nmda,0)
        nmda_sem_trace=stats.sem(matrix_nmda)
    
    if save_traces==True:
    
        new_file1= open('avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_avg_AMPA.txt','w') 
        np.savetxt(new_file1, ampa_avg_trace)
        new_file1.close()

        new_file2= open('avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_avg_NMDA.txt','w') 
        np.savetxt(new_file2, nmda_avg_trace)
        new_file2.close()
        
        new_file3= open('avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_sem_AMPA.txt','w') 
        np.savetxt(new_file3, ampa_sem_trace)
        new_file3.close()

        new_file4= open('avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_sem_NMDA.txt','w') 
        np.savetxt(new_file4, nmda_sem_trace)
        new_file4.close()


        #saving ratio
        f = open('avg_sem_ratio/'+str(input_region)+'_'+str(cell_type)+'_ratio.txt','w') 
        f.write("ratio= " )
        f.write('%f' % ratio_values_avg )
        f.write('\n' "ratio_std= " )
        f.write('%f' % ratio_values_std)
        f.close()    
    
    return ratio_values_avg, ratio_values_std, ampa_avg_trace, ampa_sem_trace, nmda_avg_trace, nmda_sem_trace

#####################################################################################################        
def plot_ratio_average_trace(file_dict, input_region, cell_type, save_plots, show_plots):   

    #calling ratio_average_trace function
    ratio_values_avg, ratio_values_std, ampa_avg_trace, ampa_sem_trace, nmda_avg_trace, nmda_sem_trace = ratio_average_trace(file_dict, input_region, cell_type, save_traces=False)  

    #selecting right color depending on cell type
    if str(cell_type) == 'dSPN':
        sel_color='orangered'
    if str(cell_type) == 'iSPN':
        sel_color='blue'
    if str(cell_type) == 'FS':
        sel_color='green'
    if str(cell_type) == 'LTS':
        sel_color='darkcyan'
    if str(cell_type) == 'ChIN':
        sel_color='darkmagenta'

    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 
    matplotlib.rcParams.update({'font.size': 12})
    
    # defining time vector
    time=np.linspace(0,1000,5000)      
    
    plt.figure('summary')    
    plt.title("%s - %s - NMDA-AMPA summary" %(str(input_region),str(cell_type)), fontsize=12)
    plt.plot(time, ampa_avg_trace, color=sel_color,label=str(cell_type))
    plt.fill_between(time, ampa_avg_trace - ampa_sem_trace , ampa_avg_trace + ampa_sem_trace , alpha=0.25, facecolor=sel_color, zorder=2)

    if nmda_avg_trace.all()!=0:
        plt.plot(time, nmda_avg_trace, color=sel_color)  
        plt.fill_between(time, nmda_avg_trace - nmda_sem_trace , nmda_avg_trace + nmda_sem_trace , alpha=0.25, facecolor=sel_color, zorder=2)  
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')    
    plt.legend()
        
    if save_plots==True:
        #plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_summary.pdf', dpi=300)
        plt.savefig('figures/'+str(input_region)+'_'+str(cell_type)+'_NMDA-AMPA_summary.png')
    if show_plots == True:
        plt.show()

#####################################################################################################        
def plot_summary(input_region,save_plots, show_plots):   

    if input_region == str('M1-contra'):
        cell_type=['dSPN','iSPN','FS'] 
    elif input_region == str('M1-ipsi'):
        cell_type=['dSPN','iSPN','FS','LTS','ChIN']    
    elif input_region == str('S1'):
        cell_type=['dSPN','iSPN','FS','ChIN']    
    elif input_region == str('PF'):
        cell_type=['dSPN','iSPN','FS','ChIN']  


    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 
    matplotlib.rcParams.update({'font.size': 12})
    
    # defining time vector
    time=np.linspace(0,1000,5000)   
    
    for cell in cell_type:
        path = '../data/'+str(input_region)+'/'+str(cell)
        # creating a dictionary containing trace name and values    
        file_dict=create_file_dict(path, input_region, cell)
        #calling ratio_average_trace function
        ratio_values_avg, ratio_values_std, ampa_avg_trace, ampa_sem_trace, nmda_avg_trace, nmda_sem_trace = ratio_average_trace(file_dict, input_region, cell, save_traces=False) 
                

        #selecting right color depending on cell type
        if str(cell) == 'dSPN':
            sel_color='orangered'
        if str(cell) == 'iSPN':
            sel_color='blue'
        if str(cell) == 'FS':
            sel_color='green'
        if str(cell) == 'LTS':
            sel_color='darkorange'
        if str(cell) == 'ChIN':
            sel_color='darkmagenta'

        plt.figure('summary')    
        plt.plot(time, ampa_avg_trace, color=sel_color,label=str(cell))
        plt.fill_between(time, ampa_avg_trace - ampa_sem_trace , ampa_avg_trace + ampa_sem_trace , alpha=0.25, facecolor=sel_color, zorder=2)

        if nmda_avg_trace.all()!=0:
            plt.plot(time, nmda_avg_trace, color=sel_color)  
            plt.fill_between(time, nmda_avg_trace - nmda_sem_trace , nmda_avg_trace + nmda_sem_trace , alpha=0.25, facecolor=sel_color, zorder=2)          
        
        
    plt.title("%s - NMDA-AMPA summary" %(str(input_region)), fontsize=12)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')    
    plt.legend()
        
    if save_plots==True:
        #plt.savefig('figures/'+str(input_region)+'_NMDA-AMPA_summary.pdf', dpi=300)
        plt.savefig('figures/'+str(input_region)+'_NMDA-AMPA_summary.png', dpi=300)
    if show_plots == True:
        plt.show()        
        
#####################################################################################################        
def plot_summary_AMPA(input_region,save_plots, show_plots):   

    if input_region == str('M1-contra'):
        cell_type=['dSPN','iSPN','FS'] 
    elif input_region == str('M1-ipsi'):
        cell_type=['dSPN','iSPN','FS','LTS','ChIN']    
    elif input_region == str('S1'):
        cell_type=['dSPN','iSPN','FS','ChIN']    
    elif input_region == str('PF'):
        cell_type=['dSPN','iSPN','FS','ChIN']  


    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 
    matplotlib.rcParams.update({'font.size': 12})
    
    # defining time vector
    time=np.linspace(0,1000,5000)   
    
    for cell in cell_type:
        path = '../data/'+str(input_region)+'/'+str(cell)
        # creating a dictionary containing trace name and values    
        file_dict=create_file_dict(path, input_region, cell)
        #calling ratio_average_trace function
        ratio_values_avg, ratio_values_std, ampa_avg_trace, ampa_sem_trace, nmda_avg_trace, nmda_sem_trace = ratio_average_trace(file_dict, input_region, cell, save_traces=False) 
                

        #selecting right color depending on cell type
        if str(cell) == 'dSPN':
            sel_color='orangered'
        if str(cell) == 'iSPN':
            sel_color='blue'
        if str(cell) == 'FS':
            sel_color='green'
        if str(cell) == 'LTS':
            sel_color='darkorange'
        if str(cell) == 'ChIN':
            sel_color='darkmagenta'

        plt.figure('summary')    
        plt.plot(time, ampa_avg_trace, color=sel_color,label=str(cell))
        plt.fill_between(time, ampa_avg_trace - ampa_sem_trace , ampa_avg_trace + ampa_sem_trace , alpha=0.25, facecolor=sel_color, zorder=2)

              
        
    plt.title("%s - AMPA summary" %(str(input_region)), fontsize=12)
    plt.xlim(80,160)
    plt.ylim(-7e-10,1e-10)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')    
    plt.legend()
        
    if save_plots==True:
        #plt.savefig('figures/'+str(input_region)+'_AMPA_summary.pdf', dpi=300)
        plt.savefig('figures/'+str(input_region)+'_AMPA_summary.png')
    if show_plots == True:
        plt.show()          
        
#####################################################################################################        
def plot_summary_NMDA(input_region,save_plots, show_plots):   

    if input_region == str('M1-contra'):
        cell_type=['dSPN','iSPN','FS'] 
    elif input_region == str('M1-ipsi'):
        cell_type=['dSPN','iSPN','FS','LTS','ChIN']    
    elif input_region == str('S1'):
        cell_type=['dSPN','iSPN','FS','ChIN']    
    elif input_region == str('PF'):
        cell_type=['dSPN','iSPN','FS','ChIN']  


    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 
    matplotlib.rcParams.update({'font.size': 12})
    
    # defining time vector
    time=np.linspace(0,1000,5000)   
    
    for cell in cell_type:
        path = '../data/'+str(input_region)+'/'+str(cell)
        # creating a dictionary containing trace name and values    
        file_dict=create_file_dict(path, input_region, cell)
        #calling ratio_average_trace function
        ratio_values_avg, ratio_values_std, ampa_avg_trace, ampa_sem_trace, nmda_avg_trace, nmda_sem_trace = ratio_average_trace(file_dict, input_region, cell, save_traces=False) 
                

        #selecting right color depending on cell type
        if str(cell) == 'dSPN':
            sel_color='orangered'
        if str(cell) == 'iSPN':
            sel_color='blue'
        if str(cell) == 'FS':
            sel_color='green'
        if str(cell) == 'LTS':
            sel_color='darkorange'
        if str(cell) == 'ChIN':
            sel_color='darkmagenta'

        plt.figure('summary')    
        if nmda_avg_trace.all()!=0:
            plt.plot(time, nmda_avg_trace, color=sel_color,label=str(cell))  
            plt.fill_between(time, nmda_avg_trace - nmda_sem_trace , nmda_avg_trace + nmda_sem_trace , alpha=0.25, facecolor=sel_color, zorder=2)          
        
        
    plt.title("%s - NMDA summary" %(str(input_region)), fontsize=12)
    plt.xlim(40,600)
    plt.ylim(-1e-10,7e-10)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')    
    plt.legend()
        
    if save_plots==True:
        #plt.savefig('figures/'+str(input_region)+'_NMDA_summary.pdf', dpi=300)
        plt.savefig('figures/'+str(input_region)+'_NMDA_summary.png')
    if show_plots == True:
        plt.show()          
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
