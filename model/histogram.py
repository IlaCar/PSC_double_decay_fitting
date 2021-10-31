import json
import matplotlib.pyplot as plt
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="Plot NMDA and AMPA currents of a selected input region")    
parser.add_argument("--region", help="select input region",
                  choices=["M1-contra","M1-ipsi", "S1", "PF"])
args = parser.parse_args()

input_region = args.region

print("Selected input region : " + args.region)


plt.figure(figsize=(9,6))
cell_type1='dSPN'
with open('json_files/'+str(input_region)+'_'+str(cell_type1)+'_fitting.json', 'r') as fp:
    info1 = json.load(fp)

cell_type2='iSPN'
with open('json_files/'+str(input_region)+'_'+str(cell_type2)+'_fitting.json', 'r') as fp:
    info2 = json.load(fp)

cell_type3='FS'
with open('json_files/'+str(input_region)+'_'+str(cell_type3)+'_fitting.json', 'r') as fp:
    info3 = json.load(fp)

if str(input_region)=='S1' or str(input_region)=='PF' or str(input_region)=='M1-ipsi':
    cell_type4='ChIN'
    with open('json_files/'+str(input_region)+'_'+str(cell_type4)+'_fitting.json', 'r') as fp:
        info4 = json.load(fp)

if str(input_region)=='M1-ipsi':
    cell_type5='LTS'
    with open('json_files/'+str(input_region)+'_'+str(cell_type5)+'_fitting.json', 'r') as fp:
        info5 = json.load(fp)


# set width of bar
barWidth = 0.25

if str(input_region)=='M1-contra':

    bars1 = [info1['data']['test_my_fitting_ampa'], info1['data']['test_my_fitting_nmda'],\
            info2['data']['test_my_fitting_ampa'], info2['data']['test_my_fitting_nmda'],
            info3['data']['test_my_fitting_ampa']]

    bars2 = [info1['data']['test_single_fitting_ampa'], info1['data']['test_single_fitting_nmda'],\
            info2['data']['test_single_fitting_ampa'], info2['data']['test_single_fitting_nmda'],
            info3['data']['test_single_fitting_ampa']]

    bars3 = [info1['data']['test_w_fitting_ampa'], info1['data']['test_w_fitting_nmda'],\
            info2['data']['test_w_fitting_ampa'], info2['data']['test_w_fitting_nmda'],
            info3['data']['test_w_fitting_ampa']]

if str(input_region)=='S1':
    bars1 = [info1['data']['test_my_fitting_ampa'], info1['data']['test_my_fitting_nmda'],\
            info2['data']['test_my_fitting_ampa'], info2['data']['test_my_fitting_nmda'],
            info3['data']['test_my_fitting_ampa'],info3['data']['test_my_fitting_nmda'],
            info4['data']['test_my_fitting_ampa'],info4['data']['test_my_fitting_nmda']]

    bars2 = [info1['data']['test_single_fitting_ampa'], info1['data']['test_single_fitting_nmda'],\
            info2['data']['test_single_fitting_ampa'], info2['data']['test_single_fitting_nmda'],
            info3['data']['test_single_fitting_ampa'],info3['data']['test_single_fitting_nmda'],
            info4['data']['test_single_fitting_ampa'],info4['data']['test_single_fitting_nmda']]

    bars3 = [info1['data']['test_w_fitting_ampa'], info1['data']['test_w_fitting_nmda'],\
            info2['data']['test_w_fitting_ampa'], info2['data']['test_w_fitting_nmda'],
            info3['data']['test_w_fitting_ampa'],info3['data']['test_w_fitting_nmda'],
            info4['data']['test_w_fitting_ampa'],info4['data']['test_w_fitting_nmda']]

if str(input_region)=='PF':
    bars1 = [info1['data']['test_my_fitting_ampa'], info1['data']['test_my_fitting_nmda'],\
            info2['data']['test_my_fitting_ampa'], info2['data']['test_my_fitting_nmda'],
            info3['data']['test_my_fitting_ampa'],
            info4['data']['test_my_fitting_ampa'],info4['data']['test_my_fitting_nmda']]

    bars2 = [info1['data']['test_single_fitting_ampa'], info1['data']['test_single_fitting_nmda'],\
            info2['data']['test_single_fitting_ampa'], info2['data']['test_single_fitting_nmda'],
            info3['data']['test_single_fitting_ampa'],
            info4['data']['test_single_fitting_ampa'],info4['data']['test_single_fitting_nmda']]

    bars3 = [info1['data']['test_w_fitting_ampa'], info1['data']['test_w_fitting_nmda'],\
            info2['data']['test_w_fitting_ampa'], info2['data']['test_w_fitting_nmda'],
            info3['data']['test_w_fitting_ampa'],
            info4['data']['test_w_fitting_ampa'],info4['data']['test_w_fitting_nmda']]


if str(input_region)=='M1-ipsi':
    bars1 = [info1['data']['test_my_fitting_ampa'], info1['data']['test_my_fitting_nmda'],\
            info2['data']['test_my_fitting_ampa'], info2['data']['test_my_fitting_nmda'],
            info3['data']['test_my_fitting_ampa'],info3['data']['test_my_fitting_nmda'],
            info4['data']['test_my_fitting_ampa'],info4['data']['test_my_fitting_nmda'],
            info5['data']['test_my_fitting_ampa'],info5['data']['test_my_fitting_nmda']]

    bars2 = [info1['data']['test_single_fitting_ampa'], info1['data']['test_single_fitting_nmda'],\
            info2['data']['test_single_fitting_ampa'], info2['data']['test_single_fitting_nmda'],
            info3['data']['test_single_fitting_ampa'],info3['data']['test_single_fitting_nmda'],
            info4['data']['test_single_fitting_ampa'],info4['data']['test_single_fitting_nmda'],
            info5['data']['test_single_fitting_ampa'],info5['data']['test_single_fitting_nmda']]

    bars3 = [info1['data']['test_w_fitting_ampa'], info1['data']['test_w_fitting_nmda'],\
            info2['data']['test_w_fitting_ampa'], info2['data']['test_w_fitting_nmda'],
            info3['data']['test_w_fitting_ampa'],info3['data']['test_w_fitting_nmda'],
            info4['data']['test_w_fitting_ampa'],info4['data']['test_w_fitting_nmda'],
            info5['data']['test_w_fitting_ampa'],info5['data']['test_w_fitting_nmda']]


#computing the ratios
fitting_ratio_o=np.zeros(len(bars1))
fitting_ratio_w=np.zeros(len(bars1))
for i in range(len(bars1)):
    fitting_ratio_o[i]=bars2[i]/bars1[i]
    fitting_ratio_w[i]=bars3[i]/bars1[i]

print('RMSE single/double',fitting_ratio_o)
print('RMSE weighted/double',fitting_ratio_w)


print('min and max fitting_ratio_o:', np.min(fitting_ratio_o),np.max(fitting_ratio_o))
print('min and max fitting_ratio_w:', np.min(fitting_ratio_w),np.max(fitting_ratio_w))


# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#377eb8', width=barWidth, edgecolor='white', label='double_exp_fitting')
plt.bar(r2, bars2, color='#4daf4a', width=barWidth, edgecolor='white', label='single_exp')
plt.bar(r3, bars3, color='#984ea3', width=barWidth, edgecolor='white', label='double_exp_w')

 
# Add xticks on the middle of the group bars
if str(input_region)=='M1-contra':
    plt.xticks([r + barWidth for r in range(len(bars1))], ['dSPN' +'\n' + 'AMPA', 'dSPN' +'\n' + 'NMDA', 'iSPN' +'\n' + 'AMPA', 'iSPN' +'\n' + 'NMDA', 'FS' +'\n' + 'AMPA'])

if str(input_region)=='S1':
    plt.xticks([r + barWidth for r in range(len(bars1))], ['dSPN' +'\n' + 'AMPA', 'dSPN' +'\n' + 'NMDA', 'iSPN' +'\n' + 'AMPA', 'iSPN' +'\n' + 'NMDA', 'FS' +'\n' + 'AMPA', 'FS' +'\n' + 'NMDA', 'ChIN' +'\n' + 'AMPA', 'ChIN' +'\n' + 'NMDA'])

if str(input_region)=='PF':
    plt.xticks([r + barWidth for r in range(len(bars1))], ['dSPN' +'\n' + 'AMPA', 'dSPN' +'\n' + 'NMDA', 'iSPN' +'\n' + 'AMPA', 'iSPN' +'\n' + 'NMDA', 'FS' +'\n' + 'AMPA', 'ChIN' +'\n' + 'AMPA', 'ChIN' +'\n' + 'NMDA'])

if str(input_region)=='M1-ipsi':
    plt.xticks([r + barWidth for r in range(len(bars1))], ['dSPN' +'\n' + 'AMPA', 'dSPN' +'\n' + 'NMDA', 'iSPN' +'\n' + 'AMPA', 'iSPN' +'\n' + 'NMDA', 'FS' +'\n' + 'AMPA', 'FS' +'\n' + 'NMDA', 'ChIN' +'\n' + 'AMPA', 'ChIN' +'\n' + 'NMDA','LTS' +'\n' + 'AMPA', 'LTS' +'\n' + 'NMDA'])


plt.ylabel('RMSE', size='12') 
plt.ylim(0,0.055)
plt.legend(loc='upper left',frameon=True,edgecolor='black',fontsize='medium')
plt.title(str(input_region) + '  -  fitting comparison')
plt.savefig('histogram/'+str(input_region)+'_fitting_comparison.png')
#plt.savefig('histogram/'+str(input_region)+'_fitting.pdf', dpi=600)


plt.show()


