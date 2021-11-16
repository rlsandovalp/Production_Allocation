import numpy as np 
import pandas as pd 
import random
from scipy import stats
from scipy.optimize import LinearConstraint, Bounds, fmin_slsqp
from PA_functions import *
from scipy.stats import t

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def residuals(x,A,b):
    hola = (A @ x) -b
    return np.linalg.norm(hola)

def eq_cond(x, *args):
    return sum(x) - 1.0

######################   DEFINE VARIABLES   #########################
folder = './../Data_Base/G510_FT3H_MSK2H/'                                 
file = 'G510_FT3H_MSK2H_corrected'

use_all_peaks = 1                           # Use all peaks? (1 Yes, 0 No)
peaks_to_analyze = 11                       # How many peaks shall be used (if use_all_peaks == 1 this parameter is not read)
repetitions = 1000                          # Number of random lists to be generated in the bootstrapping
tolerance = 0.20                            # Tolerance of the error in the bootstrapping

pp = 1                                      # Preprocess? (1 Yes, 0 No)
normalize = 1                               # Normalize the peaks? (1 Yes, 0 No)
max_cv_peaks = 20                           # Max intrasample CV for peaks
max_cv_samples = 10                         # Max intrasample CV for repetitions

################    HOW MANY END-MEMBERS, HOW MANY MIXTURES?    ########################
dataset = pd.read_csv(folder+"/"+file+".csv").set_index('Mix')
nEM = len([i for i in dataset.index.unique().values.tolist() if i.startswith('EM')])
nMix = len([i for i in dataset.index.unique().values.tolist() if i.startswith('M')])

################    PREPROCESSING    ########################
if use_all_peaks == 0: 
    peaks = dataset.iloc[:,0:peaks_to_analyze+1]
else:
    peaks = dataset
if pp == 1: 
    peaks = preprocess(peaks,pp,max_cv_peaks,max_cv_samples)

######################   INITIALIZE VARIABLES AND READ DATA   #########################
rango1 = list((peaks).columns[1:])
peaks = peaks.set_index('Mix').mean(level=0)
results = np.zeros((nMix,nEM))
x0 = np.ones(nEM)
saving = np.empty((repetitions,nEM*nMix))
saving_last = np.empty((repetitions,nEM*nMix))
mixtures = [x+1 for x in range(nMix)]

################# DO THIS FOR EACH MIXTURE #########################
for mixture in mixtures:
    print('Mixture: ' + str(mixture))
    ################ CREATE THE SPACE OF ANALYSIS (i.e. select end members and mixture) #################
    space = [x for x in range(nEM)]
    space.append(mixture+nEM-1)
    peaks_mixture = peaks.iloc[space]
    deleted = []
    rango = rango1.copy()
    veces = 515615
    pico = rango[-1]
    ######### While the number of peaks falling outside a given tolerance is large ###########
    is_the_first_time = 1
    while veces > repetitions*0.2:
        rango.remove(pico)                                  # Remove the bad peak from the range
        if len(deleted)>0.6*len(rango1): break
        all_list, bad_peaks, EM = [], [], []                # Initialize the lists that you will use in the algorithm
        for i in range(nEM): EM.append([])
        ######### DO THIS PROCEDURE MANY MANY TIMES TO BE SURE THAT THE RANDOM RESULT IS RELIABLE ###########
        for i in range(1,repetitions+1):
            random.seed(1)
            randomlist = random.choices(rango, k=int(len(rango)*0.7))        # Create a random range
            peaks_rango = peaks_mixture[randomlist]
            all_list.append(randomlist)                             # Save that range
            p_emr, p_mr = f_normalize(normalize,peaks_rango.values,nEM)
            A = p_emr
            cop = np.copy(p_mr)
            b = p_mr[:,0]
            # calcula = fmin_slsqp(residuals, x0, eqcons = [eq_cond], bounds = [(0,1),(0,1),(0,1)], args = (A, b), iprint = -1)
            calcula = fmin_slsqp(residuals, x0, args = (A, b), iprint = -1)
            EM[0].append(calcula[0])
            EM[1].append(calcula[1])
            # EM[2].append(calcula[2])
        if is_the_first_time==1:
            saving[:,(mixture-1)*nEM] = np.array(EM[0])
            saving[:,(mixture-1)*nEM+1] = np.array(EM[1])
            # saving[:,(mixture-1)*nEM+2] = np.array(EM[2])
        is_the_first_time=0
        a = []
        for i in range(nEM): a.append(sum(EM[i]) / len(EM[i]))
        

        for i in range(repetitions-1):            # For each random list check if the result is bad (if bad, save the peaks of that list in bad_peaks)
            for j in range(nEM):
                if (EM[j][i] < a[j]-tolerance) or (EM[j][i] > a[j]+tolerance):
                    bad_peaks.append(all_list[i])
    
        if len(np.array(bad_peaks)) == 0:           # If no bad_peaks, wonderful
            break 
        picoa, veces = stats.mode(np.array(bad_peaks), axis = None)     # If bad peaks, identify the most problematic one
        pico = picoa[0]
        deleted.append(pico)
        if veces<repetitions*0.2:
            saving_last[:,(mixture-1)*nEM] = np.array(EM[0])
            saving_last[:,(mixture-1)*nEM+1] = np.array(EM[1])
            # saving_last[:,(mixture-1)*nEM+2] = np.array(EM[2])

    peaks_pa = peaks_mixture[rango]
    p_emr, p_mr = f_normalize(normalize,peaks_pa.values,nEM) # Take the peaks, normalize them if you want and separate EMs from mixture
    A = p_emr
    b = p_mr[:,0]
    calcula = fmin_slsqp(residuals, x0, bounds = [(0,1),(0,1)], eqcons = [eq_cond], args = (A, b), iprint = -1)
    print('Deleted: ', deleted)
    results[mixture-1,:] = calcula*100

# plt.show()
X_todos = np.transpose(results)/100
X_todos1 = X_todos
conf_int_inf = X_todos - 5
conf_int_sup = X_todos + 5
for i in mixtures:
    print('M'+str(i),X_todos1[0,i-1]*100,X_todos1[1,i-1]*100)

real_results = pd.read_csv(folder+"/p_"+file+".csv", delimiter=',').values[nEM:,1:]
print(np.mean(abs(real_results - np.transpose(X_todos1*100))))
plot_results(X_todos1,conf_int_inf,conf_int_sup,real_results)
# np.savetxt('../Results/Paper/McCaffrey_improved/estimations.csv',np.r_[np.transpose(X_todos)*100,np.transpose(conf_int_inf)*100,np.transpose(conf_int_sup)*100])

