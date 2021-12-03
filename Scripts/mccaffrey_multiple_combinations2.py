import numpy as np 
import pandas as pd 
import random
from scipy import stats
from scipy.optimize import fmin_slsqp
from PA_functions import preprocess, f_normalize, plot_results

def residuals(x,A,b):
    hola = (A @ x) -b
    return np.linalg.norm(hola)

folder = './../Data_Base/NGS_J19_DRK/'                                 
file = 'NGS_J19_DRK'

use_all_peaks = 1                           # Use all peaks? (1 Yes, 0 No)
peaks_to_analyze = 11                       # How many peaks shall be used (if use_all_peaks == 1 this parameter is not read)

repetitions = 1000                          # Number of random lists to be generated in the bootstrapping
tolerance = 0.15                            # Tolerance of the error in the bootstrapping
rep_allowed_outside = 0.2                   # Percentage of the number of repetitions allowed to be outside of mean+-tolerance
max_peaks_deleted = 0.6                     # Percentage of peaks allowed to be deleted during bootstrapping
size_peak_boots = 0.7                       # Percentage of peaks used for bootstrapping

normalize = 1                               # Normalize the peaks? (1 Yes, 0 No)
pp = 1                                      # Preprocess? (1 Yes, 0 No)
max_cv_peaks = 20                           # Max intrasample CV for peaks
max_cv_samples = 10                         # Max intrasample CV for repetitions

unknown_mixture = 'M2'

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
rango1 = list(peaks.columns[1:])
peaks = peaks.set_index('Mix')
mixtures = [x+1 for x in range(nMix)]

################ CREATE THE SPACE OF ANALYSIS (i.e. select end members and mixture) #################

dataset.set_index('Mix', inplace = True)
reps_em1 = dataset.loc['EM1'].values.shape[0]
reps_em2 = dataset.loc['EM2'].values.shape[0]
reps_em3 = dataset.loc['EM3'].values.shape[0]
reps_mix = dataset.loc[unknown_mixture].values.shape[0]
peaks = dataset.loc['EM1'].values.shape[1]

combinations_results = np.zeros((reps_em1*reps_em2*reps_em3*reps_mix,nEM+1))

muchos_resultados = np.zeros((reps_em1*reps_em2*reps_em3*reps_mix,3))

hola = 0
for repetition_endmember_1 in range(reps_em1):
    for repetition_endmember_2 in range(reps_em2):
        for repetition_endmember_3 in range(reps_em3):
            for repetition_mixture in range(reps_mix):
                reduced_dataset_1 = dataset.loc['EM1'].iloc[[repetition_endmember_1]]
                reduced_dataset_2 = dataset.loc['EM2'].iloc[[repetition_endmember_2]]
                reduced_dataset_3 = dataset.loc['EM3'].iloc[[repetition_endmember_3]]
                reduced_dataset_4 = dataset.loc[unknown_mixture].iloc[[repetition_mixture]]
                reduced_dataset = pd.merge(reduced_dataset_1, reduced_dataset_2, how = 'outer')
                reduced_dataset = pd.merge(reduced_dataset, reduced_dataset_3, how = 'outer')
                reduced_dataset = pd.merge(reduced_dataset, reduced_dataset_4, how = 'outer')
                deleted = []
                rango = rango1.copy()
                veces = 515615
                pico = rango[-1]


                ######### While the number of peaks falling outside a given tolerance is large ###########
                while veces > repetitions*rep_allowed_outside:
                    rango.remove(pico)                                  # Remove the bad peak from the range
                    if len(deleted)>max_peaks_deleted*len(rango1): break
                    all_list, bad_peaks, EM = [], [], []                # Initialize the lists that you will use in the algorithm
                    for i in range(nEM): EM.append([])
                    ######### DO THIS PROCEDURE MANY MANY TIMES TO BE SURE THAT THE RANDOM RESULT IS RELIABLE ###########
                    for i in range(1, repetitions+1):
                        randomlist = random.choices(rango, k=int(len(rango)*size_peak_boots))        
                        super_reduced_dataset = reduced_dataset[randomlist]       # Select random peaks
                        all_list.append(randomlist)                             # Save that range
                        peaks_endmembers, peaks_mixtures = f_normalize(normalize,super_reduced_dataset.values,nEM)
                        mass_fractions = fmin_slsqp(residuals, np.ones(nEM), args = (peaks_endmembers, peaks_mixtures.reshape(-1)), iprint = -1)
                        for e in range(nEM): EM[e].append(mass_fractions[e])
                    average_results_bootstrapping = []
                    for i in range(nEM): average_results_bootstrapping.append(sum(EM[i]) / len(EM[i]))
                    
                    for i in range(repetitions-1):            # For each random list check if the result is bad (if bad, save the peaks of that list in bad_peaks)
                        for j in range(nEM):
                            if (abs (EM[j][i] - average_results_bootstrapping[j]) > tolerance):
                                bad_peaks.append(all_list[i])

                    if len(np.array(bad_peaks)) == 0:           # If no bad_peaks, wonderful
                        break
                    picoa, veces = stats.mode(np.array(bad_peaks), axis = None)     # If bad peaks, identify the most problematic one
                    pico = picoa[0]
                    deleted.append(pico)


                peaks_pa = reduced_dataset[rango]
                peaks_endmembers, peaks_mixtures = f_normalize(normalize,peaks_pa.values,nEM) # Take the peaks, normalize them if you want and separate EMs from mixture
                muchos_resultados[hola,:] = fmin_slsqp(residuals, np.ones(nEM), bounds = [(0,1),(0,1),(0,1)], args = (peaks_endmembers, peaks_mixtures.reshape(-1)), iprint = -1)
                muchos_resultados[hola,:] = muchos_resultados[0,:]/np.sum(muchos_resultados[0,:])
                hola = hola + 1
                print(hola,"/",reps_em1*reps_em2*reps_em3*reps_mix)
np.savetxt('../Results/'+file+'/McCaffrey/'+unknown_mixture+'.txt', muchos_resultados)