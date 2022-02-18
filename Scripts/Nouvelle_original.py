import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize, Bounds, LinearConstraint
from PA_functions import ratios, ratios_sd, model, preprocess_nouvelle, loss_function_1, loss_function_2
import winsound

###############################     MODIFY!!!  ##################################
folder = './../Data_Base/NGS_J19_DRK_11/'                                 
file = 'NGS_J19_DRK_11'
nSM = 1
ratios_type = '11'
use_all_peaks = 1
peaks_to_analyze = 11
pp = 1
max_cv_peaks = 20                           # Max intrasample CV for peaks
max_cv_samples = 15                         # Max intrasample CV for repetitions
#################################################################################

################    HOW MANY END-MEMBERS, HOW MANY MIXTURES?    ########################
dataset = pd.read_csv(folder+"/"+file+".csv").set_index('Mix')
nEM = len([i for i in dataset.index.unique().values.tolist() if i.startswith('EM')])
nMix = len([i for i in dataset.index.unique().values.tolist() if i.startswith('M')])

################    PREPROCESSING    ########################
if use_all_peaks == 0:
    peaks = dataset.iloc[:,0:peaks_to_analyze+1]
else:
    peaks = dataset.iloc[:,:]
if pp == 1: 
    peaks, ignorar = preprocess_nouvelle(peaks,pp,max_cv_peaks,max_cv_samples)
else:
    ignorar = []

################    WHERE IS THE OPERATIONS FILE?    ########################
operations = 'operations_'+ratios_type+'.txt'
if use_all_peaks == 0: operations = 'operations'+str(peaks_to_analyze)+'.txt'
operations_path=folder+operations

# Define end members and mixtures list, then create combinations
end_members = ['EM'+str(x+1) for x in range(nEM)]
mixtures = ['M' + str(x+1) for x in range(nMix)]
comb = list(combinations(mixtures, nSM))
real_results = pd.read_csv(folder+"/p_"+file+".csv", delimiter=',')
real_results_values = real_results.values[nEM:,1:]

errors = []
estimates = []

for i in comb:
    # Create intermediate variables
    synthetic_mixtures, unknown_mixtures = [], []
    [synthetic_mixtures.append(x) for x in i[:]]
    [unknown_mixtures.append(x) for x in mixtures if x not in i[:]]

    # Read dataset, compute ratios, read proportions
    em_peaks = peaks.loc[end_members]
    syn_mix_peaks = peaks.loc[synthetic_mixtures]
    um_peaks = peaks.loc[unknown_mixtures]
    ratios_results, w_r, tipo, def_operations = ratios(operations_path, peaks) #Includes weight ratios
    syn_mix_ratios = ratios_results.loc[synthetic_mixtures]
    um_ratios = ratios_results.loc[unknown_mixtures]
    proportions = real_results.set_index('Mix').loc[synthetic_mixtures]

    # Compute number of mixtures, unknown mixtures, end members and ratios
    nUM = len(unknown_mixtures)                     # Number of Unknown Mixtures
    nratios = len(ratios_results.columns)           # Number of ratios

    #Compute means and SD (variance)
    em_peaks_mean = em_peaks.mean(level=0)
    syn_mix_peaks_mean = syn_mix_peaks.mean(level=0)
    syn_mix_peaks_variance = syn_mix_peaks.var(level=0,ddof=1)
    syn_mix_ratios_mean = syn_mix_ratios.mean(level=0)
    um_ratios_mean = um_ratios.mean(level=0)
    syn_mix_ratios_sd = ratios_sd(operations_path, def_operations, syn_mix_ratios_mean, syn_mix_peaks_mean, syn_mix_peaks_variance, tipo)

    MR = np.ones(nEM-1)
    C, it = 1, 1

    while (C>0.01) and (it<30):
        Mix_Q = w_r/syn_mix_ratios_sd
        res = minimize(loss_function_1, MR, method='nelder-mead', options={'xatol': 1e-4}, args = (nSM, nratios, proportions.values, em_peaks_mean.values, def_operations, tipo, nEM, Mix_Q, syn_mix_ratios_mean.values, ignorar))

        MR = res.x
        syn_Mix_ratios_modeled = model(MR, nSM, nratios, proportions.values, em_peaks_mean.values, def_operations, tipo, nEM)

        #NEW SD_r
        B = (np.abs(syn_mix_ratios_mean.values-syn_Mix_ratios_modeled)/nSM+syn_mix_ratios_sd)/2
        C = np.mean(abs((syn_mix_ratios_sd-B)/(syn_mix_ratios_sd)))
        it += 1
        # print (it, C)
        syn_mix_ratios_sd = B

    MR1 = np.resize(MR,(1,nEM))
    MR1[0,nEM-1]=1

    linear_constraint = LinearConstraint(np.ones(nEM-1), [0], [100])
    bounds = Bounds(np.zeros(nEM-1), np.ones(nEM-1)*100)

    for um in range(1,nUM+1):   
        UM_ratios=um_ratios_mean.iloc[um-1]    
        xi = np.ones(nEM-1)*100/(nEM)

        Mix_Q = w_r/syn_mix_ratios_sd
        res = minimize(loss_function_2, xi, method='SLSQP',constraints=linear_constraint, bounds=bounds, args = (nratios, em_peaks_mean.values, tipo, nEM, MR, def_operations, Mix_Q, UM_ratios.values, ignorar))

        np.set_printoptions(suppress=True, precision=1)

        if nEM == 2:
            print(unknown_mixtures[um-1], res.x[0], 100-res.x[0])
            errors.append((abs(real_results_values[int((unknown_mixtures[um-1])[1:])-1,0]-res.x[0])+abs(real_results_values[int((unknown_mixtures[um-1])[1:])-1,1]-(100-res.x[0])))/nEM)
        elif nEM == 3:
            print(unknown_mixtures[um-1], res.x[0], res.x[1], 100-res.x[0]-res.x[1])
            estimates.append([unknown_mixtures[um-1], res.x[0], res.x[1], 100-res.x[0]-res.x[1]])
            errors.append((abs(real_results_values[int((unknown_mixtures[um-1])[1:])-1,0]-res.x[0])+abs(real_results_values[int((unknown_mixtures[um-1])[1:])-1,1]-res.x[1])+abs(real_results_values[int((unknown_mixtures[um-1])[1:])-1,2]-(100-res.x[0]-res.x[1])))/3)
        else: print('number of end-members not implemented')


np.savetxt('../Results/Comparison_methods/Nouvelle/original_'+ratios_type+'/Errors_'+str(nSM)+'CM.txt',errors)
pd.DataFrame(estimates).to_csv('../Results/Comparison_methods/Nouvelle/original_'+ratios_type+'/Estimates_'+str(nSM)+'CM.txt', header = None, index = None)


winsound.Beep(2000, 5000)