import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pandas as pd
from itertools import combinations
from nouvelle_functions import *
from PA_functions import plot_results, preprocess_nameless
from scipy.optimize import LinearConstraint

###############################     MODIFY!!!  ##################################
folder = 'G:/My Drive/05 - PhD/2_Product_Allocation/Data_Base/Third_group/NGS_J19_DRK/'      # Directory                           
file = 'NGS_J19_DRK'                        # Name of the file
simultaneous = 1
use_all_peaks = 1                           # Use all peaks? (1 Yes, 0 No)
peaks_to_analyze = 11                       # How many peaks use?
delta = 0.05                            
pp = 0                                      # Preprocess? (1 Yes, 0 No)
max_cv_samples = 10                         # Max intrasample CV for repetitions
max_cv_peaks = 30                           # Max asasfasdfasdfa
cv = 0.07
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
    peaks, ignorar = preprocess_nameless(peaks,pp,max_cv_samples, max_cv_peaks)
else:
    ignorar = []

################    WHERE ARE THE OPERATIONS?    ########################
operations = 'operations_all.txt'
if use_all_peaks == 0: operations = 'operations'+str(peaks_to_analyze)+'.txt'
operations_path=folder+operations

# Define end members and mixtures list, then create combinations
end_members = ['EM'+str(x+1) for x in range(nEM)]
mixtures = ['M' + str(x+1) for x in range(nMix)]
comb = list(combinations(mixtures, simultaneous))

# Create an array to storage the results
X_todos = np.zeros((nEM, nMix))

# For each combinations of mixtures
for ni,i in enumerate(comb):
    # Define the unknown mixtures, create end members array, create unknown mixtures array
    unknown_mixtures = [x for x in mixtures if x in i[:]]
    em_peaks = dataset.loc[end_members]
    um_peaks = dataset.loc[unknown_mixtures]

    # Convert the operations from the text file to an array easier to manipulate
    def_operations, tipo = convert_operations(operations_path)

    # Mean of end members and unknown mixtures arrays
    em_peaks_mean = em_peaks.mean(level=0)
    um_peaks_mean = um_peaks.mean(level=0)

    # Define vector of unknowns and initialize its values, give an initial value to CV, and define bounds for the optimization
    unknowns = np.ones((nEM)*simultaneous+nEM-1)  # [X1, X2, X3, ..., MR1, MR2] 
    lb = np.ones((nEM)*simultaneous+nEM-1)*(1-delta)
    ub = np.ones((nEM)*simultaneous+nEM-1)*(1+delta)
    lc = np.zeros((nEM)*simultaneous+nEM-1)
    unknowns[:-(nEM-1)] = 100/nEM
    lb[:-(nEM-1)] = 0
    ub[:-(nEM-1)] = 100
    lc[:-(nEM-1)] = 1
    
    bounds = Bounds(lb, ub)
    linear_constraint = LinearConstraint(lc.tolist(), [100], [100])

    # Minimize objective function to obtain values of unknowns
    res = minimize(ouropt4, unknowns, method = 'SLSQP', constraints=linear_constraint, bounds = bounds, args = (def_operations, em_peaks_mean.values, um_peaks_mean.values, tipo, cv, ignorar))
    unknowns = res.x
    
    # Print the results for each unknown mixture
    for mix in range(simultaneous):
        if nEM == 2:
            print(unknown_mixtures[mix], unknowns[mix*2], unknowns[mix*2+1])
            X_todos[0,ni] = unknowns[mix*2]
            X_todos[1,ni] = unknowns[mix*2+1]
        elif nEM ==3:
            print(unknown_mixtures[mix], unknowns[mix*2], unknowns[mix*2+1], unknowns[mix*2+2])
            X_todos[0,ni] = unknowns[mix*2]
            X_todos[1,ni] = unknowns[mix*2+1]
            X_todos[2,ni] = unknowns[mix*2+2]

########### Load the real results, compute estimates+-5% and plot  ################
real_results = pd.read_csv(folder+"/p_"+file+".csv", delimiter=',').values[nEM:,1:]
conf_int_sup = X_todos/100 + 5
conf_int_inf = X_todos/100 - 5
# plot_results(X_todos/100,conf_int_inf,conf_int_sup,real_results)
print(np.mean(abs(X_todos-np.transpose(real_results))))