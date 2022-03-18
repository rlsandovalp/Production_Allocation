import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pandas as pd
from itertools import combinations

from PA_functions import plot_results, preprocess_our, ouropt_1, ouropt_2, convert_operations, preprocess_our
from scipy.optimize import LinearConstraint
from scipy.optimize import fsolve

###############################     MODIFY!!!  ##################################
folder = './../Data_Base/NGS_J19_DRK/'                                 
file = 'NGS_J19_DRK'
use_all_peaks = 1                           # Use all peaks? (1 Yes, 0 No)
peaks_to_analyze = 12                       # How many peaks use?
delta = 0.05
pp = 1                                      # Preprocess? (1 Yes, 0 No)
max_cv_samples = 10                         # Max intrasample CV for repetitions
max_cv_peaks = 20                           # Max cv allowed for the repetitions of a peak
cv = 0.5
#################################################################################

################    HOW MANY END-MEMBERS, HOW MANY MIXTURES?    ########################
dataset = pd.read_csv(folder+'/'+file+".csv").set_index('Mix')
nEM = len([i for i in dataset.index.unique().values.tolist() if i.startswith('EM')])
nMix = len([i for i in dataset.index.unique().values.tolist() if i.startswith('M')])

################    PREPROCESSING    ########################
if use_all_peaks == 0:
    peaks = dataset.iloc[:,0:peaks_to_analyze+1]
else:
    peaks = dataset.iloc[:,:]
if pp == 1: 
    peaks, ignorar = preprocess_our(peaks, pp, max_cv_peaks, max_cv_samples)
else:
    ignorar = []

################    WHERE ARE THE OPERATIONS?    ########################
operations = 'operations_all.txt'
if use_all_peaks == 0: 
    operations = 'operations'+str(peaks_to_analyze)+'.txt'
operations_path=folder+'/'+operations

# Convert the operations from the text file to an array easier to manipulate
def_operations, tipo = convert_operations(operations_path)

# Define end members and mixtures list, then create combinations
end_members = ['EM'+str(x+1) for x in range(nEM)]
mixtures = ['M' + str(x+1) for x in range(nMix)]
comb = list(combinations(mixtures, 1))

# Create an array to storage the results
X_todos = np.zeros((nEM, nMix))

# For each combinations of mixtures
for i in range(nMix):
    # Define the unknown mixtures, create end members array, create unknown mixtures array
    unknown_mixtures = mixtures[i]

    # Average chromatograms
    em_peaks_mean = dataset.loc[end_members].mean(level = 0)
    um_peaks_mean = dataset.loc[unknown_mixtures].mean(level = 0)
    if dataset.loc[unknown_mixtures].shape[0] == dataset.loc[unknown_mixtures].size:
        um_peaks_mean = dataset.loc[unknown_mixtures]

    # Define vector of unknowns and initialize its values, give an initial value to CV, and define bounds for the optimization
    unknowns = np.ones(2*nEM-1)  # [X1, X2, X3, ..., MR1, MR2] 
    lb = np.ones((nEM)+nEM-1)*(1-delta)
    ub = np.ones((nEM)+nEM-1)*(1+delta)
    lc = np.zeros((nEM)+nEM-1)
    unknowns[:-(nEM-1)] = 100/nEM
    lb[:-(nEM-1)] = 0
    ub[:-(nEM-1)] = 100
    lc[:-(nEM-1)] = 1
    
    bounds_x = Bounds(lb, ub)
    bounds_cv = [(0.038,1)]
    linear_constraint = LinearConstraint(lc.tolist(), [100], [100])

    C = 10
    while C > 0.01:
    # Minimize objective function to obtain values of unknowns
        res = minimize(ouropt_1, unknowns, method = 'SLSQP', constraints=linear_constraint, bounds = bounds_x, args = (def_operations, em_peaks_mean.values, um_peaks_mean.values, tipo, cv, ignorar, nEM))
        unknowns = res.x
        C1 = cv

        # Update the value of the cv considering the values of the unknowns obtained in the previous step
        res = minimize(ouropt_2, cv, method = 'SLSQP', bounds = bounds_cv, args = (unknowns, def_operations, em_peaks_mean.values, um_peaks_mean.values, tipo, ignorar, nEM))
        cv = res.x 
        C = abs(C1-cv)
    # Print and save the results for each unknown mixture
    if nEM == 2:
        print(unknown_mixtures, unknowns[0], unknowns[1], cv)
    elif nEM ==3:
        print(unknown_mixtures, unknowns[0], unknowns[1], unknowns[2], cv)

    for j in range(nEM):
        X_todos[j,i] = unknowns[j]

########### Load the real results, compute estimates+-5% and plot  ################
real_results = pd.read_csv(folder+"/p_"+file+".csv", delimiter=',').values[nEM:,1:]
conf_int_sup = X_todos/100 + 5
conf_int_inf = X_todos/100 - 5
# plot_results(X_todos/100,conf_int_inf,conf_int_sup,real_results)
print(np.mean(abs(X_todos-np.transpose(real_results))))
print(np.transpose(X_todos))