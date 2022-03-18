import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import pandas as pd
from itertools import combinations

from PA_functions import plot_results, preprocess_our, ouropt_1, ouropt_2, convert_operations, preprocess_our
from scipy.optimize import LinearConstraint
from scipy.optimize import fsolve

folder = './../Data_Base/NGS_J19_DRK/'                                 
file = 'NGS_J19_DRK'
use_all_peaks = 0                           # Use all peaks? (1 Yes, 0 No)
peaks_to_analyze = 11                       # How many peaks use?
delta = 0.05
pp = 1                                      # Preprocess? (1 Yes, 0 No)
max_cv_samples = 10                         # Max intrasample CV for repetitions
max_cv_peaks = 20                           # Max cv allowed for the repetitions of a peak
cv = 0.5

################    HOW MANY END-MEMBERS, HOW MANY MIXTURES?    ########################
dataset = pd.read_csv(folder+'/'+file+".csv").set_index('Mix')
nEM = len([i for i in dataset.index.unique().values.tolist() if i.startswith('EM')])
nMix = len([i for i in dataset.index.unique().values.tolist() if i.startswith('M')])

if use_all_peaks == 0:
    peaks = dataset.iloc[:,0:peaks_to_analyze]
else:
    peaks = dataset.iloc[:,:]
if pp == 1: 
    peaks, ignorar = preprocess_our(peaks,pp, max_cv_peaks, max_cv_samples)
else:
    ignorar = []

################    WHERE ARE THE OPERATIONS?    ########################
operations = 'operations_all.txt'
if use_all_peaks == 0:
    operations = 'operations_11.txt'
operations_path=folder+'/'+operations

# Convert the operations from the text file to an array easier to manipulate
def_operations, tipo = convert_operations(operations_path)

# Define end members and mixtures list
end_members = ['EM'+str(x+1) for x in range(nEM)]
mixtures = ['M' + str(x+1) for x in range(nMix)]

for unknown_mixture in mixtures:

    reps_em1 = peaks.loc['EM1'].values.shape[0]
    reps_em2 = peaks.loc['EM2'].values.shape[0]
    reps_em3 = peaks.loc['EM3'].values.shape[0]
    reps_mix = peaks.loc[unknown_mixture].values.shape[0]
    npeaks = peaks.loc['EM1'].values.shape[1]


    # Create an array to storage the results
    combinations_results = np.zeros((reps_em1*reps_em2*reps_em3*reps_mix,nEM+1))

    # Define the unknown mixtures, create end members array, create unknown mixtures array

    hola = 0
    em_peaks = np.zeros((nEM,npeaks))
    em_peaks[0,:] = np.mean(peaks.loc['EM1'].values, axis = 0)
    em_peaks[1,:] = np.mean(peaks.loc['EM2'].values, axis = 0)
    em_peaks[2,:] = np.mean(peaks.loc['EM3'].values, axis = 0)
    um_peaks = np.mean(peaks.loc[unknown_mixture].values, axis = 0)

    # Define vector of unknowns and initialize its values, give an initial value to CV, and define bounds for the optimization
    unknowns = np.ones(5)  # [X1, X2, X3, ..., MR1, MR2] 
    lb = np.ones(5)*(1-delta)
    ub = np.ones(5)*(1+delta)
    lc = np.zeros(5)
    unknowns[:-2] = 33
    lb[:-2] = 0
    ub[:-2] = 100
    lc[:-2] = 1

    bounds_x = Bounds(lb, ub)
    bounds_cv = [(0.038,1)]
    linear_constraint = LinearConstraint(lc.tolist(), [100], [100])

    C = 10
    while C > 0.01:
    # Minimize objective function to obtain values of unknowns
        res = minimize(ouropt_1, unknowns, method = 'SLSQP', constraints=linear_constraint, bounds = bounds_x, args = (def_operations, em_peaks, um_peaks, tipo, cv, ignorar, nEM))
        unknowns = res.x
        C1 = cv
        # Update the value of the cv considering the values of the unknowns obtained in the previous step
        res = minimize(ouropt_2, cv, method = 'SLSQP', bounds = bounds_cv, args = (unknowns, def_operations, em_peaks, um_peaks, tipo, ignorar, nEM))
        cv = res.x
        C = abs(C1-cv)

    # Print and save the results for each unknown mixture
    combinations_results[hola,:] = (unknowns[0], unknowns[1], unknowns[2], cv)
    hola += 1
    print(unknown_mixture, unknowns[0], unknowns[1], unknowns[2])