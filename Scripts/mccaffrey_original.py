import numpy as np 
import pandas as pd 
from scipy.optimize import fmin_slsqp
from PA_functions import preprocess, plot_results, f_normalize
from scipy.stats import t

def residuals(x,A,b):
    return np.linalg.norm(A @ x -b)

def eq_cond(x, *args):
    return sum(x) - 1.0

######################   DEFINE VARIABLES   #########################
group = './../Data_Base/NGS_J19_DRK_11/'                                 
dataset = 'NGS_J19_DRK_11'
mixtures = (1,2,3,4,5,6,7,8,9,10)                        # Mixtures to be analyzed
max_cv_peaks = 20                           # Max intrasample CV for peaks
max_cv_samples = 10                         # Max intrasample CV for repetitions
pp = 1                                      # Preprocess? (1 Yes, 0 No)
confidence_level = 0.95                     # Confidence level of confidence intervals
use_all_peaks = 1                           # Use all peaks? (1 Yes, 0 No)
peaks_to_analyze = 11                       # How many peaks shall be used (if use_all_peaks == 1 this parameter is not read)
constraint_regression = 0                   # Constraint the regression so that sum of end members = 100%? (1 Yes, 0 No)

################    PREPROCESSING    ########################
end_members = pd.read_csv(group+'/p_'+dataset+'.csv').set_index('Mix').values.shape[1]
peaks = pd.read_csv(group+'/'+dataset+'.csv')
if use_all_peaks == 0: peaks = peaks.iloc[:,0:peaks_to_analyze+1]
peaks = preprocess(peaks,pp,max_cv_peaks,max_cv_samples)

######################   INITIALIZE VARIABLES AND READ DATA   #########################
peaks = peaks.set_index('Mix').mean(level=0)
real_results = (pd.read_csv(group+'/p_'+dataset+'.csv')).set_index('Mix').values[end_members:,:]
A, p_m = f_normalize(0,peaks.values,end_members)
X_todos = np.zeros((end_members, len(mixtures)))
conf_int_inf = np.zeros((end_members, len(mixtures)))
conf_int_sup = np.zeros((end_members, len(mixtures)))
x0 = np.ones(end_members)

######################   COMPUTE INTERMEDIATE VARIABLES   #########################
AtA_inv = np.linalg.inv(np.transpose(A) @ A)
p_f = t.ppf(confidence_level, np.shape(A)[0])

######################   LINEAR REGRESSION   #########################
for count, mixture in enumerate(mixtures):
    b = p_m[:,count]
    if constraint_regression == 1:
        if end_members == 3:
            X_todos[:,count] = fmin_slsqp(residuals, x0, eqcons = [eq_cond], bounds = [(0,1),(0,1),(0,1)], args = (A, b), iprint = -1)
        if end_members == 2:
            X_todos[:,count] = fmin_slsqp(residuals, x0, eqcons = [eq_cond], bounds = [(0,1),(0,1)], args = (A, b), iprint = -1)
    else:
        X_todos[:,count] = fmin_slsqp(residuals, x0, args = (A, b), iprint = -1)

X_todos1 = X_todos/np.sum(X_todos,axis=0)

######################   CONFIDENCE INTERVALS   #########################
for count, mixture in enumerate(mixtures):
    b = p_m[:,count]
    errors = A @ X_todos1[:,count] - b
    for em in range(end_members):
        conf_int_sup[em,count] = X_todos1[em,count] + 5
        conf_int_inf[em,count] = X_todos1[em,count] - 5

######################   RESET TO ZERO THE MIXTURES (ONLY FOR PLOTTING PURPOSES)   #########################
mixtures1 = []
for i in range(len(mixtures)): mixtures1.append(int(mixtures[i]-1))

######################   PLOT AND SAVE DATA   #########################
print(np.mean(abs(real_results - np.transpose(X_todos1*100))))
plot_results(X_todos1,conf_int_inf,conf_int_sup,real_results[mixtures1,:])
np.savetxt('../Results/Paper/McCaffrey/Estimates.csv',np.r_[np.transpose(X_todos)*100,np.transpose(conf_int_inf)*100,np.transpose(conf_int_sup)*100])

