import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PA_functions import preprocess
from matplotlib.ticker import AutoMinorLocator, LogLocator


data='./../Data_Base/NGS_J19_DRK/'                                 
mix = 'NGS_J19_DRK'
fluid = 'EM3'
max_cv_peaks = 30                           # Max intrasample CV for peaks
max_cv_samples = 10                         # Max intrasample CV for repetitions
pp = 0                                      # Preprocess? (1 Yes, 0 No)
end_members = 3


if pp == 1: 
    peaks = pd.read_csv(data+"/"+mix+".csv").set_index('Mix')
    peaks = preprocess(peaks,pp,max_cv_peaks,max_cv_samples)
else:
    peaks = pd.read_csv(data+"/"+mix+".csv")

dataset = peaks.set_index('Mix')
subset = dataset.loc[fluid].values

fig, axes = plt.subplots(dpi = 150)

ancho = 0.15
for i in range(-2,3):
    axes.bar(np.linspace(1+i*ancho,subset.shape[1]+i*ancho,subset.shape[1]), subset[i+2,:],
    label = 'Repetition '+str(i+3), linewidth=0.5, width=ancho)


size_ticks = 7
size_label = 9

axes.set_yscale('log')
axes.set_ylim(1E4,1E7)
axes.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
axes.tick_params(axis = "y", which = 'both', right = True, direction = "in", labelsize = size_ticks)
axes.xaxis.set_minor_locator(AutoMinorLocator())
axes.tick_params(axis = "x", which = 'both', top = True, direction = "in", labelsize = size_ticks)
axes.legend(loc='best', fontsize = size_label)
axes.set_ylabel('Peak number', fontsize = size_label)
axes.set_xlabel('Peak response (mV)', fontsize = size_label)

plt.savefig('../Repeatability'+fluid+'.svg', format = 'svg')
# plt.show()
