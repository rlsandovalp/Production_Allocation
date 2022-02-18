import pandas as pd
import numpy as np
import math
from scipy.stats import norm

mixtures = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

folder = '../Results/NGS_J19_DRK/Our/'

distances = []
for m, mixture in enumerate(mixtures):
    distances.append([])
    deconvolutions = pd.read_csv(folder+mixture+'.txt', sep = ' ').values
    true = pd.read_csv(folder+'../real_results.txt', header = None, sep = ',').set_index(0).values

    if m == 6:
        for dec in deconvolutions:
            distances[m].append(np.mean(np.abs(true[m][:2] - dec[:2])/true[m][:2])*100)
    elif m ==9:
        for dec in deconvolutions:
            distances[m].append(np.mean(np.abs(true[m][1:] - dec[1:3])/true[m][1:])*100)
    else:
        for dec in deconvolutions:
            distances[m].append(np.mean(np.abs(true[m] - dec[:3])/true[m])*100)

    print('- '*20)
    print('M'+str(m+1))
    print(np.nanmean(distances[m]))
    print(np.nanvar(distances[m]))
    print(math.sqrt(np.nanvar(distances[m]))*100/np.nanmean(distances[m]))
    np.savetxt(folder+'aitchison_distances_'+str(m)+'.txt', np.array((distances[m])))