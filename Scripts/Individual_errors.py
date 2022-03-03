import pandas as pd
import numpy as np
import math
from scipy.stats import norm

mixtures = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

folder = '../Results/NGS_J19_DRK/Our/'

def ait(x,y):
    sum = 0
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == 0:
                x[i] = 1E-6
            if x[j] == 0:
                x[j] = 1E-6
            if y[i] == 0:
                y[i] = 1E-6
            if y[j] == 0:
                y[j] = 1E-6
            sum = sum + (math.log(x[i]/x[j])-math.log(y[i]/y[j]))**2
    dis = math.sqrt(sum/(2*len(x)))
    return(dis)


distances = []
for m, mixture in enumerate(mixtures):
    distances.append([])
    deconvolutions = pd.read_csv(folder+mixture+'.txt', header = None, sep = ' ').values*100     # MC: sep = ' ', ALS: sep = '\t', Our: Sep = ' '
    true = pd.read_csv(folder+'../real_results.txt', header = None, sep = ',').set_index(0).values     

    # if m == 6:
    #     for dec in deconvolutions:
    #         distances[m].append(ait(true[m][:2], dec[:2]))
    # elif m ==9:
    #     for dec in deconvolutions:
    #         distances[m].append(ait(true[m][1:], dec[1:3]))
    for dec in deconvolutions:
        distances[m].append(ait(true[m], dec[:3]))

    # if m == 6:
    #     for dec in deconvolutions:
    #         distances[m].append(np.mean(np.abs(true[m][:2] - dec[:2])/true[m][:2])*100)
    # elif m ==9:
    #     for dec in deconvolutions:
    #         distances[m].append(np.mean(np.abs(true[m][1:] - dec[1:3])/true[m][1:])*100)
    # else:
    #     for dec in deconvolutions:
    #         distances[m].append(np.mean(np.abs(true[m] - dec[:3])/true[m])*100)

    print('- '*20)
    print('M'+str(m+1))
    print(np.nanmean(distances[m]))
    print(np.nanvar(distances[m]))
    print(math.sqrt(np.nanvar(distances[m]))*100/np.nanmean(distances[m]))
    np.savetxt(folder+'Ait_distances_'+str(m)+'.txt', np.array((distances[m])))