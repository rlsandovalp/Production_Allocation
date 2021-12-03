import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

real_results = pd.read_csv('../Data_Base/NGS_J19_DRK/p_NGS_J19_DRK.csv').values[3:,1:]

nEM = 3
nMix = 10
folder = '../Results/NGS_J19_DRK/McCaffrey/'
Mixtures = ['M'+str(mixture + 1) for mixture in range(nMix)]

results_averaged = pd.read_csv(folder+'averaged.txt', header = None, sep = ',').values

results = []

for i,_ in enumerate(Mixtures):
    results.append([])
    results[i].append(pd.read_csv(folder+'M'+str(i+1)+'.txt', header = None, sep = ' ').values[:,0]*100)
    results[i].append(pd.read_csv(folder+'M'+str(i+1)+'.txt', header = None, sep = ' ').values[:,1]*100)
    results[i].append(pd.read_csv(folder+'M'+str(i+1)+'.txt', header = None, sep = ' ').values[:,2]*100)

estimates = np.zeros((nMix, nEM))
estimates_variability = np.zeros((nMix, nEM))

for i in range(nMix):
    for j in range(nEM):
        estimates[i,j] = np.mean(results[i][j])
        estimates_variability[i,j] = np.std(results[i][j])

plt.figure(dpi = 150)
barWidth = 0.3
r = np.zeros((nEM, nMix))

plt.scatter(np.arange(nMix)+barWidth*0, real_results[:,0], color = 'darkgrey', label = 'True value EMs', marker = 'h', s = 50)
plt.scatter(np.arange(nMix)+barWidth*1, real_results[:,1], color = 'darkgrey', marker = 'h', s = 50)
plt.scatter(np.arange(nMix)+barWidth*2, real_results[:,2], color = 'darkgrey', marker = 'h', s = 50)

# plt.errorbar(np.arange(nMix)+barWidth*0, estimates[:,0], color = 'black', yerr=3*estimates_variability[:,0], label = 'Estimates', fmt='s', capsize=3)
# plt.errorbar(np.arange(nMix)+barWidth*1, estimates[:,1], color = 'black', yerr=3*estimates_variability[:,1], fmt='s', capsize=3)
# plt.errorbar(np.arange(nMix)+barWidth*2, estimates[:,2], color = 'black', yerr=3*estimates_variability[:,2], fmt='s', capsize=3)

for i in range(nMix):
    for j in range(nEM):
        plt.boxplot(results[i][j], positions = [barWidth*j+i], widths = 0.15, showmeans = True, sym = '.', meanline=True, meanprops=dict(linestyle='-', linewidth=1, color='black'),
            medianprops=dict(linestyle='-', linewidth=1, color='red'))
        plt.scatter(barWidth*j+i, results_averaged[i,j], color = 'red', marker = '.')
plt.scatter(barWidth*j+i, results_averaged[i,j], color = 'red', marker = '.', label = 'Estimates by using the mean of the repetitions')
plt.boxplot(results[i][j], positions = [barWidth*j+i], widths = 0.15, showmeans = True, sym = '.', meanline=True, 
            meanprops=dict(linestyle='-', linewidth=1, color='black', label = 'Mean of the estimates'),
            medianprops=dict(linestyle='-', linewidth=1, color='red', label = 'Median of the estimates'))
plt.xticks([r + barWidth for r in range(nMix)], Mixtures)
plt.ylabel('Mass Fractions [%]')
plt.legend()
plt.show()