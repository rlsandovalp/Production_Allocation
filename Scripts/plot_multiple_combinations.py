import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


real_results = pd.read_csv('../Data_Base/NGS_J19_DRK/p_NGS_J19_DRK.csv').values[3:,1:]


nEM = 3
nMix = 10
folder = '../Results/'
Mixtures = ['M'+str(mixture + 1) for mixture in range(nMix)]

results = []

for i,_ in enumerate(Mixtures):
    results.append([])
    results[i].append(pd.read_csv(folder+'M'+str(i+1)+'.txt', header = None, sep = ' ').values[:,0])
    results[i].append(pd.read_csv(folder+'M'+str(i+1)+'.txt', header = None, sep = ' ').values[:,1])
    results[i].append(pd.read_csv(folder+'M'+str(i+1)+'.txt', header = None, sep = ' ').values[:,2])

estimates = np.zeros((nMix, nEM))
estimates_variability = np.zeros((nMix, nEM))

for i in range(nMix):
    for j in range(nEM):
        estimates[i,j] = np.mean(results[i][j])
        estimates_variability[i,j] = np.std(results[i][j])

plt.figure(figsize=(nMix + 5, 6), dpi = 100)
barWidth = 0.3
r = np.zeros((nEM, nMix))

plt.scatter(np.arange(nMix)+barWidth*0, real_results[:,0], color = 'red', label = 'True value EM1', marker = '_', s = 300)
plt.scatter(np.arange(nMix)+barWidth*1, real_results[:,1], color = 'green', label = 'True value EM2', marker = '_', s = 300)
plt.scatter(np.arange(nMix)+barWidth*2, real_results[:,2], color = 'blue', label = 'True value EM3', marker = '_', s = 300)

plt.errorbar(np.arange(nMix)+barWidth*0, estimates[:,0], color = 'black', yerr=3*estimates_variability[:,0], label = 'Estimates', fmt='s', capsize=3)
plt.errorbar(np.arange(nMix)+barWidth*1, estimates[:,1], color = 'black', yerr=3*estimates_variability[:,1], fmt='s', capsize=3)
plt.errorbar(np.arange(nMix)+barWidth*2, estimates[:,2], color = 'black', yerr=3*estimates_variability[:,2], fmt='s', capsize=3)

for i in range(nMix):
    for j in range(nEM):
        plt.scatter(np.ones(np.size(results[i][j]))*(barWidth*j+i), results[i][j], color = 'gray', s = 8)


plt.xticks([r + barWidth for r in range(nMix)], Mixtures)
plt.ylabel('Mass Fractions [%]')
plt.legend()
plt.show()