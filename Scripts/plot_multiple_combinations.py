import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

real_results = pd.read_csv('../Data_Base/NGS_J19_DRK/p_NGS_J19_DRK.csv').values[3:,1:]

nEM = 3
nMix = 10
folder = '../Results/NGS_J19_DRK/PGM/'
Mixtures = ['M'+str(mixture + 1) for mixture in range(nMix)]

results_averaged = pd.read_csv(folder+'averaged.txt', header = None, sep = ',').values[:,1:]

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

plt.figure(dpi = 150)
barWidth = 0.3
r = np.zeros((nEM, nMix))

plt.scatter(np.arange(nMix)+barWidth*0, real_results[:,0], color = 'red', label = 'True value', marker = 'h', s = 50)
plt.scatter(np.arange(nMix)+barWidth*1, real_results[:,1], color = 'red', marker = 'h', s = 50)
plt.scatter(np.arange(nMix)+barWidth*2, real_results[:,2], color = 'red', marker = 'h', s = 50)
plt.ylim(0,100)

for i in range(nMix):
    for j in range(nEM):
        plt.boxplot(results[i][j], positions = [barWidth*j+i], widths = 0.15, showmeans = True, sym = '.', meanline=True, meanprops=dict(linestyle='-', linewidth=1, color='blue'),
            medianprops=dict(linestyle='-', linewidth=1, color='purple'))
        plt.scatter(barWidth*j+i, results_averaged[i,j], color = 'black', marker = 'x')
    plt.vlines(barWidth*j+i+0.2,-10,110, color = 'gray')

plt.boxplot(results[i][j], positions = [barWidth*j+i], widths = 0.15, showmeans = True, sym = '.', meanline=True, 
            meanprops=dict(linestyle='-', linewidth=1, color='blue', label = 'Mean deconvolutions'),
            medianprops=dict(linestyle='-', linewidth=1, color='purple'))
plt.scatter(barWidth*j+i, results_averaged[i,j], color = 'black', marker = 'x', label = 'Averaged')

plt.xticks([r + barWidth for r in range(nMix)], Mixtures)
plt.ylabel('Mass Fractions [%]')
plt.legend()
plt.show()