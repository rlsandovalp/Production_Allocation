import numpy as np
import pandas as pd

mixtures = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
folder = '../Results/NGS_J19_DRK/ALS/'

resultados = []

for m in range(10): resultados.append([])

for i in range(100):
    data_file = pd.read_csv(folder+'Best_100/xo'+str(i)+'.csv', header = None, sep = ',').values*100
    for m, _ in enumerate(mixtures):
        resultados[m].append(data_file[m,:])


for m, mixture in enumerate(mixtures):
    np.savetxt(folder+mixture+'.txt', resultados[m])
    print(np.mean(resultados[m], axis = 0))