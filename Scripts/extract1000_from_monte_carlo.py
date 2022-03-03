import numpy as np

case = 'NGS_J19_DRK_11'
algorithm = 'McCaffrey'
folder = '../Results/'+case+'/'+algorithm+'/'
mixtures = ['M'+str(i) for i in range(3,4)]

for mixture in mixtures:
    data = np.loadtxt(folder+mixture+'.txt')[:1000,:]
    data = np.where(data == 0, 1E-6, data)
    np.savetxt(folder+mixture+'_1000.csv', data, delimiter=',')