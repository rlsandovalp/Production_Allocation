import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

mixtures = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
folder = '../Results/NGS_J19_DRK/ALS/'

fig, axes = plt.subplots(3, 4, dpi = 150)

size_label = 8
size_ticks = 9

for m, mix in enumerate(mixtures):
    x = math.ceil((m+1)/4)-1
    y = (m % 4)
    data = np.loadtxt(folder+'Ait_distances_'+str(m)+'.txt')
    axes[x,y].hist(data, bins = 30)
    axes[x,y].set_title(mix, fontweight="bold", fontsize = size_label + 4, x = 0.15, y = 0.8) 
    # axes[x,y].set_ylim([0,900])
    if (m == 0) or (m==4) or (m==8):
        axes[x,y].set_ylabel('Frequency')
    if (m == 8) or (m == 9) or (m == 7) or (m == 6):
        axes[x,y].set_xlabel('Aitchison Distance \n'r'$d_a(\mathbf{\hat{x}}_{PGM},\mathbf{x}^{*})$')
    mu = np.nanmean(data)
    sigma2 = np.nanvar(data)
    axes[x,y].text(0.7, 0.9, r'$\mu=$'+str(round(mu,2)), transform=axes[x,y].transAxes, fontsize = size_label)
    axes[x,y].text(0.7, 0.8, r'$\sigma^2=$'+str(round(sigma2,5)), transform=axes[x,y].transAxes, fontsize = size_label)
    axes[x,y].yaxis.set_minor_locator(AutoMinorLocator())
    axes[x,y].tick_params(axis = "y", which = 'both', direction = "in", labelsize = size_ticks)
    axes[x,y].xaxis.set_minor_locator(AutoMinorLocator())
    axes[x,y].tick_params(axis = "x", which = 'both', direction = "in", labelsize = size_ticks)

# plt.tight_layout()
plt.show()