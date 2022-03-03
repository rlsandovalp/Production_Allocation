import ternary
import pandas as pd
import matplotlib.pyplot as plt


figure, ax = plt.subplots(frameon = False, dpi = 150)
figure.gca().get_xaxis().set_visible(False)
figure.gca().get_yaxis().set_visible(False)

mixtures = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

folder = '../Results/NGS_J19_DRK/Our/'

for m, mixture in enumerate(mixtures):

    deconvolutions = pd.read_csv(folder+mixture+'.txt', sep = ' ').values
    true = pd.read_csv(folder+'../real_results.txt', header = None, sep = ',').set_index(0)
    averaged =  pd.read_csv(folder+'Average_before_deconvolution.txt', header = None, sep = ',').set_index(0)
    compositional_mean = pd.read_csv(folder+'compositional_mean.txt', header = None, sep = ',').set_index(0)

    k = m
 
    ax = figure.add_subplot(3,4,k+1)
    figure, tax = ternary.figure(ax = ax, scale=100)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=1.0)
    tax.gridlines(color="gray", multiple=25)
    # tax.gridlines(color="blue", multiple=20, linewidth=0.5)

    # Set Axis labels and Title
    fontsize = 8
    offset = 0.14
    # tax.right_corner_label("X", fontsize=fontsize)
    # tax.top_corner_label("Y", fontsize=fontsize)
    # tax.left_corner_label("Z", fontsize=fontsize)
    tax.set_title(mixture, fontsize=fontsize+3, loc='left', y=0.7, fontweight="bold")
    tax.left_axis_label("EM3 [%]", fontsize=fontsize, offset=offset)
    tax.right_axis_label("EM2 [%]", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("EM1 [%]", fontsize=fontsize, offset=offset)
    tax.scatter(deconvolutions, marker='.', label="Estimates, " r'$\mathbf{\hat{x}}_{PGM}$', color = 'gray')
    tax.scatter([true.loc[mixture].values], marker='o', label="True value, " r'$\mathbf{x}^{*}$', color = 'green')
    tax.scatter([averaged.loc[mixture].values], marker='o', label="Estimates, " r'$\bar{\bar{\mathbf{x}}}_{PGM}$', color = 'red')
    tax.scatter([compositional_mean.loc[mixture].values], marker='x', label="Compositional mean, " r'$\bar{\mathbf{x}}_{PGM}$', color = 'black')


    # Set ticks
    tax.ticks(axis='lbr', multiple=25, fontsize = fontsize-1, linewidth=1, offset=0.025)
    tax.get_axes().axis('off')
    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.legend()

ternary.plt.show()