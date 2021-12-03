import ternary
import pandas as pd
import numpy as np


folder = '../Results/NGS_J19_DRK/our/'
mixture = 'M1'
points = pd.read_csv(folder+mixture+'.txt', header = None, sep = ' ').values


figure, tax = ternary.figure(scale=100)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
tax.gridlines(color="black", multiple=20)
tax.gridlines(color="blue", multiple=20, linewidth=0.5)

# Set Axis labels and Title
fontsize = 12
offset = 0.14
# tax.right_corner_label("X", fontsize=fontsize)
# tax.top_corner_label("Y", fontsize=fontsize)
# tax.left_corner_label("Z", fontsize=fontsize)
tax.left_axis_label("EM3 [%]", fontsize=fontsize, offset=offset)
tax.right_axis_label("EM2 [%]", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("EM1 [%]", fontsize=fontsize, offset=offset)
tax.scatter(points, marker='.', label="Estimates")

# Set ticks
tax.ticks(axis='lbr', multiple=20, linewidth=1, offset=0.025)
tax.get_axes().axis('off')
# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()
tax.legend()

ternary.plt.show()