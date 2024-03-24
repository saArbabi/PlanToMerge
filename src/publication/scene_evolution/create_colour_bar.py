# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# a%%

params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#s %%


fig, ax = plt.subplots(figsize=(7, 0.3))

cmap = mpl.cm.rainbow_r
bounds = np.linspace(1, 10, 10).astype('int')
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Create a colorbar
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax, orientation='horizontal')
cb.outline.set_visible(False)
# Adjust the tick positions to be at the center of the intervals
tick_locs = (bounds[:-1] + bounds[1:]) / 2
cb.set_ticks([])
# Add 's' to the tick labels
tick_labels = ['$' + str(bound) + '\,s$' for bound in bounds[:-1]]

# Add text to the middle of the colorbar
for i, label in enumerate(tick_labels):
    ax.text(tick_locs[i], 5, label, ha='center', va='center')

# Remove the boundaries
for location in ['top', 'right', 'bottom', 'left']:
    ax.spines[location].set_visible(False)
plt.savefig("time_color_bar.png", dpi=500, bbox_inches='tight')

# %%