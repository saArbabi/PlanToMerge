import numpy as np
import matplotlib.pyplot as plt
""" plot setup
"""
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2
          }
plt.rcParams.update(params)
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rc('text', usetex=True)

f = open('./src/utility/population_data.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
aggs = np.array(content)
_ = plt.hist(aggs, 20,  lw=1, ec="white", fc="blue", alpha=0.5)
plt.ylabel('Population count')
plt.xlabel('Driver aggressiveness $\psi$')

# percentile_50 = np.quantile(aggs, 0.5)
# percentile_10 = np.quantile(aggs, 0.1)
#
# plt.axvline(percentile_50, color='red')
# plt.text(percentile_50+0.07, 100, '50th percentile', color='red', size=13)
# plt.axvline(percentile_10, color='red')
# plt.text(percentile_10+0.07, 100, '10th percentile', color='red', size=13)

# plt.savefig('driver_population.pdf', dpi=500, bbox_inches='tight')


# %%
f = open('./src/utility/tiv.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
tivs = np.array(content)
np.array(tivs).min()
np.array(tivs).max()
np.array(tivs).mean()
_ = plt.hist(tivs, 100,  lw=1,
                              ec="white", fc="blue", alpha=0.5)
plt.ylabel('Histogram count')
plt.xlabel('TIV (s)')
percentile_50 = np.quantile(tivs, 0.5)
percentile_10 = np.quantile(tivs, 0.1)
plt.axvline(percentile_50, 0, 0.9, color='red')

plt.text(percentile_50+0.2, 350, '50th percentile', color='red', size=13)
plt.axvline(percentile_10, color='red')
plt.text(percentile_10+0.07, 390, '10th percentile', color='red', size=13)
plt.ylim(0, 405)
plt.savefig('tiv_hist.pdf', dpi=500, bbox_inches='tight')
