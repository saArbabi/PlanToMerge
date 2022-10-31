import numpy as np
import matplotlib.pyplot as plt
""" plot setup
"""
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

f = open('./src/utility/ttc.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
ttcs = np.array(content)
ttcs = ttcs[ttcs > 0]
ttcs = ttcs[ttcs < 10]
_ = plt.hist(ttcs, 100,  lw=1,
                              ec="white", fc="blue", alpha=0.5)
plt.ylabel('Histogram count')
plt.xlabel('TTC (s)')
percentile_50 = np.quantile(ttcs, 0.5)
percentile_10 = np.quantile(ttcs, 0.1)

plt.axvline(percentile_50, color='red')
plt.text(percentile_50+0.07, 100, '50th percentile', color='red', size=11)
plt.axvline(percentile_10, color='red')
plt.text(percentile_10+0.07, 100, '10th percentile', color='red', size=11)

plt.savefig('ttc_hist.pdf', dpi=500, bbox_inches='tight')


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

plt.text(percentile_50+0.2, 350, '50th percentile', color='red', size=11)
plt.axvline(percentile_10, color='red')
plt.text(percentile_10+0.2, 390, '10th percentile', color='red', size=11)
plt.ylim(0, 405)
plt.savefig('tiv_hist.pdf', dpi=500, bbox_inches='tight')
