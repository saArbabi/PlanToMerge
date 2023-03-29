import numpy as np
import matplotlib.pyplot as plt
""" plot setup
"""
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 14,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
# %%

fig = plt.figure(figsize=(6 , 3))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=1)

ax_ttc = fig.add_subplot(121)
ax_tiv = fig.add_subplot(122)

f = open('./src/utility/ttc.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
ttcs = np.array(content)
ttcs = ttcs[ttcs > 0]
ttcs = ttcs[ttcs < 10]

_ = ax_ttc.hist(ttcs, 100, fc="darkblue")
ax_ttc.set_ylabel('Histogram count', labelpad=-2)
ax_ttc.set_xlabel('TTC (s)')
percentile_50 = np.quantile(ttcs, 0.5)
percentile_10 = np.quantile(ttcs, 0.1)

ax_ttc.axvline(percentile_50, color='red')
ax_ttc.text(percentile_50, 105, '50th', color='red', size=14, horizontalalignment='center')
ax_ttc.axvline(percentile_10, color='red')
ax_ttc.text(percentile_10, 105, '10th', color='red', size=14, horizontalalignment='center')

# plt.savefig('ttc_hist.pdf', dpi=500, bbox_inches='tight')



#s %%

f = open('./src/utility/tiv.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
tivs = np.array(content)


_ = plt.hist(tivs, 100, fc="darkblue")
ax_tiv.set_xlabel('TIV (s)')
ax_tiv.set_ylim(0, 410)
percentile_50 = np.quantile(tivs, 0.5)
percentile_10 = np.quantile(tivs, 0.1)
ax_tiv.axvline(percentile_50, color='red')
ax_tiv.text(percentile_50, 415, '50th', color='red', size=14, horizontalalignment='left')
ax_tiv.axvline(percentile_10, color='red')
ax_tiv.text(percentile_10, 415, '10th', color='red', size=14, horizontalalignment='center')
plt.savefig('ttc_tiv_hist.pdf', dpi=500, bbox_inches='tight')
# %%
 np.quantile(ttcs, 0.5)
 np.quantile(tivs, 0.5)
