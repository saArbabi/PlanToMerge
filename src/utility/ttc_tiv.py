import numpy as np
import matplotlib.pyplot as plt

f = open('./src/utility/ttc.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
ttcs = np.array(content)
ttcs = ttcs[ttcs > 0]
ttcs = ttcs[ttcs < 10]
np.array(ttcs).min()
np.array(ttcs).mean()
_ = plt.hist(ttcs, 50)

# %%
f = open('./src/utility/tiv.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
tivs = np.array(content)
np.array(tivs).min()
np.array(tivs).max()
np.array(tivs).mean()
7.5/16
_ = plt.hist(tivs, 50)
