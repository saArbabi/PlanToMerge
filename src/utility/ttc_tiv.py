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
np.quantile(ttcs, 0.1)
76/18
76/19
_ = plt.hist(ttcs, 50)
np.quantile(ttcs, 0.1)
np.quantile(ttcs, 0.9)

# %%
f = open('./src/utility/tiv.txt', 'r')
content = f.read().split()
content = [float(item) for item in content]
tivs = np.array(content)
np.array(tivs).min()
np.array(tivs).max()
np.array(tivs).mean()
_ = plt.hist(tivs, 50)
np.quantile(tivs, 0.1)
