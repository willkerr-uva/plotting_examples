import numpy as np
import matplotlib.pyplot as plt

samples = 10000
#cavas 1
x = np.random.normal(loc=100, scale=6, size=samples)

ave = np.mean(x)
std = np.std(x)
entries = len(x)

counts, bins = np.histogram(x, bins=100, range=(50, 150))
bin_centers = 0.5 * (bins[1:] + bins[:-1])
errors = np.sqrt(counts)

plt.figure()
plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', label=f'Entries={len(x)}\nMean={np.mean(x):.2f}\nStd Dv={np.std(x):.2f}', markersize=2)
plt.title("Random Gaussian")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.legend(fontsize=8)
plt.savefig("canvas1_py.png")

#canvas 2
plt.figure()

plt.subplot(2, 2, 1)
plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', label=f'Entries={len(x)}\nMean={np.mean(x):.2f}\nStd Dv={np.std(x):.2f}', markersize=2)
plt.title("Random Gaussian")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.legend(fontsize=5)

plt.subplot(2, 2, 2)
x2 = np.concatenate([x, np.random.uniform(50, 150, size=samples // 3)])
ave = np.mean(x2)
std = np.std(x2)
entries = len(x2)
counts, bins = np.histogram(x2, bins=100, range=(50, 150))
bin_centers = 0.5 * (bins[1:] + bins[:-1])
errors = np.sqrt(counts)
plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', label=f'Entries={len(x2)}\nMean={np.mean(x2):.2f}\nStd Dev={np.std(x2):.2f}', markersize=2)
plt.title("Gauss + Offset")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.legend(fontsize=5)

plt.subplot(2, 2, 3)
import numpy as np

samples = 10000
num_baseline = samples * 30
x_baseline = []

#use rejection sampling to generate 1/x^2 pdf
while len(x_baseline) < num_baseline:
    trial = np.random.uniform(1, 10, size=1000)
    y = np.random.uniform(0, 1, size=1000)
    keep = trial[y < 1 / trial**2]
    x_baseline.extend(keep)
x_baseline = np.array(x_baseline[:num_baseline])
x_baseline = x_baseline * 10 + 40

x3 = np.concatenate([x, x_baseline])
ave = np.mean(x3)
std = np.std(x3)
entries = len(x3)
counts, bins = np.histogram(x3, bins=100, range=(50, 150))
bin_centers = 0.5 * (bins[1:] + bins[:-1])
errors = np.sqrt(counts)
plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', label=f'Entries={len(x3)}\nMean={np.mean(x3):.2f}\nStd Dev={np.std(x3):.2f}', markersize=2)
plt.yscale("log")
plt.title("Gauss + Offset2")
plt.xlabel("x")
plt.ylabel("Frequency (log scale)")
plt.legend(fontsize=5)

plt.subplot(2, 2, 4)
x4 = np.concatenate([x, np.random.normal(loc=100, scale=20, size=samples // 2)])
ave = np.mean(x4)
std = np.std(x4)
entries = len(x4)
counts, bins = np.histogram(x4, bins=100, range=(50, 150))
bin_centers = 0.5 * (bins[1:] + bins[:-1])
errors = np.sqrt(counts)
plt.errorbar(bin_centers, counts, yerr=errors, fmt='o', label=f'Entries={len(x4)}\nMean={np.mean(x4):.2f}\nStd Dev={np.std(x4):.2f}', markersize=2)
plt.title("Double Gaussian")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.legend(fontsize=5)

plt.tight_layout()
plt.savefig("canvas2_py.pdf")