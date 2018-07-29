import numpy as np
from matplotlib import pyplot as plt

freqs = []
words = []
with open("freq.txt","r") as f:
	for line in f:
		freq = int(line.split(" ")[0])
		word = line.split(" ")[1].strip()
		words.append(word)
		freqs.append(freq)

freqs, words = (list(t) for t in zip(*sorted(zip(freqs, words))))

fig = plt.figure()

width = 0.9
ind = np.arange(len(freqs))
plt.bar(ind, freqs, width=width)
plt.xticks(ind + width / 2, words)

locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.tick_params(axis='both', which='minor', labelsize=10)
plt.tick_params(axis='both', which='major', labelsize=8)

fig.tight_layout()
# plt.show()
fig.savefig("bar-chart.jpg")