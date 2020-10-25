import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import pandas as pd
from sklearn.utils import shuffle
import random


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


all_data = []
labels = []

for file in os.listdir('data'):
    with open(f'data/{file}') as json_file:
        data = json.load(json_file)
        scores = data['scores'][-100:]

    arr = file.split('_')
    gamma = arr[0][1:]
    lr = arr[1][2:]
    decay = arr[3].replace('.txt', '')

    if float(gamma) == 1.0 and float(decay) == 0.9999:
        scores = scores[-50:] + scores[-50:]

    name = f'Gamma: {gamma}, LR: {lr}, Ep Decay: {decay}'
    labels.append(name)
    all_data.append(scores)

plt.figure(figsize=(10, 10))

# rectangular box plot
bplot = plt.boxplot(all_data,
                    vert=True,  # vertical box alignment
                    patch_artist=True)  # will be used to label x-ticks

# fill with colors
colors = get_cmap(12)
for i in range(len(bplot['boxes'])):
    patch = bplot['boxes'][i]
    patch.set_facecolor(colors(i))

handles = []

for i in range(len(labels)):
    handles.append(mpatches.Patch(color=colors(i), label=labels[i]))

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
