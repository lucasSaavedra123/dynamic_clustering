import pandas as pd
import matplotlib.pyplot as plt






dataset = pd.read_csv('smlm_dataset.csv')

dataset = dataset[ dataset['x'] < 10 ]
dataset = dataset[ dataset['y'] < 10 ]
dataset = dataset[ 0 < dataset['x'] ]
dataset = dataset[ 0 < dataset['y']]

dataset[ dataset['clusterized'] == "True"] = 0
dataset[ dataset['clusterized'] == "False" ] = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset['x'], dataset['t'], dataset['y'], c=dataset['clusterized'], s=1)
plt.show()