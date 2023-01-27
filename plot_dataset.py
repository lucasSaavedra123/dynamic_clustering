import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('smlm_dataset.csv')

print(f"Average: {len(dataset)/7000}")

dataset = dataset[ dataset['y'] < 10 ]
dataset = dataset[ dataset['x'] < 10 ]
dataset = dataset[ 0 < dataset['x'] ]
dataset = dataset[ 0 < dataset['y']]

dataset['clusterized'] = dataset['clusterized'].map({0: 'black', 1: 'red'})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset['x'], dataset['t'], dataset['y'], c=dataset['clusterized'], s=1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7000 * 10e-3)
ax.set_zlim(0,10)

ax.set_xlabel('x [um]')
ax.set_ylabel('t [s]')
ax.set_zlabel('y [um]')

plt.show()