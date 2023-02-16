import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('smlm_dataset_4.csv')

print(f"Average: {len(dataset)/max(dataset['frame'])}")

dataset['clusterized'] = dataset['clusterized'].map({0: 'black', 1: 'red'})

projection = '2d'

if projection == '3d':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataset['x'], dataset['t'], dataset['y'], c=dataset['clusterized'], s=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100 * 10e-3)
    ax.set_zlim(0,10)

    ax.set_xlabel('x [um]')
    ax.set_ylabel('t [s]')
    ax.set_zlabel('y [um]')

if projection == '2d':
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(dataset['x'], dataset['y'], c=dataset['clusterized'], s=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')

plt.show()
