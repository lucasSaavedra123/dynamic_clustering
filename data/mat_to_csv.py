from scipy.io import loadmat
import pandas as pd

from Trajectory import Trajectory

X_POSITION_COLUMN_NAME='x'
Y_POSITION_COLUMN_NAME='y'
TIME_COLUMN_NAME='t'
FRAME_COLUMN_NAME='frame'
CLUSTER_ID_COLUMN_NAME='cluster_id'
CLUSTERIZED_COLUMN_NAME='clusterized'
PARTICLE_ID_COLUMN_NAME='particle_id'

mat_data = loadmat('./all_tracks_thunder_localizer.mat')
# Orden en la struct [BTX|mAb] [CDx|Control|CDx-Chol]
dataset = []
# Add each label and condition to the dataset
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][0][0]})
dataset.append({'label': 'BTX',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][0][1]})
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][0][2]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][1][0]})
dataset.append({'label': 'mAb',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][1][1]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][1][2]})

FRAME_RATE = 10e-3

for data in dataset:
    print("Building dataset for:", data['label'], data['exp_cond'])
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])

    smlm_dataset_rows = []

    for trajectory in trajectories:
        for length_index in range(trajectory.length):
            smlm_dataset_rows.append({
                PARTICLE_ID_COLUMN_NAME: -1
                X_POSITION_COLUMN_NAME: trajectory.get_noisy_x()[length_index],
                Y_POSITION_COLUMN_NAME: trajectory.get_noisy_y()[length_index],
                TIME_COLUMN_NAME: trajectory.get_time()[length_index],
                FRAME_COLUMN_NAME: int(trajectory.get_time()[length_index] / FRAME_RATE),
            })

    pd.DataFrame(smlm_dataset_rows).to_csv(f"{data['exp_cond']}_{data['label']}.csv")
