from scipy.io import loadmat
import pandas as pd

from data.Trajectory import Trajectory
from LocalizationClassifier import LocalizationClassifier
from ClusterEdgeRemover import ClusterEdgeRemover

X_POSITION_COLUMN_NAME='x'
Y_POSITION_COLUMN_NAME='y'
TIME_COLUMN_NAME='t'
FRAME_COLUMN_NAME='frame'
CLUSTER_ID_COLUMN_NAME='cluster_id'
CLUSTERIZED_COLUMN_NAME='clusterized'
PARTICLE_ID_COLUMN_NAME='particle_id'

mat_data = loadmat('./data/all_tracks_thunder_localizer.mat')
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

localization_classifier = LocalizationClassifier(10,10)
localization_classifier.hyperparameters['partition_size'] = 1000
localization_classifier.load_model()

edge_classifier = ClusterEdgeRemover(10,10)
edge_classifier.hyperparameters['partition_size'] = 1000
edge_classifier.load_model()

for data in dataset:
    print("Building dataset for:", data['label'], data['exp_cond'])
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])

    smlm_dataset_rows = []

    for index, trajectory in enumerate(trajectories):
        #if not trajectory.is_immobile(1.8):
        if True:
            for length_index in range(trajectory.length):
                smlm_dataset_rows.append({
                    PARTICLE_ID_COLUMN_NAME: index,
                    X_POSITION_COLUMN_NAME: trajectory.get_noisy_x()[length_index] / 1000,
                    Y_POSITION_COLUMN_NAME: trajectory.get_noisy_y()[length_index] / 1000,
                    TIME_COLUMN_NAME: trajectory.get_time()[length_index],
                    FRAME_COLUMN_NAME: int(trajectory.get_time()[length_index] / FRAME_RATE),
                    CLUSTERIZED_COLUMN_NAME: 0,
                    CLUSTER_ID_COLUMN_NAME: 0,
                })

    smlm_dataset = pd.DataFrame(smlm_dataset_rows)
    smlm_dataset = smlm_dataset[smlm_dataset['frame'] < 1000]
    smlm_dataset.to_csv(f"{data['exp_cond']}_{data['label']}.csv", index=False)

    print("Se predice...")

    magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = localization_classifier.predict(magik_dataset)
    smlm_dataset = localization_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)
    
    #magik_dataset = edge_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    #magik_dataset = edge_classifier.predict(magik_dataset, detect_clusters=True, apply_threshold=True)
    #smlm_dataset = edge_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    smlm_dataset.to_csv(f"{data['exp_cond']}_{data['label']}_prediction.csv", index=False)
