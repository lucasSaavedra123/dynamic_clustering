from mongoengine import Document, FloatField, ListField, DictField, BooleanField
import matplotlib.pyplot as plt

import numpy as np

#Example about how to read trajectories from .mat
"""
from scipy.io import loadmat
from Trajectory import Trajectory
mat_data = loadmat('data/all_tracks_thunder_localizer.mat')
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
for data in dataset:
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])
    for trajectory in trajectories:
        if not trajectory.is_immobile(1.8):
            trajectory.save()
"""

class Trajectory(Document):
    x = ListField(required=True)
    y = ListField(required=False)
    z = ListField(required=False)

    t = ListField(required=True)

    noise_x = ListField(required=False)
    noise_y = ListField(required=False)
    noise_z = ListField(required=False)

    noisy = BooleanField(required=True)

    info = DictField(required=False)

    @classmethod
    def from_mat_dataset(cls, dataset, label='no label', experimental_condition='no experimental condition', scale_factor=1000): # With 1000 we convert trajectories steps to nm
        trajectories = []
        number_of_tracks = len(dataset)
        for i in range(number_of_tracks):
            raw_trajectory = dataset[i][0]

            trajectory_time = raw_trajectory[:, 0]

            trayectory_x = raw_trajectory[:, 1] * scale_factor
            trayectory_y = raw_trajectory[:, 2] * scale_factor

            trajectories.append(Trajectory(trayectory_x, trayectory_y, t=trajectory_time, info={
                                "label": label, "experimental_condition": experimental_condition}, noisy=True))

        return trajectories

    def __init__(self, x, y=None, z=None, model_category=None, noise_x=None, noise_y=None, noise_z=None, noisy=False, t=None, exponent=None, exponent_type='anomalous', info={}):

        if exponent_type == "anomalous":
            self.anomalous_exponent = exponent
        elif exponent_type == "hurst":
            self.anomalous_exponent = exponent * 2
        elif exponent_type is None:
            self.anomalous_exponent = None
        else:
            raise Exception(
                f"{exponent_type} exponent type is not available. Use 'anomalous' or 'hurst'.")

        self.model_category = model_category
        
        super().__init__(
            x=x,
            y=y,
            z=z,
            t=t,
            noise_x=noise_x,
            noise_y=noise_y,
            noise_z=noise_z,
            noisy=noisy,
            info=info
        )

    def get_anomalous_exponent(self):
        if self.anomalous_exponent is None:
            return "Not available"
        else:
            return self.anomalous_exponent

    def get_model_category(self):
        if self.model_category is None:
            return "Not available"
        else:
            return self.model_category

    @property
    def length(self):
        return len(self.x)

    @property
    def available_dimensions(self):
        if len(self.y) == 0:
            return 1
        elif len(self.z) == 0:
            return 2
        else:
            return 3

    def get_x(self):
        if self.x is None:
            raise Exception("x was not given")
        return np.copy(np.reshape(self.x, (len(self.x))))

    def get_y(self):
        if self.y is None:
            raise Exception("y was not given")
        return np.copy(np.reshape(self.y, (len(self.y))))

    def get_z(self):
        if self.z is None:
            raise Exception("y was not given")
        return np.copy(np.reshape(self.z, (len(self.z))))

    def get_time(self):
        if self.t is None:
            raise Exception("Time was not given")
        return np.copy(np.copy(np.reshape(self.t, (len(self.t)))))

    def get_noise_x(self):   
        return np.copy(self.noise_x)

    def get_noise_y(self):
        return np.copy(self.noise_y)

    def get_noise_z(self):
        return np.copy(self.noise_z)

    def get_noisy_x(self):   
        if self.noisy:
            return self.get_x()
        
        if self.noise_x is None:
            raise Exception('no x noise was provided')
        else:
            return self.get_x() + np.array(self.noise_x)

    def get_noisy_y(self):
        if self.noisy:
            return self.get_y()
        
        if self.noise_y is None:
            raise Exception('no y noise was provided')
        else:
            return self.get_y() + np.array(self.noise_y)

    def get_noisy_z(self):
        if self.noisy:
            return self.get_z()
        
        if self.noise_z is None:
            raise Exception('no z noise was provided')
        else:
            return self.get_z() + np.array(self.noise_z)

    def displacements_on_x(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_x())
        else:
            return np.diff(self.get_x())

    def displacements_on_y(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_y())
        else:
            return np.diff(self.get_y())

    def displacements_on_z(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_z())
        else:
            return np.diff(self.get_z())

    def hurst_exponent(self):
        return self.anomalous_exponent / 2

    def plot(self):
        plt.title((self))
        if self.available_dimensions == 1:
            plt.plot(self.get_x(), marker="X")
            plt.plot(self.get_noisy_x(), marker="X")
        elif self.available_dimensions == 2:
            plt.plot(self.get_x(), self.get_y(), marker="X")
            plt.plot(self.get_noisy_x(), self.get_noisy_y(), marker="X")
        elif self.available_dimensions == 3:
            raise Exception(
                "By now, It is not possible to plot 3D trajectories")

        plt.show()

    def __str__(self):
        if self.model_category is None:
            return f"Trajectory Length: {self.length}"
        elif str(self.model_category) == 'od':
            return f"Model: {self.model_category}, Trajectory Length: {self.length}"
        else:
            anomalous_exponent_string = "%.2f" % self.anomalous_exponent
            return f"Model: {self.model_category}, Anomalous Exponent: {anomalous_exponent_string}, Trajectory Length: {self.length}"

    def is_immobile(self, threshold):
        r = 0
        delta_r = []

        # Extract coordinate values from track
        data = np.zeros(shape=(2, self.length))
        data[0, :] = self.get_noisy_x()
        data[1, :] = self.get_noisy_y()

        for j in range(self.length):
            r = r + np.linalg.norm([data[0, j] - np.mean(data[0, :]), data[1, j] - np.mean(data[1, :])]) ** 2

        for j in range(self.length - 1):
            delta_r.append(np.linalg.norm([data[0, j + 1] - data[0, j], data[1, j + 1] - data[1, j]]))

        rad_gir = np.sqrt((1 / self.length) * r)
        mean_delta_r = np.mean(delta_r)
        criteria = (rad_gir / mean_delta_r) * np.sqrt(np.pi/2)
        
        return criteria <= threshold
