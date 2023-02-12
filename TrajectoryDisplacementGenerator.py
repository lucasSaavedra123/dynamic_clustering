from functools import lru_cache

import numpy as np

def fgn_autocovariance(hurst, n):
    """Autocovariance function for fGn."""
    ns_2h = np.arange(n + 1) ** (2 * hurst)
    return np.insert((ns_2h[:-2] - 2 * ns_2h[1:-1] + ns_2h[2:]) / 2, 0, 1)

autocovariance = lru_cache(1)(fgn_autocovariance)

class TrajectoryDisplacementGenerator:
    def __init__(self, anomalous_exponent, maximum_frame):
        self.next_step_flag = False
        self.maximum_frame = maximum_frame
        self.hurst = anomalous_exponent/2
        self.increment = 1 / self.maximum_frame
        self.scale = self.increment**self.hurst

    def __iter__(self):
        # If H = 0.5 then just generate a standard Brownian motion, otherwise
        # proceed with Hosking's method
        if self.hurst == 0.5:
            position_index = 0
            while position_index < self.maximum_frame:
                yield np.random.normal(0, 1) * self.scale

                if self.next_step_flag:
                    self.next_step_flag = False
                    position_index +=1

        else:
            # Initializations
            fgn = np.zeros(self.maximum_frame)
            phi = np.zeros(self.maximum_frame)
            psi = np.zeros(self.maximum_frame)
            cov = autocovariance(self.hurst, self.maximum_frame)

            # First increment from stationary distribution

            while not self.next_step_flag:
                gn = np.random.normal(0, 1, self.maximum_frame)
                yield gn[0] * self.scale

            self.next_step_flag = False

            fgn[0] = gn[0]
            v = 1
            old_v = 1
            phi[0] = 0
            position_index = 1

            # Generates fgn realization with n increments of size 1
            #for i in range(1, self.maximum_frame):
            while position_index < self.maximum_frame:
                phi[position_index - 1] = cov[position_index]
                for j in range(position_index - 1):
                    psi[j] = phi[j]
                    phi[position_index - 1] -= psi[j] * cov[position_index - j - 1]
                phi[position_index - 1] /= v
                for j in range(position_index - 1):
                    phi[j] = psi[j] - phi[position_index - 1] * psi[position_index - j - 2]
                old_v = v
                v *= 1 - phi[position_index - 1] * phi[position_index - 1]
                for j in range(position_index):
                    fgn[position_index] += phi[j] * fgn[position_index - j - 1]
                fgn[position_index] += np.sqrt(v) * gn[position_index]

                yield fgn[position_index] * self.scale

                if self.next_step_flag:
                    position_index += 1
                else:
                    v = old_v
                    gn[position_index] = np.random.normal(0, 1)
                    fgn[position_index] = 0

        # Scale to interval [0, T]
        #fgn *= self.scale

        #return fgn

    def next_step(self):
        self.next_step_flag = True
