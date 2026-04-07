'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = []
        num_particles = len(X_bar)

        normalized_weights = X_bar[:, -1] / sum(X_bar[:, -1])
        sample_counts = np.random.multinomial(num_particles, normalized_weights)

        for particle_idx, count in enumerate(sample_counts):
            X_bar_resampled += [X_bar[particle_idx]] * count

        X_bar_resampled = np.array(X_bar_resampled)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = np.zeros_like(X_bar)
        num_particles = len(X_bar)
        random_offset = np.random.uniform(0, 1/num_particles)
        source_idx, target_idx = 0, 0

        normalized_weights = X_bar[:, -1] / sum(X_bar[:, -1])
        cumulative_weight = normalized_weights[0]

        for sample_idx in range(num_particles):
            threshold = random_offset + sample_idx * (1/num_particles)
            while threshold > cumulative_weight:
                source_idx += 1
                cumulative_weight += normalized_weights[source_idx]
            X_bar_resampled[target_idx] = X_bar[source_idx]
            target_idx += 1

        return X_bar_resampled
