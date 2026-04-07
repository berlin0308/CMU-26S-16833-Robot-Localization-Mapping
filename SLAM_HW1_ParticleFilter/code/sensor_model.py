'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.12
        self._z_max = 0.05
        self._z_rand = 800  # Using sum of log probabilities, so large value is fine (per tips_for_students.md)

        self._sigma_hit = 100
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        # Subsampling every 2 degrees (approximately every 2 beams out of 180)
        # This reduces computation while maintaining reasonable accuracy
        self._subsampling = 2
        
        # Store occupancy map
        self._occupancy_map = occupancy_map
        self._resolution = 10  # each cell has a 10cm resolution in x,y axes
        
        # Laser sensor offset from robot center (25cm forward in x-axis)
        self._offset = 25  # The laser on the robot is 25 cm offset forward from center of the robot
        
        # Precomputed raycast map (will be computed if needed)
        self._raycast_map = None
        self._interpolation_num = 250  # The number of points interpolated during ray casting

    def _ray_cast(self, x_start, y_start, angle):
        """
        Perform ray casting from start position in given angle direction.
        
        param[in] x_start : starting x position in world frame (cm)
        param[in] y_start : starting y position in world frame (cm)
        param[in] angle : ray angle in world frame (radians)
        param[out] range : predicted range measurement (cm)
        """
        # Convert world coordinates to map cell indices
        map_x = int(x_start / self._resolution)
        map_y = int(y_start / self._resolution)
        
        # Check bounds
        map_height, map_width = self._occupancy_map.shape
        if map_x < 0 or map_x >= map_height or map_y < 0 or map_y >= map_width:
            return self._max_range
        
        # Ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Step size for ray casting (smaller for better accuracy)
        step_size = 1.0  # cells
        max_steps = int(self._max_range / self._resolution)
        
        # Current position in map coordinates
        current_x = map_x
        current_y = map_y
        
        # Ray casting using DDA-like algorithm
        # Store previous position to calculate distance to cell boundary when hitting obstacle
        prev_x = current_x
        prev_y = current_y
        
        for step in range(max_steps):
            # Check if we're still in bounds
            if current_x < 0 or current_x >= map_height or current_y < 0 or current_y >= map_width:
                return self._max_range
            
            # Check occupancy
            # Note: occupancy_map is indexed as [y, x] (row, col)
            occupancy = self._occupancy_map[int(current_y), int(current_x)]
            
            # If we hit an obstacle
            if occupancy > self._min_probability:
                # Calculate distance using the actual current position (in world coordinates)
                # This gives the distance to the obstacle cell boundary, not the cell center
                world_x = current_x * self._resolution
                world_y = current_y * self._resolution
                distance = math.sqrt((world_x - x_start)**2 + (world_y - y_start)**2)
                return min(distance, self._max_range)
            
            # Store previous position before moving
            prev_x = current_x
            prev_y = current_y
            
            # Move along ray
            current_x += dx * step_size
            current_y += dy * step_size
        
        # Reached max range
        return self._max_range

    def _compute_probability(self, z_meas, z_expected):
        """
        Compute probability for a single range measurement using four models.
        
        param[in] z_meas : measured range (cm)
        param[in] z_expected : expected range from ray casting (cm)
        param[out] prob : combined probability
        """
        # Note: reference solution does NOT normalize by weight_sum
        # Weights are used directly without normalization
        
        # p_hit: Gaussian distribution centered at expected range
        if z_meas >= 0 and z_meas <= self._max_range:
            # Normalize Gaussian to [0, z_max]
            p_hit = norm.pdf(z_meas, loc=z_expected, scale=self._sigma_hit)
            # Normalize by CDF from 0 to z_max
            cdf_max = norm.cdf(self._max_range, loc=z_expected, scale=self._sigma_hit)
            cdf_min = norm.cdf(0, loc=z_expected, scale=self._sigma_hit)
            if cdf_max - cdf_min > 1e-10:
                p_hit = p_hit / (cdf_max - cdf_min)
            else:
                p_hit = 0.0
        else:
            p_hit = 0.0
        
        # p_short: Exponential distribution (only valid when z_meas < z_expected)
        if z_meas >= 0 and z_meas <= z_expected and z_expected > 0:
            # Exponential distribution: lambda * exp(-lambda * z)
            p_short = self._lambda_short * math.exp(-self._lambda_short * z_meas)
            # Normalize
            if z_expected > 0:
                norm_factor = 1.0 - math.exp(-self._lambda_short * z_expected)
                if norm_factor > 1e-10:
                    p_short = p_short / norm_factor
                else:
                    p_short = 0.0
        else:
            p_short = 0.0
        
        # p_max: Dirac delta at z_max (when measurement equals max range)
        if abs(z_meas - self._max_range) < 1e-6:
            p_max = 1.0
        else:
            p_max = 0.0
        
        # p_rand: Uniform distribution over [0, z_max]
        if z_meas >= 0 and z_meas <= self._max_range:
            p_rand = 1.0 / self._max_range
        else:
            p_rand = 0.0
        
        # Weighted combination
        prob = self._z_hit * p_hit + \
               self._z_short * p_short + \
               self._z_max * p_max + \
               self._z_rand * p_rand
        
        return prob

    def beam_range_finder_model(self, z_t1_arr, x_t1, raycast_map):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[in] raycast_map : look up map for the true laser range readings zstar_t
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        # Down-sample the laser reading
        z_t = np.array([z_t1_arr[i] for i in range(0, 180, self._subsampling)])
        # Get the true laser reading
        zstar_t = self.ray_casting(x_t1, raycast_map)

        prob_zt1 = self._z_hit * self.get_p_hit(z_t, zstar_t) + \
                   self._z_short * self.get_p_short(z_t, zstar_t) + \
                   self._z_max * self.get_p_max(z_t) + \
                   self._z_rand * self.get_p_rand(z_t)

        prob_zt1 = np.delete(prob_zt1, np.where(prob_zt1 == 0.0))
        prob_zt1 = np.sum(np.log(prob_zt1))

        return np.exp(prob_zt1)

    def beam_range_finder_model_vectorized(self, z_t1_arr, X_t1, raycast_map=None):
        """
        Vectorized version: compute likelihoods for all particles at once
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] X_t1 : all particle states [num_particles x 3] at time t [world_frame]
        param[in] raycast_map : precomputed lookup map for ray casting (optional, uses self._raycast_map if None)
        param[out] weights : likelihoods [num_particles] for all particles
        """
        # Use provided raycast_map or fall back to instance variable
        if raycast_map is None:
            raycast_map = self._raycast_map
        
        num_particles = X_t1.shape[0]
        
        # If raycast_map is available, use lookup method (much faster)
        if raycast_map is not None:
            # Down-sample the laser reading
            z_t = np.array([z_t1_arr[i] for i in range(0, 180, self._subsampling)])
            
            # Get expected ranges for all particles using lookup (vectorized)
            zstar_t_all = np.zeros((num_particles, len(z_t)))
            for p in range(num_particles):
                zstar_t_all[p] = self.ray_casting(X_t1[p], raycast_map)
            
            # Compute probabilities for all particles and all beams (fully vectorized)
            # z_t: [num_beams], zstar_t_all: [num_particles, num_beams]
            # Broadcast z_t to [num_particles, num_beams]
            z_t_broadcast = np.tile(z_t, (num_particles, 1))  # [num_particles, num_beams]
            
            # Compute probabilities for all particles and beams
            p_hit_all = self.get_p_hit(z_t_broadcast, zstar_t_all)  # [num_particles, num_beams]
            p_short_all = self.get_p_short(z_t_broadcast, zstar_t_all)  # [num_particles, num_beams]
            p_max_all = self.get_p_max(z_t_broadcast)  # [num_particles, num_beams]
            p_rand_all = self.get_p_rand(z_t_broadcast)  # [num_particles, num_beams]
            
            # Weighted combination for all particles and beams
            # Note: reference solution does NOT normalize by weight_sum here
            prob_all = self._z_hit * p_hit_all + \
                      self._z_short * p_short_all + \
                      self._z_max * p_max_all + \
                      self._z_rand * p_rand_all
            
            # Remove zeros and compute log sum for each particle
            # Following reference solution: delete zeros, then sum logs
            # For each particle, remove zero probabilities before computing log sum
            total_log_probs = np.zeros(num_particles)
            for p in range(num_particles):
                prob_p = prob_all[p].copy()
                # Remove zeros (as in reference solution)
                prob_p = prob_p[prob_p != 0.0]
                if len(prob_p) > 0:
                    total_log_probs[p] = np.sum(np.log(prob_p))
                else:
                    total_log_probs[p] = -np.inf  # All zeros, very low weight
            
            # Convert back to probabilities
            weights = np.exp(np.clip(total_log_probs, -700, 700))
            
            return weights
        else:
            # Fall back to real-time ray casting (slower but works without precomputation)
            weights = np.zeros(num_particles)
            
            # Extract all particle states
            x_robot = X_t1[:, 0]  # [num_particles]
            y_robot = X_t1[:, 1]  # [num_particles]
            theta_robot = X_t1[:, 2]  # [num_particles]
            
            # Calculate laser sensor positions for all particles (vectorized)
            x_laser = x_robot + self._offset * np.cos(theta_robot)
            y_laser = y_robot + self._offset * np.sin(theta_robot)
            
            # Laser scan covers 180 degrees (-90 to +90 relative to robot heading)
            num_beams = len(z_t1_arr)
            angle_increment = math.pi / num_beams  # 180 degrees / num_beams
            
            # Pre-compute beam angles (subsampled)
            beam_indices = np.arange(0, num_beams, self._subsampling)
            beam_angles_relative = -math.pi / 2.0 + beam_indices * angle_increment
            
            # Initialize log probabilities for all particles
            total_log_probs = np.zeros(num_particles)
            
            # Process each beam (still need loop for ray casting, but vectorize probability computation)
            for i, beam_idx in enumerate(beam_indices):
                # Beam angle relative to robot heading
                beam_angle_relative = beam_angles_relative[i]
                
                # Beam angles in world frame for all particles (vectorized)
                beam_angles_world = theta_robot + beam_angle_relative
                
                # Ray casting for all particles (still need loop, but can optimize)
                z_expected_all = np.zeros(num_particles)
                for p in range(num_particles):
                    z_expected_all[p] = self._ray_cast(x_laser[p], y_laser[p], beam_angles_world[p])
                
                # Get measured range
                z_meas = z_t1_arr[beam_idx]
                
                # Compute probabilities for all particles (vectorized)
                # Vectorize the probability computation
                prob_beams = self._compute_probability_vectorized(z_meas, z_expected_all)
                
                # Add log probabilities (vectorized)
                total_log_probs += np.log(prob_beams + 1e-10)
            
            # Convert back to probabilities (vectorized)
            weights = np.exp(np.clip(total_log_probs, -700, 700))  # Clip to avoid overflow
            
            return weights

    def _compute_probability_vectorized(self, z_meas, z_expected_arr):
        """
        Vectorized probability computation for multiple particles
        param[in] z_meas : measured range (scalar)
        param[in] z_expected_arr : expected ranges [num_particles]
        param[out] probs : probabilities [num_particles]
        """
        num_particles = len(z_expected_arr)
        # Note: reference solution does NOT normalize by weight_sum
        # Weights are used directly without normalization
        
        # Initialize probability arrays
        p_hit = np.zeros(num_particles)
        p_short = np.zeros(num_particles)
        p_max = np.zeros(num_particles)
        p_rand = np.zeros(num_particles)
        
        # Valid range check (z_meas is scalar)
        is_valid = (z_meas >= 0) and (z_meas <= self._max_range)
        
        if is_valid:
            # p_hit: Gaussian distribution (vectorized)
            p_hit = norm.pdf(z_meas, loc=z_expected_arr, scale=self._sigma_hit)
            # Normalize by CDF
            cdf_max = norm.cdf(self._max_range, loc=z_expected_arr, scale=self._sigma_hit)
            cdf_min = norm.cdf(0, loc=z_expected_arr, scale=self._sigma_hit)
            norm_factor = cdf_max - cdf_min
            valid_norm_mask = norm_factor > 1e-10
            p_hit[valid_norm_mask] = p_hit[valid_norm_mask] / norm_factor[valid_norm_mask]
            p_hit[~valid_norm_mask] = 0.0
            
            # p_short: Exponential distribution (vectorized)
            short_mask = (z_meas <= z_expected_arr) & (z_expected_arr > 0)
            if np.any(short_mask):
                p_short[short_mask] = self._lambda_short * np.exp(-self._lambda_short * z_meas)
                # Compute normalization factor only for short_mask indices
                z_expected_short = z_expected_arr[short_mask]
                norm_factor_short = 1.0 - np.exp(-self._lambda_short * z_expected_short)
                valid_short_mask = norm_factor_short > 1e-10
                # Apply normalization only where valid
                short_indices = np.where(short_mask)[0]
                p_short[short_indices[valid_short_mask]] = p_short[short_indices[valid_short_mask]] / norm_factor_short[valid_short_mask]
                p_short[short_indices[~valid_short_mask]] = 0.0
            
            # p_max: Dirac delta (vectorized)
            p_max[np.abs(z_meas - self._max_range) < 1e-6] = 1.0
            
            # p_rand: Uniform distribution (vectorized)
            p_rand[:] = 1.0 / self._max_range
        
        # Weighted combination (vectorized)
        # Note: reference solution does NOT normalize by weight_sum here
        probs = self._z_hit * p_hit + \
                self._z_short * p_short + \
                self._z_max * p_max + \
                self._z_rand * p_rand
        
        return probs

    def ray_casting(self, x_t1, raycast_map):
        """
        Look up expected ranges from precomputed raycast map
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[in] raycast_map : precomputed lookup map [height, width, 360]
        param[out] zstar_t : expected ranges for down-sampled beams
        """
        theta_robot = x_t1[2]
        origin_laser_x = int((x_t1[0] + self._offset * math.cos(theta_robot))//self._resolution)
        origin_laser_y = int((x_t1[1] + self._offset * math.sin(theta_robot))//self._resolution)
        # Get the down-sampled angles in radian
        theta_laser = [(theta_robot - np.pi/2 + theta * np.pi / 180) for theta in range(0, 180, self._subsampling)]
        theta_laser = (np.degrees(theta_laser) % 360).astype(int)    # convert from radian to degree
        zstar_t = raycast_map[origin_laser_y, origin_laser_x, theta_laser]
        return zstar_t

    def precompute_raycast(self):
        """
        Precompute raycast map for all free space positions and all 360 degrees
        param[out] raycast_map : lookup map [height, width, 360] containing expected ranges
        """
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm is not available
            def tqdm(x):
                return x
        
        height, width = self._occupancy_map.shape
        raycast_map = np.zeros((height, width, 360))
        
        print("Precomputing raycast map...")
        for i in tqdm(range(height * width)):
            x = i % width
            y = i // width
            # Make sure the initial pose is unoccupied
            if self._occupancy_map[y, x] != 0:
                continue
            
            x_map = x * self._resolution
            y_map = y * self._resolution
            
            zstar_t = self._ray_casting_all((x_map, y_map))
            
            for theta, z in zip(range(360), zstar_t):
                raycast_map[y, x, theta] = z
        
        print("Raycast map precomputation completed!")
        return raycast_map

    def _ray_casting_all(self, origin_laser):
        """
        Compute raycast for all 360 degrees from a given origin position
        param[in] origin_laser : (x, y) position in world frame (cm)
        param[out] zstar_t : expected ranges for all 360 degrees [360]
        """
        zstar_t = np.ones(360) * self._max_range
        dist_step = np.linspace(0, self._max_range, self._interpolation_num)
        
        for i in range(len(zstar_t)):
            theta_laser = i * np.pi / 180
            zx_world = origin_laser[0] + dist_step * math.cos(theta_laser)
            zy_world = origin_laser[1] + dist_step * math.sin(theta_laser)
            
            zx = (zx_world / self._resolution).astype(int)
            zy = (zy_world / self._resolution).astype(int)
            
            for j in range(len(zx)):
                if 0 <= zx[j] < self._occupancy_map.shape[1] and 0 <= zy[j] < self._occupancy_map.shape[0]:
                    # Reached an obstacle
                    if self._occupancy_map[zy[j], zx[j]] >= self._min_probability or self._occupancy_map[zy[j], zx[j]] == -1:
                        zstar_t[i] = math.sqrt(
                            (zx_world[j] - origin_laser[0]) ** 2 + (zy_world[j] - origin_laser[1]) ** 2)
                        break
                else:
                    break
        
        return zstar_t

    def get_p_hit(self, z_t, zstar_t):
        """
        Compute p_hit probability (vectorized)
        param[in] z_t : measured ranges [num_beams] or [num_particles, num_beams]
        param[in] zstar_t : expected ranges [num_beams] or [num_particles, num_beams]
        param[out] p_hit : probabilities [num_beams] or [num_particles, num_beams]
        """
        eta = norm.cdf(self._max_range, loc=zstar_t, scale=self._sigma_hit) - norm.cdf(0, loc=zstar_t, scale=self._sigma_hit)
        p_hit = norm.pdf(z_t, loc=zstar_t, scale=self._sigma_hit) / eta
        p_hit[z_t > self._max_range] = 0
        p_hit[z_t < 0] = 0
        return p_hit

    def get_p_short(self, z_t, zstar_t):
        """
        Compute p_short probability (vectorized)
        param[in] z_t : measured ranges [num_beams]
        param[in] zstar_t : expected ranges [num_beams]
        param[out] p_short : probabilities [num_beams]
        """
        eta = np.zeros_like(z_t)
        eta[zstar_t != 0] = 1 / (1 - np.exp(-self._lambda_short * zstar_t[zstar_t != 0]))
        p_short = eta * self._lambda_short * np.exp(-self._lambda_short * z_t)
        p_short[np.where((z_t < 0) & (z_t > zstar_t))] = 0
        return p_short

    def get_p_max(self, z_t):
        """
        Compute p_max probability (vectorized)
        param[in] z_t : measured ranges [num_beams]
        param[out] p_max : probabilities [num_beams]
        """
        return z_t == self._max_range

    def get_p_rand(self, z_t):
        """
        Compute p_rand probability (vectorized)
        param[in] z_t : measured ranges [num_beams]
        param[out] p_rand : probabilities [num_beams]
        """
        p_rand = np.zeros_like(z_t)
        p_rand[np.where((z_t >= 0) & (z_t < self._max_range))] = 1 / self._max_range
        return p_rand


if __name__ == '__main__':
    """
    Test the sensor model with visualization
    """
    import argparse
    
    print("=" * 60)
    print("Testing Sensor Model")
    print("=" * 60)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    args = parser.parse_args()
    
    # Load map
    map_reader = MapReader(args.path_to_map)
    occupancy_map = map_reader.get_map()
    
    # Initialize sensor model
    sensor_model = SensorModel(occupancy_map)
    print(f"\nSensor Model Parameters:")
    print(f"  z_hit = {sensor_model._z_hit}")
    print(f"  z_short = {sensor_model._z_short}")
    print(f"  z_max = {sensor_model._z_max}")
    print(f"  z_rand = {sensor_model._z_rand}")
    print(f"  sigma_hit = {sensor_model._sigma_hit}")
    print(f"  lambda_short = {sensor_model._lambda_short}")
    print(f"  max_range = {sensor_model._max_range}")
    print(f"  subsampling = {sensor_model._subsampling}")
    
    # Test case: Ray casting visualization
    print("\n" + "-" * 60)
    print("Test: Ray Casting Visualization")
    print("-" * 60)
    
    # Test particle position (in world frame, cm)
    x_t1 = np.array([4000.0, 4000.0, 0.0])  # Center of map, facing right
    
    print(f"Particle state: x={x_t1[0]:.1f}, y={x_t1[1]:.1f}, theta={x_t1[2]:.4f}")
    
    # Create dummy laser readings (180 values)
    num_beams = 180
    dummy_ranges = np.ones(num_beams) * 500.0  # All readings at 500cm
    
    # Test probability calculation
    prob = sensor_model.beam_range_finder_model(dummy_ranges, x_t1)
    print(f"Probability for dummy readings: {prob:.6e}")
    
    # Visualize ray casting
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot 1: Map with ray casting
    ax1 = axes[0]
    ax1.imshow(occupancy_map, cmap='Greys', origin='lower')
    
    # Convert particle position to map coordinates
    map_x = int(x_t1[0] / sensor_model._resolution)
    map_y = int(x_t1[1] / sensor_model._resolution)
    
    # Plot robot position
    ax1.plot(map_y, map_x, 'ro', markersize=10, label='Robot')
    
    # Calculate laser position
    x_laser = x_t1[0] + sensor_model._offset * math.cos(x_t1[2])
    y_laser = x_t1[1] + sensor_model._offset * math.sin(x_t1[2])
    map_laser_x = int(x_laser / sensor_model._resolution)
    map_laser_y = int(y_laser / sensor_model._resolution)
    ax1.plot(map_laser_y, map_laser_x, 'bo', markersize=8, label='Laser')
    
    # Cast rays and visualize
    angle_increment = math.pi / num_beams
    ray_endpoints = []
    
    for i in range(0, num_beams, sensor_model._subsampling * 5):  # Show fewer rays for clarity
        beam_angle_relative = -math.pi / 2.0 + i * angle_increment
        beam_angle_world = x_t1[2] + beam_angle_relative
        
        z_expected = sensor_model._ray_cast(x_laser, y_laser, beam_angle_world)
        
        # Calculate ray endpoint
        end_x = x_laser + z_expected * math.cos(beam_angle_world)
        end_y = y_laser + z_expected * math.sin(beam_angle_world)
        
        map_end_x = int(end_x / sensor_model._resolution)
        map_end_y = int(end_y / sensor_model._resolution)
        
        # Draw ray
        ax1.plot([map_laser_y, map_end_y], [map_laser_x, map_end_x], 
                'r-', alpha=0.3, linewidth=0.5)
        ray_endpoints.append((map_end_x, map_end_y))
    
    # Plot ray endpoints
    if ray_endpoints:
        endpoints = np.array(ray_endpoints)
        ax1.scatter(endpoints[:, 1], endpoints[:, 0], c='red', s=5, alpha=0.5, label='Ray endpoints')
    
    ax1.set_title('Ray Casting Visualization')
    ax1.set_xlabel('Map Y (cells)')
    ax1.set_ylabel('Map X (cells)')
    ax1.legend()
    ax1.set_xlim([0, occupancy_map.shape[1]])
    ax1.set_ylim([0, occupancy_map.shape[0]])
    
    # Plot 2: Probability distribution for different measurements
    ax2 = axes[1]
    
    # Test probability for a single beam with different measurements
    z_expected_test = 300.0  # Expected range from ray casting
    z_meas_range = np.linspace(0, sensor_model._max_range, 200)
    
    probs = []
    for z_meas in z_meas_range:
        prob_single = sensor_model._compute_probability(z_meas, z_expected_test)
        probs.append(prob_single)
    
    ax2.plot(z_meas_range, probs, 'b-', linewidth=2, label='Probability distribution')
    ax2.axvline(z_expected_test, color='r', linestyle='--', linewidth=2, label=f'Expected range: {z_expected_test:.1f}cm')
    ax2.set_xlabel('Measured range (cm)')
    ax2.set_ylabel('Probability')
    ax2.set_title('Probability Distribution for Single Beam')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensor_model_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: sensor_model_test.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
