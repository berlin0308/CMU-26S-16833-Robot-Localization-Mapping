'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


def _wrap_angle(angle):
    angle_rad = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
    return angle_rad


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.00001
        self._alpha2 = 0.00001
        self._alpha3 = 0.0001
        self._alpha4 = 0.0001


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        # Extract odometry readings
        # u_t0 = [x_bar, y_bar, theta_bar] at time (t-1) in odometry frame
        # u_t1 = [x_bar', y_bar', theta_bar'] at time t in odometry frame
        x_bar = u_t0[0]
        y_bar = u_t0[1]
        theta_bar = u_t0[2]
        x_bar_prime = u_t1[0]
        y_bar_prime = u_t1[1]
        theta_bar_prime = u_t1[2]
        
        # Previous particle state in world frame
        x = x_t0[0]
        y = x_t0[1]
        theta = x_t0[2]
        
        # Compute motion components from odometry readings
        # Following the odometry model: u = <delta_rot1, delta_rot2, delta_trans>
        delta_rot1 = math.atan2(y_bar_prime - y_bar, x_bar_prime - x_bar) - theta_bar
        delta_trans = math.sqrt((x_bar_prime - x_bar)**2 + (y_bar_prime - y_bar)**2)
        delta_rot2 = theta_bar_prime - theta_bar - delta_rot1
        
        # Apply motion uncertainty model with noise sampling
        # Compute variance using squared terms (following Probabilistic Robotics)
        # Variance for rotation 1: alpha_1 * delta_rot1^2 + alpha_2 * delta_trans^2
        variance_rot1 = self._alpha1 * delta_rot1 ** 2 + self._alpha2 * delta_trans ** 2
        # Variance for translation: alpha_3 * delta_trans^2 + alpha_4 * (delta_rot1^2 + delta_rot2^2)
        variance_trans = self._alpha3 * delta_trans ** 2 + self._alpha4 * delta_rot1 ** 2 + self._alpha4 * delta_rot2 ** 2
        # Variance for rotation 2: alpha_1 * delta_rot2^2 + alpha_2 * delta_trans^2
        variance_rot2 = self._alpha1 * delta_rot2 ** 2 + self._alpha2 * delta_trans ** 2
        
        # Sample noise from zero-mean Gaussian distributions
        noise_rot1 = np.random.normal(0.0, math.sqrt(variance_rot1))
        noise_trans = np.random.normal(0.0, math.sqrt(variance_trans))
        noise_rot2 = np.random.normal(0.0, math.sqrt(variance_rot2))
        
        # Apply noise (subtract noise as in reference solution)
        delta_rot1_hat = delta_rot1 - noise_rot1
        delta_trans_hat = delta_trans - noise_trans
        delta_rot2_hat = delta_rot2 - noise_rot2
        
        # Update particle pose using noisy motion components
        # Following sample_motion_model algorithm
        x_t1 = np.zeros(3)
        x_t1[0] = x + delta_trans_hat * math.cos(theta + delta_rot1_hat)
        x_t1[1] = y + delta_trans_hat * math.sin(theta + delta_rot1_hat)
        x_t1[2] = _wrap_angle(theta + delta_rot1_hat + delta_rot2_hat)
        
        return x_t1

    def update_vectorized(self, u_t0, u_t1, X_t0):
        """
        Vectorized version: update all particles at once
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] X_t0 : all particle states [num_particles x 3] at time (t-1) [world_frame]
        param[out] X_t1 : all particle states [num_particles x 3] at time t [world_frame]
        """
        # Extract odometry readings
        x_bar = u_t0[0]
        y_bar = u_t0[1]
        theta_bar = u_t0[2]
        x_bar_prime = u_t1[0]
        y_bar_prime = u_t1[1]
        theta_bar_prime = u_t1[2]
        
        # Extract all particle states
        x = X_t0[:, 0]  # [num_particles]
        y = X_t0[:, 1]  # [num_particles]
        theta = X_t0[:, 2]  # [num_particles]
        
        # Compute motion components from odometry readings (same for all particles)
        delta_rot1 = math.atan2(y_bar_prime - y_bar, x_bar_prime - x_bar) - theta_bar
        delta_trans = math.sqrt((x_bar_prime - x_bar)**2 + (y_bar_prime - y_bar)**2)
        delta_rot2 = theta_bar_prime - theta_bar - delta_rot1
        
        num_particles = X_t0.shape[0]
        
        # Apply motion uncertainty model with noise sampling (vectorized)
        # Compute variance using squared terms (following Probabilistic Robotics)
        variance_rot1 = self._alpha1 * delta_rot1 ** 2 + self._alpha2 * delta_trans ** 2
        variance_trans = self._alpha3 * delta_trans ** 2 + self._alpha4 * delta_rot1 ** 2 + self._alpha4 * delta_rot2 ** 2
        variance_rot2 = self._alpha1 * delta_rot2 ** 2 + self._alpha2 * delta_trans ** 2
        
        # Sample noise from zero-mean Gaussian distributions (vectorized)
        noise_std_rot1 = math.sqrt(variance_rot1)
        noise_std_trans = math.sqrt(variance_trans)
        noise_std_rot2 = math.sqrt(variance_rot2)
        
        noise_rot1 = np.random.normal(0.0, noise_std_rot1, num_particles)
        noise_trans = np.random.normal(0.0, noise_std_trans, num_particles)
        noise_rot2 = np.random.normal(0.0, noise_std_rot2, num_particles)
        
        # Apply noise (subtract noise as in reference solution)
        delta_rot1_hat = delta_rot1 - noise_rot1
        delta_trans_hat = delta_trans - noise_trans
        delta_rot2_hat = delta_rot2 - noise_rot2
        
        # Update all particle poses using noisy motion components (vectorized)
        X_t1 = np.zeros((num_particles, 3))
        X_t1[:, 0] = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
        X_t1[:, 1] = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
        X_t1[:, 2] = _wrap_angle(theta + delta_rot1_hat + delta_rot2_hat)
        
        return X_t1


if __name__ == '__main__':
    """
    Test the motion model with visualization
    """
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing Motion Model")
    print("=" * 60)
    
    # Initialize motion model
    motion_model = MotionModel()
    print(f"\nMotion Model Parameters:")
    print(f"  alpha1 = {motion_model._alpha1}")
    print(f"  alpha2 = {motion_model._alpha2}")
    print(f"  alpha3 = {motion_model._alpha3}")
    print(f"  alpha4 = {motion_model._alpha4}")
    
    # Test case: Multiple particles with same odometry (showing noise effect)
    print("\n" + "-" * 60)
    print("Test: Multiple particles with same odometry")
    print("-" * 60)
    
    u_t0 = np.array([0.0, 0.0, 0.0])
    u_t1 = np.array([100.0, 0.0, 0.0])  # Move 100 units forward
    x_t0 = np.array([0.0, 0.0, 0.0])
    
    print(f"Odometry reading at t-1: {u_t0}")
    print(f"Odometry reading at t:   {u_t1}")
    print(f"Initial particle state:  {x_t0}")
    print(f"\nRunning 50 particles with same odometry (showing noise):")
    
    # Collect results for visualization
    particles_start = []
    particles_end = []
    
    for i in range(50):
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        particles_start.append(x_t0.copy())
        particles_end.append(x_t1.copy())
        if i < 5:
            print(f"  Particle {i+1}: [{x_t1[0]:.2f}, {x_t1[1]:.2f}, {x_t1[2]:.6f}]")
    
    # Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Top-down view (x-y plane)
    ax1 = axes[0]
    
    # Plot odometry path
    ax1.plot([u_t0[0], u_t1[0]], [u_t0[1], u_t1[1]], 'b-', linewidth=3, 
             label='Odometry path', marker='o', markersize=10)
    ax1.arrow(u_t0[0], u_t0[1], 30*math.cos(u_t0[2]), 30*math.sin(u_t0[2]), 
              head_width=10, head_length=8, fc='blue', ec='blue', label='Initial heading')
    ax1.arrow(u_t1[0], u_t1[1], 30*math.cos(u_t1[2]), 30*math.sin(u_t1[2]), 
              head_width=10, head_length=8, fc='green', ec='green', label='Final heading')
    
    # Plot particles
    particles_start = np.array(particles_start)
    particles_end = np.array(particles_end)
    
    ax1.scatter(particles_start[:, 0], particles_start[:, 1], c='red', marker='x', 
                s=50, label='Particles (t-1)', alpha=0.6)
    ax1.scatter(particles_end[:, 0], particles_end[:, 1], c='orange', marker='o', 
                s=30, label='Particles (t)', alpha=0.6)
    
    # Draw arrows for particle movement (sample a few)
    for i in range(0, len(particles_start), 5):
        dx = particles_end[i, 0] - particles_start[i, 0]
        dy = particles_end[i, 1] - particles_start[i, 1]
        ax1.arrow(particles_start[i, 0], particles_start[i, 1], dx, dy,
                 head_width=5, head_length=4, fc='gray', ec='gray', alpha=0.4, 
                 length_includes_head=True)
    
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Motion Model: Particle Distribution (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Orientation distribution
    ax2 = axes[1]
    
    # Compute orientation changes
    theta_changes = particles_end[:, 2] - particles_start[:, 2]
    expected_theta = u_t1[2] - u_t0[2]
    
    ax2.hist(theta_changes, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(expected_theta, color='green', linestyle='--', linewidth=2, 
                label=f'Expected: {expected_theta:.4f}')
    ax2.axvline(np.mean(theta_changes), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(theta_changes):.4f}')
    
    ax2.set_xlabel('Orientation change (radians)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Orientation Change Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('motion_model_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: motion_model_test.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
