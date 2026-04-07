'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os
import math

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_raycast(particle_state, sensor_model, occupancy_map, num_beams=180, subsampling_factor=10):
    """
    Visualize ray casting from a particle position
    param[in] particle_state : [x, y, theta] in world frame (cm)
    param[in] sensor_model : SensorModel instance
    param[in] occupancy_map : occupancy map
    param[in] num_beams : number of laser beams (default 180)
    param[in] subsampling_factor : show every Nth ray for clarity (default 10)
    """
    x_robot = particle_state[0]
    y_robot = particle_state[1]
    theta_robot = particle_state[2]
    
    # Calculate laser sensor position (25cm forward from robot center)
    x_laser = x_robot + sensor_model._offset * math.cos(theta_robot)
    y_laser = y_robot + sensor_model._offset * math.sin(theta_robot)
    
    # Convert to map coordinates
    map_laser_x = x_laser / sensor_model._resolution
    map_laser_y = y_laser / sensor_model._resolution
    map_robot_x = x_robot / sensor_model._resolution
    map_robot_y = y_robot / sensor_model._resolution
    
    # Cast rays and visualize
    angle_increment = math.pi / num_beams
    ray_endpoints = []
    
    for i in range(0, num_beams, subsampling_factor):
        beam_angle_relative = -math.pi / 2.0 + i * angle_increment
        beam_angle_world = theta_robot + beam_angle_relative
        
        z_expected = sensor_model._ray_cast(x_laser, y_laser, beam_angle_world)
        
        # Calculate ray endpoint
        end_x = x_laser + z_expected * math.cos(beam_angle_world)
        end_y = y_laser + z_expected * math.sin(beam_angle_world)
        
        map_end_x = end_x / sensor_model._resolution
        map_end_y = end_y / sensor_model._resolution
        
        # Draw ray
        plt.plot([map_laser_y, map_end_y], [map_laser_x, map_end_x], 
                'r-', alpha=0.3, linewidth=0.5)
        ray_endpoints.append((map_end_x, map_end_y))
    
    # Plot ray endpoints
    if ray_endpoints:
        endpoints = np.array(ray_endpoints)
        plt.scatter(endpoints[:, 1], endpoints[:, 0], c='red', s=3, alpha=0.5)
    
    # Plot robot and laser positions
    plt.plot(map_robot_y, map_robot_x, 'go', markersize=8, label='Robot')
    plt.plot(map_laser_y, map_laser_x, 'bo', markersize=6, label='Laser')


def visualize_timestep(X_bar, tstep, output_path, occupancy_map, sensor_model=None, show_raycast=False, num_raycast_particles=3):
    """
    Visualize particles on the occupancy map
    param[in] X_bar : particle states [num_particles x 4]
    param[in] tstep : time step
    param[in] output_path : path to save figure
    param[in] occupancy_map : occupancy map
    param[in] sensor_model : SensorModel instance (optional, for ray casting visualization)
    param[in] show_raycast : whether to show ray casting (default False)
    param[in] num_raycast_particles : number of particles to show ray casting for (default 3)
    """
    # Clear previous plot
    plt.clf()
    
    # Display occupancy map
    plt.imshow(occupancy_map, cmap='Greys', origin='lower', extent=[0, 800, 0, 800])
    
    # Convert particle positions from cm to map coordinates (10cm per cell, 800 cells = 8000cm)
    x_locs = X_bar[:, 0] / 10.0  # Convert cm to map units (0-800)
    y_locs = X_bar[:, 1] / 10.0  # Convert cm to map units (0-800)
    
    # Plot particles
    plt.scatter(x_locs, y_locs, c='r', marker='o', s=5, alpha=0.6)
    
    # Visualize ray casting for top particles (highest weights) if enabled
    if show_raycast and sensor_model is not None:
        # Get top N particles by weight
        weights = X_bar[:, 3]
        top_indices = np.argsort(weights)[-num_raycast_particles:]
        
        for idx in top_indices:
            particle_state = X_bar[idx, 0:3]
            visualize_raycast(particle_state, sensor_model, occupancy_map)
    
    # Set title and labels
    title = f'Particle Filter Localization - Time Step {tstep}'
    if show_raycast:
        title += f' (showing raycast for top {num_raycast_particles} particles)'
    plt.title(title)
    plt.xlabel('X position (map units)')
    plt.ylabel('Y position (map units)')
    plt.axis([0, 800, 0, 800])
    
    # Save figure
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep), dpi=100, bbox_inches='tight')
    plt.pause(0.00001)


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_region(num_particles, occupancy_map, x_min, x_max, y_min, y_max):
    """
    Initialize particles in a specified region of the map (for faster debugging)
    param[in] num_particles : number of particles
    param[in] occupancy_map : occupancy map
    param[in] x_min, x_max, y_min, y_max : region bounds in cm
    param[out] X_bar_init : initialized particles [num_particles x 4]
    """
    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(y_min, y_max, (num_particles, 1))
    x0_vals = np.random.uniform(x_min, x_max, (num_particles, 1))
    theta0_vals = np.random.uniform(-math.pi, math.pi, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    Initialize particles in free space areas only (faster convergence)
    """
    map_resolution = 10  # 10cm per cell
    min_probability = 0.35  # Threshold for free space
    
    # Find all free space cells (occupancy < threshold)
    free_space_mask = occupancy_map < min_probability
    free_space_indices = np.where(free_space_mask)
    
    if len(free_space_indices[0]) == 0:
        # Fallback to random initialization if no free space found
        return init_particles_random(num_particles, occupancy_map)
    
    # Randomly select free space positions
    num_free_cells = len(free_space_indices[0])
    selected_indices = np.random.choice(num_free_cells, size=num_particles, replace=True)
    
    # Convert map cell indices to world coordinates (cm)
    map_x_indices = free_space_indices[0][selected_indices]
    map_y_indices = free_space_indices[1][selected_indices]
    
    # Convert to world frame coordinates (center of cell)
    x0_vals = (map_x_indices * map_resolution + map_resolution / 2.0).reshape(-1, 1)
    y0_vals = (map_y_indices * map_resolution + map_resolution / 2.0).reshape(-1, 1)
    
    # Random orientations
    theta0_vals = np.random.uniform(-math.pi, math.pi, (num_particles, 1))
    
    # Initialize weights uniformly
    w0_vals = np.ones((num_particles, 1), dtype=np.float64) / num_particles
    
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    
    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=3000, type=int)
    parser.add_argument('--visualize', action='store_false')
    parser.add_argument('--path_to_raycast_map', default='raycast_map.npy')
    parser.add_argument('--debug_motion', action='store_true', help='Display motion model intermediate results')
    parser.add_argument('--no-vectorized', dest='vectorized', action='store_false', default=True, help='Disable vectorized implementation (default: vectorized enabled)')
    parser.add_argument('--video', action='store_true', help='Generate video from visualization images after completion')
    parser.add_argument('--video_fps', default=10, type=int, help='Frames per second for video (before speed multiplier)')
    parser.add_argument('--video_speed', default=5, type=int, help='Video speed multiplier (e.g., 5 = 5x speed)')
    parser.add_argument('--skip-odometry-only', action='store_true', default=False, help='Skip pure odometry measurements')
    parser.add_argument('--no-skip-odometry-only', dest='skip_odometry_only', action='store_false', help='Process all measurements including pure odometry')
    parser.add_argument('--debug-weights', action='store_true', help='Print weight statistics after each laser measurement')
    parser.add_argument('--debug-particles', action='store_true', help='Print particle position and orientation statistics')
    parser.add_argument('--debug-raycast', action='store_true', help='Visualize ray casting for top particles (per tips_for_students.md)')
    parser.add_argument('--raycast-particles', type=int, default=3, help='Number of particles to show ray casting for (default: 3)')
    parser.add_argument('--init-region', nargs=4, type=float, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX'), help='Initialize particles only in specified region (x_min x_max y_min y_max in cm)')
    parser.add_argument('--quick-debug', action='store_true', help='Enable quick debug mode: skip odometry-only, reduce particles, enable debug outputs')
    args = parser.parse_args()
    
    # If --video is requested, automatically enable --visualize
    if args.video and not args.visualize:
        args.visualize = True
        print("Note: --video requires --visualize. Visualization enabled automatically.")

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if not os.path.exists(args.path_to_raycast_map):
        print("Start pre-computing the ray cast map")
        raycast_map = sensor_model.precompute_raycast()
        np.save(args.path_to_raycast_map, raycast_map)
        print('Pre-compute of ray casting done!')
    else:
        raycast_map = np.load(args.path_to_raycast_map)

    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # Handle pure odometry measurements (per tips_for_students.md)
        # Update odometry variables but don't update particles
        if args.skip_odometry_only and meas_type == "O":
            if not first_time_idx:
                # Update odometry: shift u_t1 to u_t0, set u_t1 to current odometry
                # u_t1 should already be defined from previous measurement
                u_t0 = u_t1
                u_t1 = odometry_robot
            else:
                # First measurement: initialize u_t0 and u_t1
                u_t0 = odometry_robot
                u_t1 = odometry_robot  # Initialize u_t1 as well
                first_time_idx = False
            continue  # Skip particle update for pure odometry

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            # First measurement: initialize both u_t0 and u_t1
            u_t0 = odometry_robot
            u_t1 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        if args.vectorized:
            """
            VECTORIZED IMPLEMENTATION
            """
            # Extract all particle states
            X_t0 = X_bar[:, 0:3]  # [num_particles x 3]
            
            # Motion model update for all particles (vectorized)
            X_t1 = motion_model.update_vectorized(u_t0, u_t1, X_t0)
            
            # Sensor model update for all particles (vectorized)
            if (meas_type == "L"):
                z_t = ranges
                weights = sensor_model.beam_range_finder_model_vectorized(z_t, X_t1, raycast_map)
                X_bar_new = np.hstack((X_t1, weights.reshape(-1, 1)))
            else:
                X_bar_new = np.hstack((X_t1, X_bar[:, 3:4]))
        else:
            """
            LOOP-BASED IMPLEMENTATION (original)
            """
            for m in range(0, num_particles):
                """
                MOTION MODEL
                """
                x_t0 = X_bar[m, 0:3]
                x_t1 = motion_model.update(u_t0, u_t1, x_t0)
                
                # Display motion model intermediate results if debug flag is set
                if args.debug_motion and m < 3:  # Show first 3 particles only
                    print(f"\n  Particle {m} Motion Model:")
                    print(f"    u_t0 (odom t-1): [{u_t0[0]:.2f}, {u_t0[1]:.2f}, {u_t0[2]:.4f}]")
                    print(f"    u_t1 (odom t):   [{u_t1[0]:.2f}, {u_t1[1]:.2f}, {u_t1[2]:.4f}]")
                    print(f"    x_t0 (state t-1): [{x_t0[0]:.2f}, {x_t0[1]:.2f}, {x_t0[2]:.4f}]")
                    print(f"    x_t1 (state t):   [{x_t1[0]:.2f}, {x_t1[1]:.2f}, {x_t1[2]:.4f}]")
                    # Compute motion components for display
                    delta_trans = math.sqrt((u_t1[0] - u_t0[0])**2 + (u_t1[1] - u_t0[1])**2)
                    delta_rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
                    delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1
                    print(f"    Motion components: δ_trans={delta_trans:.4f}, δ_rot1={delta_rot1:.4f}, δ_rot2={delta_rot2:.4f}")

                """
                SENSOR MODEL
                """
                if (meas_type == "L"):
                    z_t = ranges
                    w_t = sensor_model.beam_range_finder_model(z_t, x_t1, raycast_map)
                    X_bar_new[m, :] = np.hstack((x_t1, w_t))
                else:
                    X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        # Debug: Print weight statistics if enabled
        if args.debug_weights and meas_type == "L":
            weights = X_bar[:, 3]
            valid_weights = weights[weights > 0]
            if len(valid_weights) > 0:
                # Calculate entropy: -sum(p * log(p)) where p is normalized probability
                normalized_weights = valid_weights / np.sum(valid_weights)
                entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-10))
                print(f"  Weight stats: min={np.min(valid_weights):.2e}, max={np.max(valid_weights):.2e}, "
                      f"mean={np.mean(valid_weights):.2e}, std={np.std(valid_weights):.2e}, "
                      f"entropy={entropy:.4f}, valid={len(valid_weights)}/{len(weights)}")
            else:
                print(f"  Weight stats: All weights are zero!")

        # Debug: Print particle statistics if enabled
        if args.debug_particles and meas_type == "L":
            x_positions = X_bar[:, 0]
            y_positions = X_bar[:, 1]
            theta_positions = X_bar[:, 2]
            print(f"  Particle stats: x=[{np.min(x_positions):.1f}, {np.max(x_positions):.1f}], "
                  f"y=[{np.min(y_positions):.1f}, {np.max(y_positions):.1f}], "
                  f"x_std={np.std(x_positions):.1f}, y_std={np.std(y_positions):.1f}, "
                  f"theta_std={np.std(theta_positions):.4f}")

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output, occupancy_map, 
                             sensor_model=sensor_model if args.debug_raycast else None,
                             show_raycast=args.debug_raycast,
                             num_raycast_particles=args.raycast_particles)
    
    # Generate video if requested
    if args.video:
        print("\n" + "=" * 60)
        print("Generating video from visualization images...")
        print("=" * 60)
        try:
            from create_video import create_video_from_images
            # Generate video output path based on log file name
            log_basename = os.path.splitext(os.path.basename(args.path_to_log))[0]
            video_output = os.path.join(os.path.dirname(args.output), f'{log_basename}_localization.mp4')
            create_video_from_images(args.output, video_output, args.video_fps, args.video_speed)
        except Exception as e:
            print(f"Error generating video: {e}")
            print("You can manually generate video using:")
            print(f"  python create_video.py --image_dir {args.output} --output {video_output} --fps {args.video_fps} --speed {args.video_speed}")
