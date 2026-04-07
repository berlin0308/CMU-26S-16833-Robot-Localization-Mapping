'''
Benchmark script to compare vectorized vs non-vectorized implementations
'''

import argparse
import numpy as np
import time
import sys
import os
import math

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel

def benchmark_motion_model(motion_model, num_particles, num_iterations=100):
    """Benchmark motion model update"""
    print(f"\n{'='*60}")
    print(f"Motion Model Benchmark ({num_particles} particles, {num_iterations} iterations)")
    print(f"{'='*60}")
    
    u_t0 = np.array([0.0, 0.0, 0.0])
    u_t1 = np.array([100.0, 50.0, 0.1])
    X_t0 = np.random.uniform(0, 7000, (num_particles, 3))
    X_t0[:, 2] = np.random.uniform(-math.pi, math.pi, num_particles)
    
    # Non-vectorized version
    start_time = time.time()
    for _ in range(num_iterations):
        X_t1_loop = np.zeros((num_particles, 3))
        for m in range(num_particles):
            X_t1_loop[m] = motion_model.update(u_t0, u_t1, X_t0[m])
    loop_time = time.time() - start_time
    
    # Vectorized version
    start_time = time.time()
    for _ in range(num_iterations):
        X_t1_vec = motion_model.update_vectorized(u_t0, u_t1, X_t0)
    vec_time = time.time() - start_time
    
    speedup = loop_time / vec_time if vec_time > 0 else 0
    
    print(f"Loop-based time:    {loop_time:.4f}s ({loop_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Vectorized time:     {vec_time:.4f}s ({vec_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Speedup:             {speedup:.2f}x")
    
    return loop_time, vec_time, speedup

def benchmark_sensor_model(sensor_model, num_particles, num_iterations=10):
    """Benchmark sensor model update"""
    print(f"\n{'='*60}")
    print(f"Sensor Model Benchmark ({num_particles} particles, {num_iterations} iterations)")
    print(f"{'='*60}")
    
    # Create dummy laser readings
    num_beams = 180
    z_t = np.random.uniform(100, 800, num_beams)
    X_t1 = np.random.uniform(1000, 6000, (num_particles, 3))
    X_t1[:, 2] = np.random.uniform(-math.pi, math.pi, num_particles)
    
    # Non-vectorized version
    start_time = time.time()
    for _ in range(num_iterations):
        weights_loop = np.zeros(num_particles)
        for m in range(num_particles):
            weights_loop[m] = sensor_model.beam_range_finder_model(z_t, X_t1[m])
    loop_time = time.time() - start_time
    
    # Vectorized version
    start_time = time.time()
    for _ in range(num_iterations):
        weights_vec = sensor_model.beam_range_finder_model_vectorized(z_t, X_t1)
    vec_time = time.time() - start_time
    
    speedup = loop_time / vec_time if vec_time > 0 else 0
    
    print(f"Loop-based time:    {loop_time:.4f}s ({loop_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Vectorized time:     {vec_time:.4f}s ({vec_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Speedup:             {speedup:.2f}x")
    
    return loop_time, vec_time, speedup

def benchmark_full_iteration(motion_model, sensor_model, num_particles, num_iterations=5):
    """Benchmark full particle filter iteration"""
    print(f"\n{'='*60}")
    print(f"Full Iteration Benchmark ({num_particles} particles, {num_iterations} iterations)")
    print(f"{'='*60}")
    
    u_t0 = np.array([0.0, 0.0, 0.0])
    u_t1 = np.array([100.0, 50.0, 0.1])
    X_bar = np.random.uniform(0, 7000, (num_particles, 4))
    X_bar[:, 2] = np.random.uniform(-math.pi, math.pi, num_particles)
    X_bar[:, 3] = 1.0 / num_particles
    
    num_beams = 180
    z_t = np.random.uniform(100, 800, num_beams)
    
    # Non-vectorized version
    start_time = time.time()
    for _ in range(num_iterations):
        X_bar_new = np.zeros((num_particles, 4))
        X_t0 = X_bar[:, 0:3]
        for m in range(num_particles):
            x_t1 = motion_model.update(u_t0, u_t1, X_t0[m])
            w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
            X_bar_new[m] = np.hstack((x_t1, w_t))
    loop_time = time.time() - start_time
    
    # Vectorized version
    start_time = time.time()
    for _ in range(num_iterations):
        X_t0 = X_bar[:, 0:3]
        X_t1 = motion_model.update_vectorized(u_t0, u_t1, X_t0)
        weights = sensor_model.beam_range_finder_model_vectorized(z_t, X_t1)
        X_bar_new = np.hstack((X_t1, weights.reshape(-1, 1)))
    vec_time = time.time() - start_time
    
    speedup = loop_time / vec_time if vec_time > 0 else 0
    
    print(f"Loop-based time:    {loop_time:.4f}s ({loop_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Vectorized time:     {vec_time:.4f}s ({vec_time/num_iterations*1000:.2f}ms per iteration)")
    print(f"Speedup:             {speedup:.2f}x")
    
    return loop_time, vec_time, speedup

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--num_iterations', default=10, type=int, help='Number of iterations for benchmark')
    args = parser.parse_args()
    
    print("="*60)
    print("Particle Filter Vectorization Benchmark")
    print("="*60)
    print(f"Platform: Apple M2")
    print(f"Number of particles: {args.num_particles}")
    print(f"Iterations per test: {args.num_iterations}")
    
    # Load map
    map_obj = MapReader(args.path_to_map)
    occupancy_map = map_obj.get_map()
    
    # Initialize models
    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    
    # Run benchmarks
    motion_loop, motion_vec, motion_speedup = benchmark_motion_model(
        motion_model, args.num_particles, args.num_iterations)
    
    sensor_loop, sensor_vec, sensor_speedup = benchmark_sensor_model(
        sensor_model, args.num_particles, args.num_iterations)
    
    full_loop, full_vec, full_speedup = benchmark_full_iteration(
        motion_model, sensor_model, args.num_particles, args.num_iterations)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Component':<20} {'Loop (s)':<15} {'Vectorized (s)':<15} {'Speedup':<10}")
    print(f"{'-'*60}")
    print(f"{'Motion Model':<20} {motion_loop:<15.4f} {motion_vec:<15.4f} {motion_speedup:<10.2f}x")
    print(f"{'Sensor Model':<20} {sensor_loop:<15.4f} {sensor_vec:<15.4f} {sensor_speedup:<10.2f}x")
    print(f"{'Full Iteration':<20} {full_loop:<15.4f} {full_vec:<15.4f} {full_speedup:<10.2f}x")
    print(f"{'='*60}")




