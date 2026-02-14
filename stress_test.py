#!/usr/bin/env python3

import time
import random
import math
import os
import sys
from vdb import VectorDatabase, VDBMetric

try:
  import matplotlib.pyplot as plt
  import matplotlib.gridspec as gridspec
  from matplotlib.patches import Rectangle
except ImportError:
  print("Installing matplotlib...")
  import subprocess
  subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "--quiet"])
  import matplotlib.pyplot as plt
  import matplotlib.gridspec as gridspec
  from matplotlib.patches import Rectangle

def generate_random_vector(dims):
  return [random.gauss(0, 1) for _ in range(dims)]

def normalize_vector(vec):
  magnitude = math.sqrt(sum(x * x for x in vec))
  if magnitude == 0:
    return vec
  return [x / magnitude for x in vec]

def benchmark_insertion(dims, counts):
  print(f"\n{'='*60}")
  print(f"BENCHMARK: Insertion Performance")
  print(f"{'='*60}")
  
  results = {'counts': [], 'times': [], 'throughput': []}
  
  for count in counts:
    print(f"  Testing {count:,} vectors with {dims} dimensions...")
    
    db = VectorDatabase(dims, VDBMetric.COSINE)
    vectors = [generate_random_vector(dims) for _ in range(count)]
    
    start = time.time()
    for i, vec in enumerate(vectors):
      db.add_vector(vec, f"vec_{i}")
    elapsed = time.time() - start
    
    throughput = count / elapsed
    results['counts'].append(count)
    results['times'].append(elapsed)
    results['throughput'].append(throughput)
    
    print(f"    Time: {elapsed:.3f}s, Throughput: {throughput:.0f} vectors/sec")
    del db
  
  return results

def benchmark_dimensionality(dimensions, fixed_count=1000):
  print(f"\n{'='*60}")
  print(f"BENCHMARK: Dimensionality Impact")
  print(f"{'='*60}")
  
  results = {'dimensions': [], 'insert_time': [], 'search_time': []}
  
  for dims in dimensions:
    print(f"  Testing {dims} dimensions with {fixed_count} vectors...")
    
    db = VectorDatabase(dims, VDBMetric.COSINE)
    vectors = [generate_random_vector(dims) for _ in range(fixed_count)]
    
    start = time.time()
    for i, vec in enumerate(vectors):
      db.add_vector(vec, f"vec_{i}")
    insert_time = time.time() - start
    
    query = generate_random_vector(dims)
    start = time.time()
    for _ in range(50):
      db.search(query, 10)
    search_time = (time.time() - start) / 50 * 1000
    
    results['dimensions'].append(dims)
    results['insert_time'].append(insert_time)
    results['search_time'].append(search_time)
    
    print(f"    Insert: {insert_time:.3f}s, Search: {search_time:.2f}ms")
    del db
  
  return results

def benchmark_scalability(base_dims=64, multipliers=[1, 2, 4, 8, 16]):
  print(f"\n{'='*60}")
  print(f"BENCHMARK: Scalability Test (Memory & Performance)")
  print(f"{'='*60}")
  
  results = {
    'total_vectors': [],
    'memory_mb': [],
    'insert_throughput': [],
    'search_latency': []
  }
  
  for mult in multipliers:
    dims = base_dims * mult
    count = max(100, 10000 // mult)
    
    print(f"  Testing {count:,} vectors × {dims} dims = {count * dims:,} total elements")
    
    db = VectorDatabase(dims, VDBMetric.COSINE)
    vectors = [generate_random_vector(dims) for _ in range(count)]
    
    start = time.time()
    for i, vec in enumerate(vectors):
      db.add_vector(vec, f"vec_{i}")
    insert_time = time.time() - start
    throughput = count / insert_time
    
    query = generate_random_vector(dims)
    start = time.time()
    for _ in range(20):
      db.search(query, 5)
    search_latency = (time.time() - start) / 20 * 1000
    
    memory_estimate = (count * dims * 4 + count * 100) / (1024 * 1024)
    
    results['total_vectors'].append(count * dims)
    results['memory_mb'].append(memory_estimate)
    results['insert_throughput'].append(throughput)
    results['search_latency'].append(search_latency)
    
    print(f"    Throughput: {throughput:.0f} vec/s, Latency: {search_latency:.2f}ms")
    del db
  
  return results

def create_comprehensive_plots(all_results):
  print(f"\n{'='*60}")
  print(f"Generating visualization...")
  print(f"{'='*60}")
  
  fig = plt.figure(figsize=(18, 6))
  gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
  
  ax1 = fig.add_subplot(gs[0, 0])
  insertion = all_results['insertion']
  ax1.plot(insertion['counts'], insertion['times'], 'o-', linewidth=3, markersize=10, color='#2E86AB', label='Time')
  ax1.set_xlabel('Number of Vectors', fontsize=12, fontweight='bold')
  ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
  ax1.set_title('Insertion Performance', fontsize=14, fontweight='bold', pad=15)
  ax1.grid(True, alpha=0.3, linestyle='--')
  ax1.set_xscale('log')
  ax1_twin = ax1.twinx()
  ax1_twin.plot(insertion['counts'], insertion['throughput'], 's--', linewidth=2, markersize=8, color='#A23B72', label='Throughput', alpha=0.7)
  ax1_twin.set_ylabel('Throughput (vectors/sec)', fontsize=12, fontweight='bold', color='#A23B72')
  ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
  ax1.legend(loc='upper left')
  ax1_twin.legend(loc='upper right')
  
  ax2 = fig.add_subplot(gs[0, 1])
  dims = all_results['dimensionality']
  ax2.plot(dims['dimensions'], dims['insert_time'], '^-', linewidth=3, markersize=10, color='#F18F01', label='Insert')
  ax2_twin = ax2.twinx()
  ax2_twin.plot(dims['dimensions'], dims['search_time'], 'v-', linewidth=3, markersize=10, color='#C73E1D', label='Search')
  ax2.set_xlabel('Dimensions', fontsize=12, fontweight='bold')
  ax2.set_ylabel('Insert Time (s)', fontsize=12, fontweight='bold', color='#F18F01')
  ax2_twin.set_ylabel('Search Time (ms)', fontsize=12, fontweight='bold', color='#C73E1D')
  ax2.set_title('Impact of Dimensionality', fontsize=14, fontweight='bold', pad=15)
  ax2.grid(True, alpha=0.3, linestyle='--')
  ax2.tick_params(axis='y', labelcolor='#F18F01')
  ax2_twin.tick_params(axis='y', labelcolor='#C73E1D')
  ax2.legend(loc='upper left')
  ax2_twin.legend(loc='upper right')
  
  ax3 = fig.add_subplot(gs[0, 2])
  scale = all_results['scalability']
  ax3.scatter(scale['total_vectors'], scale['memory_mb'], s=300, c=scale['search_latency'], 
             cmap='viridis', alpha=0.8, edgecolors='black', linewidth=2)
  for i, txt in enumerate(scale['total_vectors']):
    ax3.annotate(f"{scale['memory_mb'][i]:.1f}MB\n{scale['insert_throughput'][i]:.0f}v/s", 
                (scale['total_vectors'][i], scale['memory_mb'][i]),
                textcoords="offset points", xytext=(0,12), ha='center', fontsize=9, fontweight='bold')
  ax3.set_xlabel('Total Vector Elements', fontsize=12, fontweight='bold')
  ax3.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
  ax3.set_title('Scalability: Memory vs Size', fontsize=14, fontweight='bold', pad=15)
  ax3.set_xscale('log')
  ax3.grid(True, alpha=0.3, linestyle='--')
  cbar = plt.colorbar(ax3.collections[0], ax=ax3)
  cbar.set_label('Search Latency (ms)', fontsize=11, fontweight='bold')
  
  plt.suptitle('vdb vector database - performance benchmarks', 
              fontsize=16, fontweight='bold', y=1.02)
  
  output_file = 'benchmark.png'
  plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
  print(f"  Saved plot to: {output_file}")
  
  return output_file

def main():
  print("\n" + "="*60)
  print("  vdb stress tests")
  print("="*60)
  
  all_results = {}
  
  all_results['insertion'] = benchmark_insertion(
    dims=128,
    counts=[100, 500, 1000, 5000, 10000, 25000]
  )
  
  all_results['dimensionality'] = benchmark_dimensionality(
    dimensions=[32, 64, 128, 256, 512, 1024],
    fixed_count=1000
  )
  
  all_results['scalability'] = benchmark_scalability(
    base_dims=64,
    multipliers=[1, 2, 4, 8, 16]
  )
  
  plot_file = create_comprehensive_plots(all_results)
  
  print(f"\n{'='*60}")
  print(f"  completed...")
  print(f"{'='*60}\n")
  
  return plot_file

if __name__ == "__main__":
  main()