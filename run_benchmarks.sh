#!/bin/bash

# Function to run benchmark with specific configuration
run_benchmark() {
    local nodes=$1
    local procs=$2
    local wait_time=$3
    
    echo "Running benchmark with $nodes nodes and $procs processes per node"
    
    # Start master node
    torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=0 --master-addr="vm-master" --master-port=29500 distributed_train.py &
    
    # If using multiple nodes, start worker nodes
    if [ $nodes -gt 1 ]; then
        ssh vm-worker1 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=1 --master-addr='vm-master' --master-port=29500 distributed_train.py" &
        
        if [ $nodes -eq 3 ]; then
            ssh vm-worker2 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=2 --master-addr='vm-master' --master-port=29500 distributed_train.py" &
        fi
    fi
    
    # Wait for all processes to complete
    wait
    
    # Wait between runs to ensure clean state
    sleep $wait_time
}

# Create results directory
mkdir -p benchmark_results

# Test configurations
echo "Starting benchmarks..."

# Single node (master only), 1 process
echo "Testing single node (master only) with 1 process"
run_benchmark 1 1 30

# Single node (master only), 2 processes
echo "Testing single node (master only) with 2 processes"
run_benchmark 1 2 30

# Two nodes, 1 process each
echo "Testing two nodes with 1 process each"
run_benchmark 2 1 30

# Two nodes, 2 processes each
echo "Testing two nodes with 2 processes each"
run_benchmark 2 2 30

# Three nodes, 1 process each
echo "Testing three nodes with 1 process each"
run_benchmark 3 1 30

# Three nodes, 2 processes each
echo "Testing three nodes with 2 processes each"
run_benchmark 3 2 30

echo "All benchmarks completed!"
echo "Results are saved in the benchmark_results directory"

# Display summary of results
echo "Summary of results:"
python3 << END
import json
from pathlib import Path
import glob

results = []
for file in glob.glob('benchmark_results/benchmark_*.json'):
    with open(file) as f:
        data = json.load(f)
        results.append({
            'config': file.split('_')[1],
            'total_time': data['total_time'],
            'avg_epoch_time': data['avg_epoch_time'],
            'final_loss': data['epoch_losses'][-1]
        })

print("\nPerformance comparison:")
print(f"{'Configuration':<20} {'Total Time':<15} {'Avg Epoch Time':<15} {'Final Loss':<10}")
print("-" * 60)
for r in sorted(results, key=lambda x: x['total_time']):
    print(f"{r['config']:<20} {r['total_time']:<15.2f} {r['avg_epoch_time']:<15.2f} {r['final_loss']:<10.4f}")
END
