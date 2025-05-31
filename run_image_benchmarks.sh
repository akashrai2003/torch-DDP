#!/bin/bash

# Function to run benchmark with specific configuration
# Function to cleanup processes and ports
cleanup() {
    # Kill any remaining torchrun processes
    pkill -f torchrun
    ssh vm-worker1 "pkill -f torchrun" || true
    ssh vm-worker2 "pkill -f torchrun" || true
    
    # Wait for ports to be released
    sleep 10
}

# Function to run benchmark with specific configuration
run_benchmark() {
    local nodes=$1
    local procs=$2
    local wait_time=$3
    local script=$4
    
    # Cleanup before starting new run
    cleanup
    
    # Use different port for each run to avoid conflicts
    local port=$((29500 + RANDOM % 1000))
    
    echo "Running benchmark with $nodes nodes and $procs processes per node using $script on port $port"
    
    # Start master node
    torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=0 \
        --master-addr="vm-master" --master-port=$port $script &
    
    # If using multiple nodes, start worker nodes
    if [ $nodes -gt 1 ]; then
        ssh vm-worker1 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && \
            torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=1 \
            --master-addr='vm-master' --master-port=$port $script" &
        
        if [ $nodes -eq 3 ]; then
            ssh vm-worker2 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && \
                torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=2 \
                --master-addr='vm-master' --master-port=$port $script" &
        fi
    fi
    
    # Wait for all processes to complete
    wait
    
    # Cleanup after run
    cleanup
    
    # Additional wait between runs
    sleep $wait_time
}

# Create results directory
mkdir -p benchmark_results

# Test configurations for ResNet50 on CIFAR-100
echo "Starting ResNet50 benchmarks..."

SCRIPT="distributed_image_train.py"

# Single node (master only), 1 process
# echo "Testing single node (master only) with 1 process"
# run_benchmark 1 1 30 $SCRIPT

# # Single node (master only), 2 processes
# echo "Testing single node (master only) with 2 processes"
# run_benchmark 1 2 30 $SCRIPT

# Two nodes, 1 process each
# echo "Testing two nodes with 1 process each"
# run_benchmark 2 1 30 $SCRIPT

# Two nodes, 2 processes each
echo "Testing two nodes with 2 processes each"
run_benchmark 2 2 30 $SCRIPT

# Three nodes, 1 process each
echo "Testing three nodes with 1 process each"
run_benchmark 3 1 30 $SCRIPT

# Three nodes, 2 processes each
echo "Testing three nodes with 2 processes each"
run_benchmark 3 2 30 $SCRIPT

echo "All benchmarks completed!"
echo "Results are saved in the benchmark_results directory"

# Display summary of results
echo "Summary of results:"
python3 << END
import json
from pathlib import Path
import glob

def load_metrics(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return {
        'config': file_path.split('_')[1:3],  # nodes and procs info
        'total_time': data['total_time'],
        'avg_epoch_time': data['avg_epoch_time'],
        'best_accuracy': data.get('best_accuracy', 0),
        'final_loss': data['epoch_losses'][-1]
    }

print("\nResNet50 Performance Comparison:")
print(f"{'Configuration':<20} {'Total Time':<12} {'Epoch Time':<12} {'Best Acc':<10} {'Final Loss':<10}")
print("-" * 65)

# Get all ResNet benchmark files
for file in sorted(glob.glob('benchmark_results/benchmark_resnet_*.json')):
    metrics = load_metrics(file)
    config = '_'.join(metrics['config'])
    print(f"{config:<20} {metrics['total_time']:<12.2f} "
          f"{metrics['avg_epoch_time']:<12.2f} {metrics['best_accuracy']:<10.2f} "
          f"{metrics['final_loss']:<10.4f}")
END
