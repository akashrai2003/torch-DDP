#!/bin/bash
# filepath: /home/azureuser/pytorch-distributed/run_model_parallel_benchmarks.sh

# Function to run benchmark with specific configuration
run_model_parallel_benchmark() {
    local nodes=$1
    local procs=$2
    local wait_time=$3
    local port=$4
    
    echo "Running model parallel benchmark with $nodes nodes and $procs processes per node"
    
    # Start master node
    RPC_PORT=$((port+1))
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=0 \
    --master-addr="vm-master" --master-port=$port \
    model_parallel_train.py &
    
    # If using multiple nodes, start worker nodes
    if [ $nodes -gt 1 ]; then
        ssh vm-worker1 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && \
            TORCH_DISTRIBUTED_DEBUG=DETAIL \
            torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=1 \
            --master-addr='vm-master' --master-port=$port \
            model_parallel_train.py" &
        
        if [ $nodes -gt 2 ]; then
            ssh vm-worker2 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && \
                TORCH_DISTRIBUTED_DEBUG=DETAIL \
                torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=2 \
                --master-addr='vm-master' --master-port=$port \
                model_parallel_train.py" &
        fi
    fi
    
    # Wait for all processes to complete
    wait
    
    # Wait between runs to ensure clean state
    sleep $wait_time
}

# Function to run data parallel benchmark for comparison
run_data_parallel_benchmark() {
    local nodes=$1
    local procs=$2
    local wait_time=$3
    local port=$4
    
    echo "Running data parallel benchmark with $nodes nodes and $procs processes per node"
    
    # Start master node
    torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=0 \
    --master-addr="vm-master" --master-port=$port \
    distributed_train.py &
    
    # If using multiple nodes, start worker nodes
    if [ $nodes -gt 1 ]; then
        ssh vm-worker1 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && \
            torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=1 \
            --master-addr='vm-master' --master-port=$port \
            distributed_train.py" &
        
        if [ $nodes -gt 2 ]; then
            ssh vm-worker2 "cd ~/pytorch-distributed && source ~/pytorch_dist/bin/activate && \
                torchrun --nproc-per-node=$procs --nnodes=$nodes --node-rank=2 \
                --master-addr='vm-master' --master-port=$port \
                distributed_train.py" &
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

# Run data parallel benchmarks for comparison
echo "Running DATA PARALLEL benchmarks for baseline comparison..."

# Single node (master only), 1 process
echo "Testing single node (master only) with 1 process (DATA PARALLEL)"
run_data_parallel_benchmark 1 1 30 29500

# Single node (master only), 2 processes
echo "Testing single node (master only) with 2 processes (DATA PARALLEL)"
run_data_parallel_benchmark 1 2 30 29500

# Two nodes, 1 process each
echo "Testing two nodes with 1 process each (DATA PARALLEL)"
run_data_parallel_benchmark 2 1 30 29500

# Two nodes, 2 processes each
echo "Testing two nodes with 2 processes each (DATA PARALLEL)"
run_data_parallel_benchmark 2 2 30 29500

# Run model parallel benchmarks
echo "Running MODEL PARALLEL benchmarks..."

# Single node (master only), 1 process (will allocate model parts to additional workers)
echo "Testing single node (master only) with 1 process (MODEL PARALLEL)"
run_model_parallel_benchmark 1 1 30 29600

# Two nodes, 1 process each
echo "Testing two nodes with 1 process each (MODEL PARALLEL)"
run_model_parallel_benchmark 2 1 30 29600

# Three nodes, 1 process each (ideal for model parallelism with 3 parts)
echo "Testing three nodes with 1 process each (MODEL PARALLEL)"
run_model_parallel_benchmark 3 1 30 29600

echo "All benchmarks completed."

# Run script to compare results
echo "Generating comparison report..."
python3 compare_parallelism_results.py

echo "Done!"
