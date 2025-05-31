#!/bin/bash

# List of worker nodes
WORKERS=("vm-worker1" "vm-worker2")

# Create directory on workers
for worker in "${WORKERS[@]}"; do
    echo "Setting up $worker..."
    ssh $worker "mkdir -p ~/pytorch-distributed"
done

# Copy the necessary files to all workers
for worker in "${WORKERS[@]}"; do
    echo "Copying files to $worker..."
    scp ~/pytorch-distributed/distributed_image_train.py $worker:~/pytorch-distributed/
    scp ~/pytorch-distributed/run_image_benchmarks.sh $worker:~/pytorch-distributed/
done

echo "Files have been copied to all workers"
echo "You can now run ./run_image_benchmarks.sh"
