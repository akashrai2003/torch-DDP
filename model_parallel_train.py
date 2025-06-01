#!/usr/bin/env python3
# filepath: /home/azureuser/pytorch-distributed/model_parallel_train.py
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import socket
import time
from datetime import timedelta
import json
from pathlib import Path
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote, rpc_sync, rpc_async
import functools
import numpy as np

# Set environment variables for debugging
os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def setup_process_group():
    """Setup process group for distributed data parallel."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    timeout = 1800  # 30 minutes timeout
    
    init_method = f"env://"
    
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timedelta(seconds=timeout)
        )
        print(f"[Rank {dist.get_rank()}] Initialized process group with {backend} backend")
        torch.manual_seed(0)
    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Failed to initialize process group: {str(e)}")
        sys.exit(1)

def setup_rpc(rank, world_size, master_addr, master_port):
    """Setup RPC for model parallelism."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=60,
        init_method=f"tcp://{master_addr}:{master_port}"
    )
    
    worker_name = f"worker{rank}"
    rpc.init_rpc(
        worker_name,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )
    print(f"[RPC {worker_name}] Initialized RPC")

def cleanup():
    """Cleanup process group."""
    try:
        dist.destroy_process_group()
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Destroyed process group")
    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Error in cleanup: {str(e)}")
    
    # Shutdown RPC if initialized
    try:
        if rpc.is_initialized():
            rpc.shutdown()
            print(f"[Rank {os.getenv('RANK', 'Unknown')}] Shutdown RPC")
    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Error in RPC cleanup: {str(e)}")

def save_metrics(metrics, run_config):
    """Save metrics to a JSON file."""
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_model_parallel_{run_config}_{timestamp}.json"
    
    with open(output_dir / filename, "w") as f:
        json.dump(metrics, f, indent=2)
    
    if dist.get_rank() == 0:
        print(f"Saved metrics to {filename}")


class ModelParallelResNet50(nn.Module):
    """Split ResNet50 into multiple stages for model parallelism."""
    def __init__(self, num_classes=10, split_size=4, device=None):
        super(ModelParallelResNet50, self).__init__()
        # Load pre-trained ResNet50 but don't move to device yet
        import torchvision.models as models
        self.num_classes = num_classes
        self.split_size = split_size
        self.device = device
        
        # Create a ResNet50 model with random weights
        self.resnet = models.resnet50(weights=None)
        
        # Modify final fully connected layer for custom number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # Split model into sequential stages for model parallelism
        self.stage1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        )
        
        # Split the 4 ResNet layer blocks into roughly equal chunks for model parallelism
        resnet_layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        self.stages = []
        for i in range(split_size):
            if i < len(resnet_layers):
                self.stages.append(resnet_layers[i])
            else:
                # For the last stage, add the final layers
                if i == split_size - 1:
                    self.stages.append(nn.Sequential(
                        self.resnet.avgpool,
                        nn.Flatten(),
                        self.resnet.fc
                    ))
                else:
                    # Create dummy layers for extra stages if split_size > 5
                    self.stages.append(nn.Identity())
        
        # Move stage1 to current device if specified
        if device:
            self.stage1 = self.stage1.to(device)
        
        # No need to move other stages here - they'll be placed on worker devices

    def forward(self, x):
        # We only execute the first stage on the current device
        x = self.stage1(x)
        return x


class RemoteModule(nn.Module):
    """Wrapper for modules that will run on remote workers."""
    def __init__(self, module, rank):
        super(RemoteModule, self).__init__()
        self.module = module
        self.rank = rank
    
    def forward(self, x):
        return self.module(x)


class DistributedModelParallel(nn.Module):
    """Implements model parallelism across multiple machines/processes."""
    def __init__(self, model, device_ids, output_device=None):
        super(DistributedModelParallel, self).__init__()
        self.model = model
        self.device_ids = device_ids
        self.output_device = output_device if output_device is not None else device_ids[0]
        self.worker_rrefs = []
        
        # Create remote module references for each stage
        for i, stage in enumerate(model.stages):
            if i < len(device_ids):
                # RPC reference to remote worker
                worker_rref = rpc.remote(
                    f"worker{device_ids[i]}",
                    RemoteModule,
                    args=(stage, device_ids[i])
                )
                self.worker_rrefs.append(worker_rref)
    
    def forward(self, x):
        # Execute first stage on current device (already in the model)
        out = self.model(x)
        
        # Pass output through each remote stage
        for worker_rref in self.worker_rrefs:
            out = rpc_sync(worker_rref.owner(), lambda worker, x: worker.forward(x), 
                          args=(worker_rref, out))
        
        return out


def create_dummy_data(batch_size, input_shape, num_classes, num_batches=100):
    """Create synthetic data for model parallel training."""
    dataset = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, *input_shape)
        targets = torch.randint(0, num_classes, (batch_size,))
        dataset.append((inputs, targets))
    return dataset


def main():
    try:
        start_time = time.time()
        
        # Initialize process group for multi-node communication
        setup_process_group()
        
        # Get process information
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        
        # Collect node information
        node_info = {
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'hostname': socket.gethostname()
        }
        
        if rank == 0:
            print(f"Node configuration: {node_info}")
        
        # Initialize RPC for model parallelism
        master_addr = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '29501')  # Use a different port than DDP
        
        # Initialize RPC framework
        setup_rpc(rank, world_size, master_addr, master_port)
        
        # Select device based on availability
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        print(f"[Rank {rank}] Using device: {device}")
        
        # Set up model parallel parameters - only for rank 0 (master process)
        if rank == 0:
            # Create model with parallel stages
            # Number of splits should be based on number of workers
            split_size = min(4, world_size)  # Use at most 4 splits to avoid too small partitions
            
            # For demonstration, using synthetic image data (3 channels, 224x224 images)
            batch_size = 32
            num_classes = 10
            input_shape = (3, 224, 224)
            
            model = ModelParallelResNet50(
                num_classes=num_classes,
                split_size=split_size,
                device=device
            )
            
            # Device mapping strategy - distribute stages across workers
            device_ids = list(range(1, split_size))  # Skip device 0 as it's used for stage1
            
            # Wrap model in DistributedModelParallel
            model = DistributedModelParallel(model, device_ids)
            
            # Create synthetic dataset for training
            dataset = create_dummy_data(batch_size, input_shape, num_classes)
            
            # Create optimizer - only for parameters on this rank
            # Note: Each worker will need to manage its own parameters
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            # Loss function
            criterion = nn.CrossEntropyLoss()
            
            # Training metrics
            metrics = {
                'setup_time': time.time() - start_time,
                'node_info': node_info,
                'epoch_times': [],
                'epoch_losses': [],
                'batch_times': [],
                'batch_losses': [],
                'forward_times': [],
                'backward_times': [],
                'optimizer_step_times': [],
                'model_parallel_config': {
                    'split_size': split_size,
                    'device_ids': device_ids,
                }
            }
            
            total_start_time = time.time()
            
            # Training loop
            num_epochs = 5
            for epoch in range(num_epochs):
                try:
                    epoch_start = time.time()
                    running_loss = 0.0
                    
                    for batch_idx, (inputs, targets) in enumerate(dataset):
                        batch_start = time.time()
                        
                        # Move data to device
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        # Zero gradients
                        optimizer.zero_grad()
                        
                        # Forward pass - measure time
                        forward_start = time.time()
                        outputs = model(inputs)
                        forward_time = time.time() - forward_start
                        metrics['forward_times'].append(forward_time)
                        
                        # Calculate loss
                        loss = criterion(outputs, targets)
                        
                        # Backward pass - measure time
                        backward_start = time.time()
                        loss.backward()
                        backward_time = time.time() - backward_start
                        metrics['backward_times'].append(backward_time)
                        
                        # Optimizer step - measure time
                        optimizer_start = time.time()
                        optimizer.step()
                        optimizer_step_time = time.time() - optimizer_start
                        metrics['optimizer_step_times'].append(optimizer_step_time)
                        
                        # Record metrics
                        batch_time = time.time() - batch_start
                        metrics['batch_times'].append(batch_time)
                        metrics['batch_losses'].append(loss.item())
                        running_loss += loss.item()
                        
                        if batch_idx % 10 == 0:
                            print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, "
                                  f"Loss: {loss.item():.4f}, "
                                  f"Batch Time: {batch_time:.4f}s, "
                                  f"Forward: {forward_time:.4f}s, "
                                  f"Backward: {backward_time:.4f}s, "
                                  f"Optimizer: {optimizer_step_time:.4f}s")
                    
                    # Calculate epoch metrics
                    epoch_time = time.time() - epoch_start
                    avg_loss = running_loss / len(dataset)
                    metrics['epoch_times'].append(epoch_time)
                    metrics['epoch_losses'].append(avg_loss)
                    
                    print(f"[Rank {rank}] Epoch {epoch} done, "
                          f"Avg loss: {avg_loss:.4f}, "
                          f"Time: {epoch_time:.4f}s")
                
                except Exception as e:
                    print(f"[Rank {rank}] Error in epoch {epoch}: {str(e)}")
                    raise e
            
            # Calculate total time and averages
            total_time = time.time() - total_start_time
            metrics['total_time'] = total_time
            metrics['avg_epoch_time'] = sum(metrics['epoch_times']) / len(metrics['epoch_times'])
            metrics['avg_batch_time'] = sum(metrics['batch_times']) / len(metrics['batch_times'])
            metrics['avg_forward_time'] = sum(metrics['forward_times']) / len(metrics['forward_times'])
            metrics['avg_backward_time'] = sum(metrics['backward_times']) / len(metrics['backward_times'])
            metrics['avg_optimizer_step_time'] = sum(metrics['optimizer_step_times']) / len(metrics['optimizer_step_times'])
            
            # Compare with data parallelism
            metrics['comparison'] = {
                'model_parallel': {
                    'total_time': total_time,
                    'avg_epoch_time': metrics['avg_epoch_time'],
                    'avg_batch_time': metrics['avg_batch_time'],
                    'final_loss': metrics['epoch_losses'][-1],
                    'split_size': split_size,
                },
                'data_parallel': {
                    'note': "Data from most recent benchmark run for comparison",
                    'avg_epoch_time': None,  # To be filled by benchmark script
                    'avg_batch_time': None,  # To be filled by benchmark script
                }
            }
            
            # Save metrics
            run_config = f"nodes{world_size}_split{split_size}"
            save_metrics(metrics, run_config)
            
            print("\nModel Parallel Training Summary:")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average epoch time: {metrics['avg_epoch_time']:.2f}s")
            print(f"Average batch time: {metrics['avg_batch_time']:.2f}s")
            print(f"Average forward time: {metrics['avg_forward_time']:.2f}s")
            print(f"Average backward time: {metrics['avg_backward_time']:.2f}s")
            print(f"Average optimizer step time: {metrics['avg_optimizer_step_time']:.2f}s")
            print(f"Final loss: {metrics['epoch_losses'][-1]:.4f}")
            print(f"Model was split into {split_size} parts across {world_size} workers")

        else:
            # Worker processes just wait for RPC calls
            print(f"[Rank {rank}] Worker ready to receive model parts")
            # Workers will be handling remote execution of model parts, they don't need to do anything here
        
        # Wait for all processes to finish
        if dist.is_initialized():
            dist.barrier()
        
        # Keep RPC alive until all workers are done
        if rpc.is_initialized():
            # This is necessary to allow the script to continue running on workers
            # while they wait for RPC calls from the master
            rpc.shutdown()

    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Error in main: {str(e)}")
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
