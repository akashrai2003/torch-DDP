import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import socket
import time
from datetime import timedelta
import json
from pathlib import Path

def setup():
    # First try NCCL, fallback to GLOO
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Set timeout values
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    timeout = 1800  # 30 minutes timeout
    
    init_method = f"env://"  # Use environment variables for initialization
    
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

def cleanup():
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Error in cleanup: {str(e)}")

def save_metrics(metrics, run_config):
    """Save metrics to a JSON file"""
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{run_config}_{timestamp}.json"
    
    with open(output_dir / filename, "w") as f:
        json.dump(metrics, f, indent=2)
    
    if dist.get_rank() == 0:
        print(f"Saved metrics to {filename}")

def main():
    try:
        start_time = time.time()
        setup()
        setup_time = time.time() - start_time
        
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

        # Select device based on availability
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        print(f"[Rank {rank}] Using device: {device}")

        # Dummy model and data
        model = nn.Linear(10, 1).to(device)
        
        # Wrap model with DDP
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

        dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 10),
            torch.randn(1000, 1)
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            pin_memory=True if torch.cuda.is_available() else False
        )

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Training metrics
        metrics = {
            'setup_time': setup_time,
            'node_info': node_info,
            'epoch_times': [],
            'epoch_losses': [],
            'batch_times': [],
            'batch_losses': []
        }

        total_start_time = time.time()
        
        for epoch in range(100):
            try:
                epoch_start = time.time()
                sampler.set_epoch(epoch)
                running_loss = 0.0
                
                for batch_idx, batch in enumerate(dataloader):
                    batch_start = time.time()
                    
                    inputs, targets = batch
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    batch_time = time.time() - batch_start
                    metrics['batch_times'].append(batch_time)
                    metrics['batch_losses'].append(loss.item())
                    running_loss += loss.item()
                    
                    if batch_idx % 10 == 0 and rank == 0:
                        print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Time: {batch_time:.4f}s")
                
                epoch_time = time.time() - epoch_start
                avg_loss = running_loss / len(dataloader)
                metrics['epoch_times'].append(epoch_time)
                metrics['epoch_losses'].append(avg_loss)
                
                if rank == 0:
                    print(f"[Rank {rank}] Epoch {epoch} done, Avg loss: {avg_loss:.4f}, Time: {epoch_time:.4f}s")
                
                dist.barrier()
                
            except Exception as e:
                print(f"[Rank {rank}] Error in epoch {epoch}: {str(e)}")
                raise e

        total_time = time.time() - total_start_time
        metrics['total_time'] = total_time
        metrics['avg_epoch_time'] = sum(metrics['epoch_times']) / len(metrics['epoch_times'])
        metrics['avg_batch_time'] = sum(metrics['batch_times']) / len(metrics['batch_times'])
        
        # Save metrics if rank 0
        if rank == 0:
            run_config = f"nodes{world_size}_procs{os.getenv('LOCAL_WORLD_SIZE', '1')}"
            save_metrics(metrics, run_config)
            
            print("\nTraining Summary:")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average epoch time: {metrics['avg_epoch_time']:.2f}s")
            print(f"Average batch time: {metrics['avg_batch_time']:.2f}s")
            print(f"Final loss: {metrics['epoch_losses'][-1]:.4f}")

    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Error in main: {str(e)}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
