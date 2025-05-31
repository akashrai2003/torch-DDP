import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import socket
import time
from datetime import timedelta
import json
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights

def setup():
    # First try NCCL, fallback to GLOO
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Set timeout values
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    timeout = 1800
    
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

def cleanup():
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Error in cleanup: {str(e)}")

def save_metrics(metrics, run_config):
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_resnet_{run_config}_{timestamp}.json"
    
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

        # Data augmentation and normalization
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        # Load CIFAR-100 dataset
        if rank == 0:
            print("Loading CIFAR-100 dataset...")
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=True,
            download=True, 
            transform=transform_train
        )
        # Only use 25% of training data
        trainset = torch.utils.data.Subset(trainset, indices=range(0, len(trainset), 8))
        
        valset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=False,
            download=True, 
            transform=transform_val
        )
        # Only use 25% of validation data
        valset = torch.utils.data.Subset(valset, indices=range(0, len(valset), 8))

        # Create distributed samplers
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank)

        # Create dataloaders
        train_loader = DataLoader(
            trainset, 
            batch_size=64,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            valset,
            batch_size=64,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Create model
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the last fully connected layer for CIFAR-100 (100 classes)
        model.fc = nn.Linear(model.fc.in_features, 100)
        model = model.to(device)
        
        # Wrap model with DDP
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # Training metrics
        metrics = {
            'setup_time': setup_time,
            'node_info': node_info,
            'epoch_times': [],
            'epoch_losses': [],
            'epoch_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'batch_times': [],
            'batch_losses': []
        }

        total_start_time = time.time()
        best_acc = 0.0
        
        num_epochs = 2  # Quick test configuration with fewer epochs
        
        for epoch in range(num_epochs):
            try:
                model.train()
                epoch_start = time.time()
                train_sampler.set_epoch(epoch)
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    batch_start = time.time()
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    batch_time = time.time() - batch_start
                    metrics['batch_times'].append(batch_time)
                    metrics['batch_losses'].append(loss.item())
                    running_loss += loss.item()
                    
                    if batch_idx % 20 == 0 and rank == 0:
                        acc = 100. * correct / total
                        print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, "
                              f"Loss: {loss.item():.4f}, Acc: {acc:.2f}%, "
                              f"Time: {batch_time:.4f}s")
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                
                epoch_time = time.time() - epoch_start
                train_acc = 100. * correct / total
                val_acc = 100. * val_correct / val_total
                avg_loss = running_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                metrics['epoch_times'].append(epoch_time)
                metrics['epoch_losses'].append(avg_loss)
                metrics['epoch_accuracies'].append(train_acc)
                metrics['val_losses'].append(avg_val_loss)
                metrics['val_accuracies'].append(val_acc)
                
                if rank == 0:
                    print(f"\n[Rank {rank}] Epoch {epoch} Summary:")
                    print(f"Training Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")
                    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
                    print(f"Time: {epoch_time:.2f}s")
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        print(f"New best accuracy: {best_acc:.2f}%")
                
                scheduler.step()
                dist.barrier()
                
            except Exception as e:
                print(f"[Rank {rank}] Error in epoch {epoch}: {str(e)}")
                raise e

        total_time = time.time() - total_start_time
        metrics['total_time'] = total_time
        metrics['avg_epoch_time'] = sum(metrics['epoch_times']) / len(metrics['epoch_times'])
        metrics['avg_batch_time'] = sum(metrics['batch_times']) / len(metrics['batch_times'])
        metrics['best_accuracy'] = best_acc
        
        if rank == 0:
            run_config = f"nodes{world_size}_procs{os.getenv('LOCAL_WORLD_SIZE', '1')}"
            save_metrics(metrics, run_config)
            
            print("\nTraining Summary:")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average epoch time: {metrics['avg_epoch_time']:.2f}s")
            print(f"Average batch time: {metrics['avg_batch_time']:.2f}s")
            print(f"Best validation accuracy: {best_acc:.2f}%")
            print(f"Final training loss: {metrics['epoch_losses'][-1]:.4f}")

    except Exception as e:
        print(f"[Rank {os.getenv('RANK', 'Unknown')}] Error in main: {str(e)}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
