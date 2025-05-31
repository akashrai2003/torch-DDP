import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from datasets import load_dataset
from torch.optim import AdamW
import socket
import time
from datetime import timedelta
import json
from pathlib import Path

def setup():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
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
    filename = f"benchmark_nlp_{run_config}_{timestamp}.json"
    
    with open(output_dir / filename, "w") as f:
        json.dump(metrics, f, indent=2)
    
    if dist.get_rank() == 0:
        print(f"Saved metrics to {filename}")

def compute_metrics(preds, labels):
    accuracy = (preds == labels).mean()
    return {"accuracy": float(accuracy)}

def main():
    try:
        start_time = time.time()
        setup()
        setup_time = time.time() - start_time
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        
        node_info = {
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'hostname': socket.gethostname()
        }
        
        if rank == 0:
            print(f"Node configuration: {node_info}")

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        
        print(f"[Rank {rank}] Using device: {device}")

        # Load BERT model and tokenizer
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification
        )
        model = model.to(device)

        # Wrap model with DDP
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

        # Load SST-2 dataset (binary sentiment classification)
        if rank == 0:
            print("Loading SST-2 dataset...")
        
        dataset = load_dataset("glue", "sst2")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["sentence"],
                padding=False,
                truncation=True,
                max_length=128,
                return_tensors=None  # Ensure we don't get batched tensors
            )

        # Tokenize datasets
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['idx', 'sentence']  # Keep the label column
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Create distributed samplers
        train_sampler = DistributedSampler(
            tokenized_datasets["train"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            tokenized_datasets["validation"],
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        # Create dataloaders
        train_loader = DataLoader(
            tokenized_datasets["train"],
            batch_size=32,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=32,
            sampler=val_sampler,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Training setup
        num_epochs = 2
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = num_training_steps // 10

        optimizer = AdamW(model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

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
        
        for epoch in range(num_epochs):
            try:
                model.train()
                epoch_start = time.time()
                train_sampler.set_epoch(epoch)
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    batch_start = time.time()
                    
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    if loss is not None:  # Make sure we have a valid loss
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                    else:
                        print(f"[Rank {rank}] Warning: Got None loss for batch {batch_idx}")
                        continue
                    
                    # Calculate accuracy
                    predictions = outputs.logits.argmax(-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
                    
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
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        
                        predictions = outputs.logits.argmax(-1)
                        val_correct += (predictions == batch['labels']).sum().item()
                        val_total += batch['labels'].size(0)
                        val_loss += loss.item()
                
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
