import ray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import time
import os

# Global dictionary to store actor handles
actor_handles = {}

# Connect to the Ray cluster with a specific namespace
ray.init(address="auto", ignore_reinit_error=True, namespace="model_parallel_training")

# Define a model shard class
class ModelShard(nn.Module):
    def __init__(self, model_part, is_first_shard=False, is_last_shard=False):
        super().__init__()
        if model_part is None:
            raise ValueError("model_part cannot be None")
        self.model_part = model_part
        self.is_first_shard = is_first_shard
        self.is_last_shard = is_last_shard

    def forward(self, inputs):
        if self.is_first_shard:
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.float32)
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            x = self.model_part[0](input_ids) + self.model_part[1](position_ids)
            x = self.model_part[2](x)  # Dropout
            for layer in self.model_part[3:]:
                x = layer(x, attention_mask=attention_mask)[0]
            return x
        elif self.is_last_shard:
            hidden_states = inputs["hidden_states"]
            labels = inputs["labels"]
            for layer in self.model_part[:-2]:
                hidden_states = layer(hidden_states)[0]
            hidden_states = self.model_part[-2](hidden_states)  # Layer norm
            logits = self.model_part[-1](hidden_states)  # LM head
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            hidden_states = inputs["hidden_states"]
            for layer in self.model_part:
                hidden_states = layer(hidden_states)[0]
            return hidden_states

# Remote actor for each model shard
@ray.remote(num_cpus=4)
class ModelParallelWorker:
    def __init__(self, rank, model_part, device, is_first_shard=False, is_last_shard=False):
        self.rank = rank
        self.device = device
        print(f"Initializing worker rank {rank} (PID: {os.getpid()}) with model_part type: {type(model_part)}, len: {len(model_part)}")
        self.model = ModelShard(model_part, is_first_shard, is_last_shard).to(device)
        self.model.train()

    def forward(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(inputs)
        return outputs

    def backward(self, grad):
        if grad is not None:
            self.model.zero_grad()
            grad.backward()

    def get_parameters(self):
        return list(self.model.parameters())

    def is_alive(self):
        return True  # Simple liveness check

# Remote training task
@ray.remote(num_cpus=4)
def train_model_parallel(config):
    rank = config["rank"]
    world_size = config["world_size"]
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {k: v.squeeze(1) if v.dim() > 2 else v for k, v in tokenized.items()}
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    
    tokenized_dataset.set_format("torch")
    train_dataset = tokenized_dataset["train"]

    def collate_fn(batch):
        return {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.config.pad_token_id = tokenizer.eos_token_id

    # Split the model
    device = torch.device("cpu")
    if rank == 0:
        model_part = nn.ModuleList([
            model.transformer.wte.to(device),
            model.transformer.wpe.to(device),
            model.transformer.drop.to(device),
            model.transformer.h[0].to(device),
            model.transformer.h[1].to(device),
            model.transformer.h[2].to(device)
        ])
        is_first_shard = True
        is_last_shard = False
    else:
        model_part = nn.ModuleList([
            model.transformer.h[3].to(device),
            model.transformer.h[4].to(device),
            model.transformer.h[5].to(device),
            model.transformer.ln_f.to(device),
            model.lm_head.to(device)
        ])
        is_first_shard = False
        is_last_shard = True
    print(f"Rank {rank} model_part initialized with {len(model_part)} components")

    # Create worker actor with unique name
    worker = ModelParallelWorker.options(num_cpus=4, name=f"train_worker_{rank}").remote(
        rank, model_part, device, is_first_shard, is_last_shard
    )
    
    # Store actor handle globally
    global actor_handles
    actor_handles[f"train_worker_{rank}"] = worker
    print(f"Stored actor handle for train_worker_{rank}")

    # Verify actor creation
    try:
        ray.get_actor(f"train_worker_{rank}", namespace="model_parallel_training")
        print(f"Actor train_worker_{rank} created successfully")
    except ValueError as e:
        print(f"Failed to create actor train_worker_{rank}: {e}")
        raise

    # Optimizer
    params = ray.get(worker.get_parameters.remote())
    optimizer = torch.optim.Adam(params, lr=5e-5)

    # Training loop
    for epoch in range(1):
        for batch_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            if rank == 0:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"Rank {rank} attempting to get actor train_worker_1 (attempt {attempt + 1})")
                        # Check if worker_1 is alive
                        worker_1 = ray.get_actor(f"train_worker_{1}", namespace="model_parallel_training")
                        if not ray.get(worker_1.is_alive.remote()):
                            raise ValueError("train_worker_1 is not alive")
                        hidden_states = ray.get(worker.forward.remote(inputs))
                        output = ray.get(worker_1.forward.remote(
                            {"hidden_states": hidden_states, "labels": inputs["labels"]}
                        ))
                        loss = output["loss"]
                        break
                    except (RuntimeError, ValueError) as e:
                        print(f"Rank {rank} forward attempt {attempt + 1} failed: {e}")
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(1)  # Wait before retrying
            else:
                continue
            
            # Backward pass
            optimizer.zero_grad()
            if rank == 0:
                try:
                    worker_1 = ray.get_actor(f"train_worker_{1}", namespace="model_parallel_training")
                    ray.get(worker_1.backward.remote(loss))
                    ray.get(worker.backward.remote(None))
                    if batch_idx % 100 == 0:
                        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
                except (RuntimeError, ValueError) as e:
                    print(f"Rank {rank} backward error: {e}")
                    raise
            
            optimizer.step()

# Launch training on each worker
ray.get([
    train_model_parallel.options(num_cpus=4).remote({"rank": rank, "world_size": 2})
    for rank in range(2)
])