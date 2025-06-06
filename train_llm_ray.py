import ray
from ray import train
from ray.train.torch import TorchTrainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Connect to the Ray cluster
ray.init(address="auto")

def train_func(config):
    # Initialize model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Preprocess the dataset with labels
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # Reduced for memory efficiency
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()  # Add labels for loss computation
        return tokenized
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    
    # Use a small subset for testing
    train_dataset = tokenized_dataset["train"].select(range(1000))  # Limit to 1000 samples
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_dir="./logs",
        report_to="none",
        gradient_checkpointing=True,
    )
    
    # Initialize and run trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

# Configure and run the TorchTrainer with adjusted resources
trainer = TorchTrainer(
    train_func,
    scaling_config=train.ScalingConfig(
        num_workers=4,
        use_gpu=False,
        resources_per_worker={"CPU": 8}  # Reduced to fit available resources
    ),
    run_config=train.RunConfig(storage_path="/home/azureuser/ray_results"),
)

result = trainer.fit()