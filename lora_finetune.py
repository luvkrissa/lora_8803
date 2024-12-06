from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
import torch
from transformers.utils import logging

logging.set_verbosity_info()

# Step 1: Define paths and model details
json_file = "Generated_Instruction_Dataset.json"  # Path to your JSON dataset
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your base model

# Step 2: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Load the model efficiently
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Use device IDs for mapping
device_map = infer_auto_device_map(
    model,
    max_memory={0: "24GiB", "cpu": "12GiB"},  # Use integers for GPU devices
)

# Load the model with the device map
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch.float16,
)

# Step 4: Set up LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type
    r=8,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    target_modules=["q_proj", "v_proj"],  # Target modules
)

model = get_peft_model(model, lora_config)

# Step 5: Load the JSON dataset
dataset = load_dataset("json", data_files=json_file)

# Inspect dataset
print("Loaded dataset structure:", dataset)

# Step 6: Preprocess the dataset
def preprocess_function(examples):
    # Create a concatenated input format
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    full_input = [
        f"Instruction: {instr}\nInput: {inp}\nOutput:" for instr, inp in zip(instructions, inputs)
    ]
    tokenized = tokenizer(
        full_input,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    with_labels = tokenizer(
        outputs,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    # Prepare model inputs and labels
    tokenized["labels"] = with_labels["input_ids"]
    return tokenized

# Apply preprocessing
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "input", "output"])

# Step 7: Split the dataset
train_dataset = processed_dataset["train"]
if "validation" in processed_dataset:
    val_dataset = processed_dataset["validation"]
else:
    train_dataset, val_dataset = train_dataset.train_test_split(test_size=0.1).values()

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_bias_classification",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-4,
    per_device_train_batch_size=1,  # Adjust based on GPU memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=200,
    fp16=True,  # Mixed precision training
    gradient_accumulation_steps=16,
    report_to="tensorboard",
)

# Step 9: Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Step 10: Train the model
trainer.train()

# Step 11: Save the fine-tuned model
trainer.save_model("./lora_bias_classification")
