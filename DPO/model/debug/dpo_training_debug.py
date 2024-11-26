#Code origin
#
#Author: Alexander Valentini (alexander.valentini@epfl.ch)
#Made on: 26/5-2024
#Made for: Project report for course CS-552. submission on 2/6-2024

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_from_disk
import os
from trl import DPOTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from pathlib import Path
import wandb
torch.cuda.empty_cache()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

#We use DataCollatorWithPadding per default, which dynamically pads the inputs to the maximum length in the batch.
project_dir = os.path.dirname(os.path.abspath(os.getcwd()))

model_name = 'stabilityai/stablelm-zephyr-3b'
#safetensors_path = os.path.join(project_dir, 'model/sft_stablelm_zephyr_3b/29.05.2024_v4') 
safetensors_path = os.path.join(project_dir, 'sft_stablelm_zephyr_3b') 

previous_checkpoint_path = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name='stabilityai/stablelm-zephyr-3b'
tokenizer = AutoTokenizer.from_pretrained(safetensors_path)
model = AutoModelForCausalLM.from_pretrained(safetensors_path, attn_implementation="sdpa", use_safetensors = True, quantization_config=bnb_config)
#model = AutoModelForCausalLM.from_pretrained(safetensors_path, attn_implementation="sdpa", use_safetensors = True, quantization_config=bnb_config)
#model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", quantization_config=bnb_config)

data_dir = os.path.join(project_dir, 'data/dpo_preference_data/processed/EPFL_preference_data_below_1024')
preference_data = load_from_disk(data_dir)
train_dataset = preference_data['train']
eval_dataset = preference_data['eval']

peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
wandb_path = os.path.join(project_dir, 'wandb_key_alex.txt')
with open(wandb_path, "r") as f:
        wandb_key = f.read().strip()
        
wandb.login(key=wandb_key)
wandb.init(
    project="MNLP_dpo",
)
#Select 500 data points for training and 100 data points for evaluation
train_dataset = train_dataset.select(range(30))
eval_dataset = eval_dataset.select(range(5))


args = TrainingArguments(
    output_dir="checkpoints/dpo_model_test",
    num_train_epochs=5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=5e-5,
#    learning_rate = 5e-4,
    max_grad_norm=0.3,
#    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy = "steps",
    save_steps=100,
    save_total_limit=4,
    evaluation_strategy="steps",
    eval_steps=1,
    bf16=True,
    tf32=True,
    push_to_hub=False,
    report_to="wandb",
)

dpo_args = {
    "beta": 0.1,
    "loss_type": "sigmoid"
}

prompt_max_length = 3000
max_seq_length = 3000

trainer = DPOTrainer(
    model,
    ref_model=None,
    peft_config=peft_config,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=prompt_max_length,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"],
)

# Find the latest checkpoint
if previous_checkpoint_path is not None:
    trainer.train(resume_from_checkpoint=previous_checkpoint_path)
else:
    trainer.train()

# Save model at the end of training
trainer.save_model()

# Merge LoRA adapter with the base model
#peft_model = AutoPeftModelForCausalLM.from_pretrained(
#    args.output_dir,
#    torch_dtype=torch.float16,
#    low_cpu_mem_usage=True,
#)

# Merge LoRA and base model and save
#merged_model = peft_model.merge_and_unload()
#merged_model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="2GB")
#tokenizer.save_pretrained(args.output_dir)