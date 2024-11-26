#Code origin
#
#Author: Alexander Valentini (alexander.valentini@epfl.ch)
#Made on: 26/5-2024
#Made for: Project report for course CS-552. submission on 2/6-2024


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_from_disk, load_dataset
import os
from trl import DPOTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from pathlib import Path
import wandb
torch.cuda.empty_cache()



#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    llm_int8_threshold=6.0,
#    llm_int8_has_fp16_weight=False,
#    bnb_4bit_compute_dtype=torch.bfloat16,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type="nf4",
#)

#We use DataCollatorWithPadding per default, which dynamically pads the inputs to the maximum length in the batch.
project_dir = os.path.dirname(os.path.abspath(os.getcwd()))

model_name = 'AlexVal/sft-model'
#safetensors_path = os.path.join(project_dir, 'model/sft_stablelm_zephyr_3b/29.05.2024_v4') 
#safetensors_path = os.path.join(project_dir, 'model/sft_stablelm_zephyr_3b/Full_model_training') 

previous_checkpoint_path = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name='stabilityai/stablelm-zephyr-3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa")

#data_dir = os.path.join(project_dir, 'data/dpo_preference_data/processed/EPFL_preference_data_below_1024')
#data_dir = os.path.join(project_dir, 'data/dpo_preference_data/processed/EPFL_preference_data_99_percentile_filtered')

train_data_path = 'data/dpo_preference_data/processed/train_preference_epfl_data.jsonl'
vali_data_path = 'data/dpo_preference_data/processed/eval_preference_epfl_data.jsonl'

dataset=load_dataset('json', data_files={"train":train_data_path, "validation":vali_data_path})

train_dataset = load_dataset('json',data_files=train_data_path)
valid_dataset = load_dataset('json',data_files=vali_data_path)

#preference_data = load_from_disk(data_dir)

#train_dataset = preference_data['train']
#eval_dataset = preference_data['eval']

peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=256,
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

#new_pad_token = "<|pad|>"
#if new_pad_token not in tokenizer.get_vocab():
#    tokenizer.add_special_tokens({'pad_token': new_pad_token})
#tokenizer.pad_token = new_pad_token
#tokenizer.save_pretrained('new_tokenizer/')

#tokenizer = AutoTokenizer.from_pretrained('new_tokenizer')
#model.resize_token_embeddings(len(tokenizer))


args = TrainingArguments(
    output_dir="checkpoints/dpo_model_test",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=5e-7,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=25,
    save_strategy = "steps",
    save_steps=600,
    save_total_limit=4,
    evaluation_strategy="steps",
    eval_steps=300,
    fp16 = True,
    tf32=True,
    push_to_hub=False,
    report_to="wandb",
)
dpo_args = {
    "beta": 0.1,
    "loss_type": "sigmoid"
}

prompt_max_length = 1087
max_seq_length = 1087

trainer = DPOTrainer(
    model,
    ref_model=None,
    peft_config=peft_config,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
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