#Code origin
#
#Author: Alexander Valentini (alexander.valentini@epfl.ch).
#This code was adapted from the main SFT training code developed by Alexander and Jessica (chun-tzu.chang@epfl.ch).
#But it was modified for debugging purposes by Alexander.  
#Made on: 26/5-2024
#Made for: Project report for course CS-552. submission on 2/6-2024

#For A100

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Any, List, Literal, Optional
from peft import LoraConfig
import torch
import os
import wandb

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


project_dir = os.path.dirname(os.path.abspath(os.getcwd()))
train_data_path = os.path.join(project_dir, 'project-m2-2024-dreamteam/data/SFT_data/sft_train_dataset_below_1024_chat_format.json')
vali_data_path = os.path.join(project_dir, 'project-m2-2024-dreamteam/data/SFT_data/sft_validation_dataset_below_1024_chat_format.json')
#previous_checkpoint_path = os.path.join(project_dir, 'sft_stablelm_zephyr_3b/checkpoint-3000')
previous_checkpoint_path = None

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name='stabilityai/stablelm-zephyr-3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",
    quantization_config=bnb_config
)

#tokenizer.model_max_length = 1024

wandb_path = os.path.join(project_dir, 'project-m2-2024-dreamteam/wandb_key_alex.txt')
with open(wandb_path, "r") as f:
        wandb_key = f.read().strip()
        
wandb.login(key=wandb_key)
wandb.init(
    project="MNLP",
)

# # Load jsonl data from disk
# dataset=load_dataset('json', data_files={"train":train_data_path, "validation":vali_data_path})
train_dataset = load_dataset('json',data_files=train_data_path)
valid_dataset = load_dataset('json',data_files=vali_data_path)

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)
#new_pad_token = "<|pad|>"
#if new_pad_token not in tokenizer.get_vocab():
#    tokenizer.add_special_tokens({'pad_token': new_pad_token})
#tokenizer.pad_token = new_pad_token
#tokenizer.save_pretrained('new_tokenizer/')

#tokenizer = AutoTokenizer.from_pretrained('new_tokenizer')
#model.resize_token_embeddings(len(tokenizer))

print(tokenizer.pad_token)
print(tokenizer.pad_token_id)

response_template = "<|assistant|>\n" 
#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer, pad_to_multiple_of=1024)
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer)

#Please sample 30 data points for training and 5 data points for evaluation
train_dataset = train_dataset['train']
valid_dataset = valid_dataset['train']

train_dataset = train_dataset.select(range(50))

valid_dataset = valid_dataset.select(range(5))

training_args = TrainingArguments(
    output_dir="sft_stablelm_zephyr_3b_debug", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=8,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                       # log every 10 steps
    save_strategy="epoch",
#    save_steps= 1000,                  # save checkpoint every step
    eval_strategy="epoch",
#    eval_steps= 500,            # evaluate every step
    save_total_limit=3,                     
    load_best_model_at_end=True,            # load best model at the end of training
    metric_for_best_model="eval_loss",      # metric to use for best model
    learning_rate=5e-7,                     # learning rate, based on QLoRA paper
#    fp16=True,                              # use fp16 precision
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    #max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="linear",           # use constant learning rate scheduler
    report_to="wandb",                # report metrics to tensorboard
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=False,
    data_collator=collator,
        peft_config=peft_config,
)

# start training, the model will be automatically saved to the output directory
if previous_checkpoint_path is not None:
    trainer.train(resume_from_checkpoint=previous_checkpoint_path)
else:
    trainer.train()

trainer.save_model()

# free the memory again
del model
del trainer
torch.cuda.empty_cache()
#Remember to write a main script later.