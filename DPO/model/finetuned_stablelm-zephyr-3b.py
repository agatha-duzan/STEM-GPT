#Code origin
#
#Author: Alexander Valentini (alexander.valentini@epfl.ch) and Jessica (chun-tzu.chang@epfl.ch)
#Made on: 26/5-2024
#Made for: Project report for course CS-552. submission on 2/6-2024


#For A100

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Any, List, Literal, Optional
from peft import LoraConfig
import torch
import os
import wandb

project_dir = os.path.dirname(os.path.abspath(os.getcwd()))
#train_data_path = os.path.join(project_dir, 'data/SFT_data/sft_train_dataset_below_1024_chat_format_no_length.json')
#vali_data_path = os.path.join(project_dir, 'data/SFT_data/sft_validation_dataset_below_1024_chat_format_no_length.json')

train_data_path = 'data/sft_data_latest/sft_train_dataset.jsonl'
vali_data_path = 'data/sft_data_latest/sft_validation_dataset.jsonl'

dataset=load_dataset('json', data_files={"train":train_data_path, "validation":vali_data_path})

#train_dataset = load_dataset('json',data_files=train_data_path)
#valid_dataset = load_dataset('json',data_files=vali_data_path)

#data_dir = os.path.join(project_dir, 'data/SFT_data/SFT_data_no_length_1024')
#dataset = load_from_disk(data_dir)

#previous_checkpoint_path = os.path.join(project_dir, 'sft_stablelm_zephyr_3b/checkpoint-3000')
previous_checkpoint_path = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name='stabilityai/stablelm-zephyr-3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
).to(device)

#tokenizer.model_max_length = 1024

wandb_path = os.path.join(project_dir, 'wandb_key_alex.txt')
with open(wandb_path, "r") as f:
        wandb_key = f.read().strip()
        
wandb.login(key=wandb_key)
wandb.init(
    project="MNLP",
)

# # Load jsonl data from disk
# dataset=load_dataset('json', data_files={"train":train_data_path, "validation":vali_data_path})
#train_dataset = load_dataset('json',data_files=train_data_path)
#valid_dataset = load_dataset('json',data_files=vali_data_path)

#Alexander: Not using Lora since GPU can fit model
#peft_config = LoraConfig(
#        lora_alpha=128,
#        lora_dropout=0.05,
#        r=256,
#        bias="none",
#        target_modules="all-linear",
#        task_type="CAUSAL_LM",
#)

#Alexander: Changing padding token
new_pad_token = "<|pad|>"
if new_pad_token not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'pad_token': new_pad_token})
tokenizer.pad_token = new_pad_token
tokenizer.save_pretrained('new_tokenizer/')

tokenizer = AutoTokenizer.from_pretrained('new_tokenizer')
model.resize_token_embeddings(len(tokenizer))

print(tokenizer.pad_token)
print(tokenizer.pad_token_id)


response_template = "<|assistant|>\n" 


#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer, pad_to_multiple_of=1024)
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer)

#This part was inspired by the Phil Schmid tutorial:  
training_args = TrainingArguments(
    output_dir="sft_stablelm_zephyr_3b", # directory to save and repository id
    num_train_epochs=3,                     
    per_device_train_batch_size=8,          
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            #Alexander: use gradient checkpointing to save memory - from tutorial
    optim="adamw_torch_fused",              # Alexander: use fused adamw optimizer for faster training
    logging_steps=10,                       # Alexander: log every 10 steps for better debugging
    save_strategy="steps",
    save_steps= 1000,                  
    eval_strategy="steps",
    eval_steps= 500,            
    save_total_limit=3,                     
    load_best_model_at_end=True,            #Load best model at the end of training
    metric_for_best_model="eval_loss",      # metric to use for best model
    learning_rate=5e-7,                     # Lower learning rate to avoid 
#    fp16=True,                             
    bf16=True,                              
    tf32=True,                              
    warmup_ratio=0.1,                      
    lr_scheduler_type="linear",           # use constant learning rate scheduler
    report_to="wandb",                # report metrics to tensorboard
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    packing=False,
    data_collator=collator,
#    peft_config=peft_config,
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