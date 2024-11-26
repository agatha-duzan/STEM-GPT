import json
import torch
import wandb
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftType
from trl import DPOTrainer, DPOConfig

def main():
    torch.cuda.empty_cache()
    model_name = 'microsoft/Phi-3-mini-4k-instruct'

    # load train and eval datasets
    train_path = 'model/datasets/DPO_train.jsonl'
    eval_path = 'model/datasets/DPO_eval.jsonl'
    dataset = load_dataset('json', data_files={"train": train_path, "evaluation": eval_path})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable()

    wandb.init(project="dpo_lora", dir="model/wandb")

    # Play with LORA configuration: start from hyperparams seen in literature, try different values later
    lora_config = LoraConfig(
        r= 32, # attention dimension; default is 8, try higher for more precision
        lora_alpha=16, # decrease it if we see unstable
        target_modules='all-linear', # try with all-linear for more radical changes 
        lora_dropout=0.01, #same as original LORA paper
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(model, lora_config)

    dpo_config = DPOConfig(
        beta=0.1,
        label_smoothing=0,
        loss_type="sigmoid",
        precompute_ref_log_probs=True,
        max_length=1647,
        max_prompt_length=528,
        max_target_length=1351, # exact lengths calculated in dpo_debug
        disable_dropout=True,
        generate_during_eval=False,
        truncation_mode="keep_end",
        learning_rate=2e-5,
        per_device_train_batch_size=1, # for peak VRAM: estim 28.47 GB
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        output_dir='model/checkpoints/dpo_lora',
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        report_to="wandb",
        fp16 = True
    )

    # check if there is a checkpoint to resume from
    '''last_checkpoint = None
    if os.path.isdir(dpo_config.output_dir):
        checkpoints = [os.path.join(dpo_config.output_dir, d) for d in os.listdir(dpo_config.output_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Resuming from checkpoint: {last_checkpoint}")'''

    last_checkpoint = None
    if os.path.isdir(dpo_config.output_dir):
        print(f"Checkpoint directory exists: {dpo_config.output_dir}")
        checkpoints = [os.path.join(dpo_config.output_dir, d) for d in os.listdir(dpo_config.output_dir) if d.startswith('checkpoint-')]
        print(f"Found checkpoints: {checkpoints}")
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Resuming from checkpoint: {last_checkpoint}")
        else:
            print("No checkpoints found.")
    else:
        print(f"Checkpoint directory does not exist: {dpo_config.output_dir}")
        os.makedirs(dpo_config.output_dir, exist_ok=True)

    trainer = DPOTrainer(
        model=lora_model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['evaluation'],
        tokenizer=tokenizer,
        data_collator=None,
        optimizers=(None, None) #trainer default is AdamW
    )

    print("Training start")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("Training finished !")

    print("Saving model:")
    model.save_pretrained('dpo_model')
    tokenizer.save_pretrained('dpo_model')
    print("Model saved !")

if __name__ == "__main__":
    main()
