import torch
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

torch.cuda.empty_cache()
gc.collect()

print("Initial GPU memory usage:")
print(torch.cuda.memory_summary())

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
if tokenizer.pad_token is None:
    print("Adding pad token to tokenizer")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

print("Loading datasets")
train_dataset = load_dataset('json', data_files='datasets/DPO_train.jsonl')['train']
eval_dataset = load_dataset('json', data_files='datasets/DPO_eval.jsonl')['train']
print("Datasets loaded")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
model.resize_token_embeddings(len(tokenizer))
# because we added pad token

print("Model loaded")
print("GPU memory usage after loading model:")
print(torch.cuda.memory_summary())

training_args = DPOConfig(
    output_dir="./results",
    per_device_train_batch_size=1,  # reduce batch size if necessary to fit into GPU memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    learning_rate=5e-7, # literature on gpt-neo: learning rates from 2e − 5 to 3e − 7
    beta=0.1,
    label_smoothing=0,
    loss_type="sigmoid",
    truncation_mode="keep_end",
    max_length=1668,
    max_prompt_length=741,
    max_target_length=1352,
    precompute_ref_log_probs=False,
    report_to=[],  # disable logging to W&B
    fp16=True
)

print("Trainer initialization...")
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # default: creates a ref model with the same architecture
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("Trainer initialized")

torch.cuda.empty_cache()
gc.collect()

print("GPU memory usage before training:")
print(torch.cuda.memory_summary())
print("Cuda cleaned, begin training")

try:
    trainer.train()
except RuntimeError as e:
    print(f"Training failed with error: {e}")
    print("GPU memory usage at the time of failure:")
    print(torch.cuda.memory_summary())
    raise

print("Training finished, saving...")
trainer.save_model("./results")
tokenizer.save_pretrained("./results")
