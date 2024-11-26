from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

train_data_path = 'datasets/MCQA_train.jsonl'
eval_data_path = 'datasets/MCQA_eval.jsonl'

dataset=load_dataset('json', data_files={"train":train_data_path, "validation":eval_data_path})

model_name = "agatha-duzan/dpo_gpt_neo_3"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = example['question'][i] + example['answer'][i]
        output_texts.append(text)
    return output_texts

# we want to FT our model on its responses only
response_template = "Answer:"
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

sft_args = SFTConfig(
    learning_rate=5e-5,
    dataset_batch_size=1,
    num_train_epochs=3,
    warmup_steps=100,
    gradient_accumulation_steps=2,
    output_dir="./sft_model_dpo",
    logging_dir='./logs',
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    max_seq_length = 512,
    formatting_func=formatting_prompts_func,
    packing=False, # train model on the generated prompts only
    data_collator=data_collator
)

trainer.train()
