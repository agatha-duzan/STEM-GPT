import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.cuda.empty_cache()
gc.collect()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
if tokenizer.pad_token is None:
    print("Adding pad token to tokenizer")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.resize_token_embeddings(len(tokenizer))
# because we added pad token

model.save_pretrained("gpt_2_padded")
tokenizer.save_pretrained("gpt_2_padded")