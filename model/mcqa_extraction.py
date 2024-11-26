import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.model_dpo import AutoDPOModelForCausalLM

def main():
    model_name = "agatha-duzan/gpt_2_padded"
    save_directory = "mcqa_gpt_2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    mcqa_model = AutoDPOModelForCausalLM(base_model)

    mcqa_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

if __name__ == "__main__":
    main()
