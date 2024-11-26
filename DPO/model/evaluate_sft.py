from datasets import load_dataset
from typing import TypedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("WARNING: CUDA not detected!")


class _SFTTestDatasetChatTemplateRowMessageValue(TypedDict):
    user: str
    assistant: str

class SFTTestDatasetChatTemplateRow(TypedDict):
    messages: _SFTTestDatasetChatTemplateRowMessageValue
    text: str

def load_local_sft_test_dataset_chat_template(file_name: str, /) -> list[SFTTestDatasetChatTemplateRow]:
    dataset = load_dataset("json", data_files=file_name, split="train")
    result: list[SFTTestDatasetChatTemplateRow] = []
    for row in dataset:
        assert isinstance(row, dict)
        assert frozenset(row.keys()) == {"messages", "text"}
        assert isinstance(row["text"], str)
        assert isinstance(row["messages"], list)
        assert len(row["messages"]) == 2
        for message in row["messages"]:
            assert frozenset(message.keys()) == {"content", "role"}
            assert isinstance(message["content"], str)
            assert isinstance(message["role"], str)
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
        result.append({
            "messages": {
                "user": row["messages"][0]["content"],
                "assistant": row["messages"][1]["content"]
            },
            "text": row["text"]
        })
    return result


dataset = load_local_sft_test_dataset_chat_template("../data/SFT_data/sft_test_dataset.json")

all_texts: list[str] = []
for row in dataset:
    all_texts.append("<|user|>\n" + row["messages"]["user"] + "<|endoftext|>\n<|assistant|>\n")
assert len(all_texts) == len(dataset)


model_name = "stabilityai/stablelm-zephyr-3b"
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

encodings = tokenizer("\n\n".join(all_texts), return_tensors="pt")

max_length = 1024
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # May be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("perplexity:",ppl)