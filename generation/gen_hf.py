import json
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model response"},
    )
    tokenizer: Optional[str] = field(
        default="HuggingFaceH4/mistral-7b-sft-beta",
        metadata={"help": "the tokenizer to use"},
    )
    ports: List[str] = field(default_factory=lambda: ["8000"], metadata={"help": "ports of the model response"})
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    dataset_name_or_path: Optional[str] = field(
        default="cornfieldrm/iterative-prompt-v1-iter1-2K",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8.jsonl",
        metadata={"help": "the location of the output file"},
    )
    bos_format: Optional[str] = field(
        default="",
        metadata={"help": "the format of the beginning of the sentence"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    max_workers: Optional[int] = field(
        default=1024,
        metadata={"help": "the number of workers"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds_dir = script_args.dataset_name_or_path
output_dir = script_args.output_dir
K = script_args.K
ports = script_args.ports

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)


def query_model(prompt, args, port):
    json = {
        **args,
        "prompt": prompt,
    }
    response = requests.post(url=script_args.url + ":" + str(port) + "/generate", json=json)
    response_json = response.json()
    return [response_json["text"][i][len(prompt) :] for i in range(len(response_json["text"]))]


default_args = {
    "use_beam_search": script_args.use_beam_search,
    "n": script_args.K,
    "temperature": script_args.temperature,
    "max_tokens": script_args.max_new_tokens,
    "seed": script_args.seed,
    "top_p": 1.0,
    "top_k": -1,
    "stop_token_ids": [tokenizer.eos_token_id] + script_args.eos_ids,
}

print(default_args)

ds = load_dataset(ds_dir, split="train")
# load_dataset("json", data_files=ds_dir, split="train", field="instances")
print(ds)

# use tokenizer.apply_template to apply the template to the prompt
ds = ds.map(
    lambda x: {
        "prompt": tokenizer.apply_chat_template(x[script_args.dataset_key], tokenize=False, add_generation_prompt=True)
    }
)


with ThreadPoolExecutor(max_workers=script_args.max_workers) as executor:
    result = [
        executor.submit(query_model, ds[i]["prompt"], default_args, ports[i % len(ports)]) for i in range(len(ds))
    ]
    # use tqdm to show progress
    for _ in tqdm(as_completed(result), total=len(result)):
        pass

    responses = [r.result() for r in result]


gathered_data = []
for i in range(len(ds)):
    tmp_data = {"prompt": ds[i][script_args.dataset_key], "responses": responses[i]}
    gathered_data.append(tmp_data)

print("I collect ", len(gathered_data), "samples")


with open(output_dir, 'w', encoding='utf8') as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')
