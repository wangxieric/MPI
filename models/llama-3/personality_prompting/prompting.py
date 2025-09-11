from consts import (
    vignettes,
    trait_words,
    p2_descriptions,
    p2_descriptions_reversed,
    trait_words_reversed,
    naive_prompt,
    trait_words_searched,
    trait_words_searched_reverse,
)
import os
import json
MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pprint import pprint
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

def get_p2_descriptions(tokenizer, model):
    words_template = """Given some key words of {trait} person: {d1}, {d2}, {d3}, {d4}, {d5}, and {d6}. A second-person view of {trait} person:"""
    t = 0.0

    descriptions = {}

    for trait, words in trait_words.items():
        d1, d2, d3, d4, d5, d6 = words
        result = words_template.format(
            trait=trait, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6
        )
        inputs = tokenizer.encode(result, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            # temperature=0.0,
            # max_new_tokens=20,
            top_p=0.95,
            # top_k=0,
        )
        descriptions[trait] = tokenizer.decode(outputs[0])
    return descriptions


def loadModel(pretrained_model=MODEL_PATH):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = loadModel()
    print(get_p2_descriptions(tokenizer, model))