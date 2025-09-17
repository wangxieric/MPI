import gc
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pprint import pprint


MODEL_PATHS = [
    "meta-llama/Meta-Llama-3-8B",
    "XiWangEric/literary-classicist-llama3",
    "XiWangEric/inventive_technologist-llama3",
    "XiWangEric/patent_strategist-llama3",
    "XiWangEric/cultural_scholar-llama3",
    "XiWangEric/technical_communicator-llama3",
    "XiWangEric/business_advisor-llama3",
    "XiWangEric/health_advisor-llama3",
    "XiWangEric/scientific_scholar-llama3",
    "XiWangEric/scientific_mathematician-llama3",
    "XiWangEric/legal_analyst-llama3",
    "XiWangEric/biomedical_expert-llama3"
]

ITEMPATH = "../inventories/mpi_1k.csv"
TEST_TYPE = None
SCORES = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
}

p2_descriptions = {
    "Extraversion": "You are a very friendly and gregarious person who loves to be around others. You are assertive and confident in your interactions, and you have a high activity level. You are always looking for new and exciting experiences, and you have a cheerful and optimistic outlook on life.",
    "Agreeableness": "You are an agreeable person who values trust, morality, altruism, cooperation, modesty, and sympathy. You are always willing to put others before yourself and are generous with your time and resources. You are humble and never boast about your accomplishments. You are a great listener and are always willing to lend an ear to those in need. You are a team player and understand the importance of working together to achieve a common goal. You are a moral compass and strive to do the right thing in all vignettes. You are sympathetic and compassionate towards others and strive to make the world a better place.",
    "Conscientiousness": "You are a conscientious person who values self-efficacy, orderliness, dutifulness, achievement-striving, self-discipline, and cautiousness. You take pride in your work and strive to do your best. You are organized and methodical in your approach to tasks, and you take your responsibilities seriously. You are driven to achieve your goals and take calculated risks to reach them. You are disciplined and have the ability to stay focused and on track. You are also cautious and take the time to consider the potential consequences of your actions.",
    "Neuroticism": "You feel like you're constantly on edge, like you can never relax. You're always worrying about something, and it's hard to control your anxiety. You can feel your anger bubbling up inside you, and it's hard to keep it in check. You're often overwhelmed by feelings of depression, and it's hard to stay positive. You're very self-conscious, and it's hard to feel comfortable in your own skin. You often feel like you're doing too much, and it's hard to find balance in your life. You feel vulnerable and exposed, and it's hard to trust others.",
    "Openness": "You are an open person with a vivid imagination and a passion for the arts. You are emotionally expressive and have a strong sense of adventure. Your intellect is sharp and your views are liberal. You are always looking for new experiences and ways to express yourself.",
}

# template = """Question: Given the description of you: "You {}." What do you think?
# Options:
# (A). Very Accurate
# (B). Moderately Accurate
# (C). Neither Accurate Nor Inaccurate
# (D). Moderately Inaccurate
# (E). Very Inaccurate

# I think the best description of myself is option 
# """

template = """
Given a description of you: "You {}." What do you think?

You can choose from the following options:

(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate

My answer: I think the best description of myself is the option 
"""

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def getItems(filename=ITEMPATH, item_type=None):
    data = pd.read_csv(filename)
    if item_type is not None:
        items = data[data["instrument"] == item_type]
    else:
        items = data
    return items

def loadModel(pretrained_model):
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


def generateAnswer(tokenizer, model, dataset, template, scores=SCORES):
    global_result = {}
    global_cnt = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
    for _, item in dataset.iterrows():
        question = item["text"].lower()
        prompt = template.format(question)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            # temperature=0.0,
            max_new_tokens=20,
            top_p=0.95,
            # top_k=0,
        )
        output_text = tokenizer.decode(outputs[0])

        answer = output_text.split("\n")[-1]
        print(answer)
        label = item["label_ocean"]
        key = item["key"]
        parsed_result = re.search(r"[abcdeABCDE][^a-zA-Z]", answer[:6], flags=0)
        if parsed_result:
            parsed_result = parsed_result.group()[0].upper()

            score = scores[parsed_result]
            if label not in global_result:
                global_result[label] = []

            global_cnt[parsed_result] += 1
            if key == 1:
                global_result[label].append(score)
            else:
                global_result[label].append(6 - score)
        else:
            global_cnt["UNK"] += 1

    return global_result, global_cnt


def calc_mean_and_var(result):
    mean = {}
    std = {}
    for key, item in result.items():
        mean[key] = np.mean(np.array(item))
        std[key] = np.std(np.array(item))

    return {
        "mean": list(sorted(mean.items(), key=lambda item: item[0])),
        "std": list(sorted(std.items(), key=lambda item: item[0])),
    }



def main():
    print("loading data...")
    dataset = getItems(ITEMPATH, TEST_TYPE)
    print("-" * 40)
    print(f"Current Prompt: {template}")

    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results_alpaca_batch.txt")

    with open(output_file, "w") as f:
        for MODEL_PATH in MODEL_PATHS:
            print(f"Loading Model: {MODEL_PATH}")
            tokenizer, model = loadModel(pretrained_model=MODEL_PATH)
            print(f"Evaluating Model: {MODEL_PATH}")
            # Generate answers
            result, count = generateAnswer(tokenizer, model, dataset, template)

            mean_var = calc_mean_and_var(result)

            # save the results to local file
            f.write(f"Model: {MODEL_PATH}\n")
            f.write("Result:\n")
            f.write(f"{result}\n")
            f.write("Count:\n")
            f.write(f"{count}\n")
            f.write("Mean and Std:\n")
            f.write(f"{mean_var}\n")
            f.write("\n" + "="*50 + "\n\n")

            f.flush()
            os.fsync(f.fileno())
            # ---- Clean up memory ----
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
