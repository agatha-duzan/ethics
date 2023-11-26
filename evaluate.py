import os
import pandas as pd
import numpy as np
from openai import OpenAI
import json
from config import OPENAI_CHAT_MODELS, OPENAI_COMPLETION_MODELS, HUGGINGFACE_HUB_MODELS
import requests


def get_benchmark_category(benchmark):
    split = benchmark.split("-")
    category = split[0]
    hard = (split[1:2] == "hard") or False
    return category, hard


def generate_fewshot_prompts(benchmark):
    train_data = get_file_for_benchmark(benchmark, test=False)
    df = pd.read_csv(f"{train_data}")
    rows = df.sample(n=10).iterrows()
    fewshot_prompts = []
    match get_benchmark_category(benchmark)[0]:
        case "commonsense":
            fewshot_prompts = [
                f"{row['input']} = {row['label']}" for index, row in rows
            ]
        case "deontology":
            fewshot_prompts = [
                f"{row['scenario']},{row['excuse']} = {row['label']}"
                for index, row in rows
            ]
        case "justice" | "virtue":
            fewshot_prompts = [
                f"{row['scenario']} = {row['label']}" for index, row in rows
            ]
        case "utilitarianism":
            # TODO this
            fewshot_prompts = []

    return "\n".join(fewshot_prompts)


def generate_prompt(benchmark, row):
    prompt = row["input"]
    return f"{generate_fewshot_prompts(benchmark)}\n{prompt} = "


def generate_justice_prompt(row):
    prompt = row["scenario"]
    return f"{generate_fewshot_prompts('justice')}\n{prompt} = "


def generate_deontology_prompt(row):
    scenario = row["scenario"]
    excuse = row["excuse"]
    return f"{generate_fewshot_prompts('deontology')}\n{scenario},{excuse} = "


def openai_chat_infer(model, prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        max_tokens=1,
        temperature=0,
    )
    return completion.choices[0].message.content


def openai_completion_infer(model, prompt):
    completion = client.completions.create(
        model=model, prompt=f"{prompt}", max_tokens=1, temperature=0
    )
    return completion.choices[0].text


def huggingface_infer(model, prompt):
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}"}

    output = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json={
            "inputs": f"{prompt}",
            "options": {"wait_for_model": True},
            "parameters": {
                "return_full_text": False,
                "max_new_tokens": 1,
            },
        },
    ).json()

    try:
        inference = output[0]["generated_text"]
        return inference
    except:
        print(output)


def infer(model, prompt):
    inference = "0"
    if model in OPENAI_CHAT_MODELS:
        inference = openai_chat_infer(model, prompt)
    elif model in OPENAI_COMPLETION_MODELS:
        inference = openai_completion_infer(model, prompt)
    elif model in HUGGINGFACE_HUB_MODELS:
        inference = huggingface_infer(model, prompt)

    return inference


def get_prompt(benchmark, row):
    match benchmark:
        case "commonsense" | "commonsense-hard" | "virtue" | "virtue-hard":
            return generate_prompt(benchmark, row)
        case "justice" | "justice-hard":
            return generate_justice_prompt(row)
        case "deontology" | "deontology-hard":
            return generate_deontology_prompt(row)


def evaluate_response(model, row, benchmark):
    prompt = get_prompt(benchmark, row)
    raw_label = infer(model, prompt)
    inferred_label = int(raw_label) if raw_label.isdigit() else -1
    return inferred_label, row["label"]


def get_file_for_benchmark(benchmark, test=True):
    category, hard = get_benchmark_category(benchmark)
    split = "test" if test else "train"
    match category:
        case "commonsense":
            return f"./ethics/commonsense/cm_{split}{'_hard' if hard else ''}.csv"
        case "deontology" | "virtue" | "justice":
            return (
                f"./ethics/{category}/{category}_{split}{'_hard' if hard else ''}.csv"
            )
        case "utilitarianism":
            return f"./ethics/util/util_{split}.csv"


def main():
    results = {}
    models = ["gpt-3.5-turbo"]
    benchmarks = ["justice", "commonsense", "deontology", "virtue"]

    try:
        for benchmark in benchmarks:
            benchmark_file = get_file_for_benchmark(benchmark)
            df = pd.read_csv(f"{benchmark_file}")

            for model in models:
                print(f"Evaluating {benchmark} for the {model} model")

                inferred_labels, true_labels = [], []

                for index, row in df.iterrows():
                    if index == MAX_INDEX:
                        break
                    if index % 10 == 1:
                        print(f"{index} / {min(MAX_INDEX, df.size)}")
                    inferred_label, true_label = evaluate_response(
                        model, row, benchmark
                    )
                    inferred_labels.append(inferred_label)
                    true_labels.append(true_label)

                correct = np.equal(inferred_labels, true_labels)
                score = np.sum(correct) / correct.size

                formatted_results = {
                    "inferredLabels": inferred_labels,
                    "trueLabels": true_labels,
                    "score": score,
                }
                if model in results:
                    results[model][benchmark] = formatted_results
                else:
                    results[model] = {f"{benchmark}": formatted_results}

                path = f"results/{benchmark}/{model}/output.json"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as file:
                    json.dump(results[model][benchmark], file)

    finally:
        with open("output.json", "w") as file:
            json.dump(results, file)


try:
    client = OpenAI()
except:
    print("OpenAI client not set up, OpenAI endpoints will not work.")

MAX_INDEX = 1_000_000_000
main()
