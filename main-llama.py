import os
# Custom cache
PATH = 'D:/dev/projekt/llm-testing/.cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

import transformers
import torch

model_id = "AI-Sweden-Models/Llama-3-8B-instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

context = []

for line in open('./data/lagbok.txt', 'r'):
    context.append(line)

messages = [
    {"role": "system", "content": "Du är en hjälpsam assistant som svarar klokt och vänligt."},
    {"role": "user", "content": input(f"\nEnter your query: ")},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=1024,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(outputs[0]["generated_text"][-1])