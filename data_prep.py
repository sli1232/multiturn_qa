from datasets import load_dataset
import pandas as pd
import json
from tqdm import tqdm


ds_1 = load_dataset("THUDM/LongBench", "passage_retrieval_en")
ds_2 = load_dataset("THUDM/LongBench", "passage_retrieval_en_e") 
# Combine the two datasets by converting them to DataFrames and using pandas.concat
df_1 = pd.DataFrame(ds_1['test'])
df_2 = pd.DataFrame(ds_2['test'])
df = pd.concat([df_1, df_2], ignore_index=True)
print(df.info())

input_text = df.loc[0]['context'] + df.loc[0]['input']
# count the number of words in the context
print(len(input_text.split()))


total_samples = len(df)

print("Extracting answer and noise paragraphs...")

answer_paragraphs = [] # list of answer paragraphs
noise_paragraphs = [] 
inputs = []

from tqdm import tqdm

total_samples = len(df)

print("Extracting answer and noise paragraphs...")

answer_paragraphs = [] # list of answer paragraphs
noise_paragraphs = [] 
inputs = []

# for each sample in the dataset, extract the answer paragraph and noise paragraphs
for i in tqdm(range(total_samples)):  
    sample = df.loc[i]
    context = sample['context']
    answer = sample['answers'][0]
    input = sample['input']
    ans_idx = int(re.search(r"\d+" , answer).group()) - 1

    paragraphs = context.split("\n\n")
    # remove the "Paragraph x:" prefix
    paragraphs = [re.sub(r"Paragraph \d+: ", "", para) for para in paragraphs]
    # get the answer paragraph
    ans_para = paragraphs[ans_idx]
    answer_paragraphs.append(ans_para)
    # remove the answer paragraph from the list of paragraphs
    paragraphs.remove(paragraphs[ans_idx])
    noise_paragraphs.extend(paragraphs)

    inputs.append(input)

print(f"# of answer paragraphs: {len(answer_paragraphs)}, # of noise paragraphs: {len(noise_paragraphs)}")

print("Generating new data samples...")
num_questions = 3
num_noise = 27
num_paras_total = num_questions + num_noise
num_generated_samples = 5

assert (num_generated_samples * num_questions) <= len(answer_paragraphs)

# randomly shuffle answer and noise paragraphs
import random
random.shuffle(noise_paragraphs)

new_samples = []

for i in tqdm(range(num_generated_samples)):
    # take the first "num_questions" paragraphs as the answer paragraphs
    ans_paras = answer_paragraphs[i*num_questions:(i+1)*num_questions]
    ans_inputs = inputs[i*num_questions:(i+1)*num_questions]
    noise_paras = noise_paragraphs[i*num_noise:(i+1)*num_noise]

    #randomly generate "num_questions" integers in range of "num_noise + num_questions"
    ans_indices = random.sample(range(num_paras_total), num_questions)

    sample_paras = [None for _ in range(num_paras_total)]
    for j, idx in enumerate(ans_indices):
        sample_paras[idx] = ans_paras[j]
    j = 0
    for i, para in enumerate(noise_paras):
        while sample_paras[j] is not None:
            j += 1
        sample_paras[j] = para


    # add the prefix "Paragraph x:" to new_samples
    sample_paras = [f"Paragraph {i+1}: {para}" for i, para in enumerate(sample_paras)]

    answers = [f"Paragraph {idx+1}: " for idx in ans_indices]
    new_sample = {
        "inputs": ans_inputs,
        "answers": answers,
        "context": "\n\n".join(sample_paras)
    }
    new_samples.append(new_sample)

print(f"Generated {len(new_samples)} new samples")

# save the new samples to a file
import json
with open("new_samples.json", "w") as f:
    json.dump(new_samples, f, indent=4)




