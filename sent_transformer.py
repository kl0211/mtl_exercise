import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states, attention_mask):
    # We don't want to include padded tokens, so use masked_fill to zero them out.
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")
df = pd.read_csv("hf://datasets/codesignal/sms-spam-collection/sms-spam-collection.csv")

inputs = tokenizer(
    df[:5]["message"].tolist(),  # Let's just grab the first 5 and encode those
    padding=True,  # Make sure to pad out to the longest sequence of these inputs
    truncation=True,  # truncate any sequence longer than what the model supports
    return_tensors='pt'  # return pytorch tensors
)

with torch.no_grad():  # We're not training, so no need to calculate gradients
    outputs = model(**inputs)  # Give the model 'input_ids', 'token_type_ids', and 'attention_mask'

embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
print(embeddings)
print(embeddings.shape)
