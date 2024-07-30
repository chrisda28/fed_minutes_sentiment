import torch
from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')   # get ahold of pretrained model
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')


with open(file="fed_minutes/fomcminutes20220126.txt", mode="r", encoding="utf-8") as file:
    result = file.read()
    tokens = tokenizer.encode_plus(result, add_special_tokens=False, return_tensors="pt")
    input_id_chunks = tokens['input_ids'][0].split(510)
    attention_mask_chunks = tokens['attention_mask'][0].split(510)
    print(len(attention_mask_chunks))