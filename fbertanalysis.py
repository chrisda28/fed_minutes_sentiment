import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os
from collections import Counter

FOLDER_PATH = "fed_minutes"
MODEL_NAME = "ProsusAI/finbert"
KEYWORDS = ['inflation', 'interest rates', 'unemployment', 'gdp growth', 'monetary policy',
            'fiscal policy', 'quantitative easing', 'quantitative tightening', 'labor market', 'consumer spending',
            'supply chain', 'financial stability', 'deficit', 'surplus']


def load_model_and_tokenizer():
    """initialize model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)  # get a hold of pretrained model
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


def extract_keywords(keywords, text):  # LIMITATION: currently finds partial matches (i.e. 'cat' in 'category')
    """returns dict of keyword and its count within the text"""
    text = text.lower()  # ensure everything is lowercase
    # key_count = Counter()  # returns dict of tally amount for each keyword
    key_count = {}
    for keyword in keywords:
        count = text.count(keyword)  # count instance of each keyword
        key_count[keyword] = count  # set key as keyword and value as its corresponding count
    return key_count


def get_sentiment(tokenizer, text, chunksize=512):
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
    attention_mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))

    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([
            torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
        ])

        attention_mask_chunks[i] = torch.cat([
            torch.tensor([1]), attention_mask_chunks[i], torch.tensor([1])
        ])

        pad_length = chunksize - input_id_chunks[i].shape[0]

        if pad_length > 0:
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_length)
            ])

            attention_mask_chunks[i] = torch.cat([
                attention_mask_chunks[i], torch.Tensor([0] * pad_length)
            ])
    return {
        "input_ids": torch.stack(input_id_chunks),
        "attention_mask": torch.stack(attention_mask_chunks)
    }


def analyze_doc(folder_path, tokenizer, model, keywords):
    pass


def analyze_multiple_docs():
    pass

