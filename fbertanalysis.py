import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os
import matplotlib.pyplot as plt
import seaborn as sb

FOLDER_PATH = "fed_minutes"
MODEL_NAME = "ProsusAI/finbert"
KEYWORDS = ['inflation', 'interest rates', 'unemployment', 'gdp growth', 'monetary policy',  # keywords to track
            'fiscal policy', 'debt', 'liquidity', 'labor market', 'consumer spending',
            'supply chain', 'financial stability', 'deficit', 'surplus', 'yield', 'leverage', 'housing', 'mortgage']


def load_model_and_tokenizer():
    """initialize model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)  # initialize pretrained model and tokenizer
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


def extract_keywords(keywords, text):  # LIMITATION: currently finds partial matches (i.e. 'cat' in 'category')
    """returns dict of keyword and its count within the text"""
    text = text.lower()  # ensure everything is lowercase
    key_count_dict = {}
    for keyword in keywords:
        count = text.count(keyword)  # count instance of each keyword
        key_count_dict[keyword] = count  # set key as keyword and value as its corresponding count
    return key_count_dict


def get_sentiment_dict(tokenizer, text, chunksize=512):  # chunksize max is 512
    """prepare tokens for sentiment analysis"""
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    # tokenizing text, returns PT Tensors
    input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))  # splitting each chunk into 510 tokens
    attention_mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))

    for i in range(len(input_id_chunks)):  # iterating over each chunk
        input_id_chunks[i] = torch.cat([  # Add special tokens: 101 (CLS) at start and 102 (SEP) at end of each chunk
            torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
        ])
        attention_mask_chunks[i] = torch.cat([  # Add attention mask values for special tokens
            torch.tensor([1]), attention_mask_chunks[i], torch.tensor([1])
        ])
        # Calculate how much padding is needed to reach the chunksize
        pad_length = chunksize - input_id_chunks[i].shape[0]

        if pad_length > 0:  # chunk is not 512, filling in padding so all chunks are the max token length (512)
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.tensor([0] * pad_length, dtype=torch.long)
            ])
            attention_mask_chunks[i] = torch.cat([   # Add padding (0s) to input IDs and attention mask if needed
                attention_mask_chunks[i], torch.tensor([0] * pad_length, dtype=torch.long)
            ])
    return {   # Stack chunks into single tensors, creating batches for efficient processing
        "input_ids": torch.stack(input_id_chunks).long(),
        "attention_mask": torch.stack(attention_mask_chunks).long()
    }
# attention mask value for real token = 1 while for padding, attention_mask value = 0
# input_ids are a numerical representation of each token


def get_sentiment(input_dict, model):
    """find the final sentiment (0, 1, or 2) meaning negative, neutral, or positive"""
    all_probabilities = []
    for i in range(input_dict['input_ids'].shape[0]):  # iterate over each sample in input batch
        chunk_input = {
            'input_ids': input_dict['input_ids'][i].unsqueeze(0),
            'attention_mask': input_dict['attention_mask'][i].unsqueeze(0)
        }
        with torch.no_grad():  # no gradient calculation this improves efficiency
            outputs = model(**chunk_input)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)  # use softmax to get probabilities
        all_probabilities.append(probabilities)  # put probabilities into list
    mean_probabilities = torch.cat(all_probabilities).mean(dim=0)
    return torch.argmax(mean_probabilities).item()  # find occurrence of highest probability to get final sentiment


def analyze_doc(file_path, tokenizer, model, keywords):
    """find sentiment and keyword count for single document"""
    with open(file=file_path, mode='r', encoding='utf-8') as file:
        text = file.read()   # reading file
        keyword_count = extract_keywords(keywords, text)  # find keyword count within text
        input_dict = get_sentiment_dict(tokenizer, text)  # preparing text for sentiment analysis
        sentiment = get_sentiment(input_dict, model)  # finding sentiment of text
        return keyword_count, sentiment


def analyze_multiple_docs(tokenizer, model):
    """get dataframe storing sentiment and keyword count for all text files"""
    results = []
    for text_file in os.listdir(FOLDER_PATH):  # use analyze_doc function for multiple articles
        keyword_dict, sentiment = analyze_doc(file_path=f"{FOLDER_PATH}/{text_file}",
                                              tokenizer=tokenizer,
                                              model=model,
                                              keywords=KEYWORDS)
        result_dict = {
            "file_name": text_file,
            "sentiment": sentiment,
            **keyword_dict  # unpacking the keyword dict into this one, so it won't be a nested dict
        }
        results.append(result_dict)  # store result into list of dictionaries
    return pd.DataFrame(results)  # convert to a dataframe


def plot_keyword_heatmap(df, keywords):
    """generate heatmap showing keyword frequency for each file"""
    heatmap_data = df[keywords]
    filenames = df['file_name'].tolist()  # turning file_names into list

    plt.figure(figsize=(16, 10))  # set figure size
    ax = sb.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt="d")  # Create heatmap

    plt.title("Keyword Frequency Heatmap")
    plt.xlabel("Keywords")
    plt.ylabel("FOMC Minutes")
    # Set y-tick labels to filenames and adjust their position
    ax.set_yticks(range(len(filenames)))
    ax.set_yticklabels(filenames, rotation=0)  # each file name will be tick mark on y-axis

    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # to prevent overlapping
    plt.show()


def plot_sentiment_trend(df):
    """Plot sentiment trend over time"""
    # Convert file_name to datetime and formatting filename to get just the date
    df['date'] = pd.to_datetime(df['file_name'].str.replace('fomcminutes', '').str.replace('.txt', ''), format='%Y%m%d')

    df_sorted = df.sort_values('date')   # Sort by date

    plt.figure(figsize=(12, 6))  # set figure size
    plt.plot(df_sorted['date'], df_sorted['sentiment'], marker='o')  # Create line plot

    plt.title("Sentiment Trend of FOMC Minutes")
    plt.xlabel("Date")
    plt.ylabel("Sentiment (0: Negative, 1: Neutral, 2: Positive)")
    plt.ylim(-0.5, 2.5)  # Set y-axis limits
    plt.yticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()  # to prevent overlapping
    plt.xticks(rotation=45)
    plt.show()






