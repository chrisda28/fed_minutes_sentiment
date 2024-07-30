import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os

FOLDER_PATH = "fed_minutes"
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)   # get ahold of pretrained model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

for filename in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, filename)
    with open(file=file_path, mode="r", encoding="utf-8") as file:
        result = file.read()
        tokens = tokenizer.encode_plus(result, add_special_tokens=False, return_tensors="pt")
        chunksize = 512
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

        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(attention_mask_chunks)

        input_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.int()
        }


        # use input_dict[‘input_ids’].shape  to see that all windows are 512, first number in output is batch size
        outputs = model(**input_dict)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)

        mean_probabilities = probabilities.mean(dim=0)

        print(torch.argmax(mean_probabilities).item())  # gives single classification (0 ,1, or 2)  2 is neutral
