import os
import tiktoken
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader

CORPUS_PATH = "data/the-verdict.txt"


def _download_corpus():
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)

    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, CORPUS_PATH)


def _play_with_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")

    text = "<|endoftext|>"
    tokens = tokenizer.encode(text, allowed_special="all")
    print(tokens)

    text = "Akwirw ier"
    tokens = tokenizer.encode(text, allowed_special="all")
    print(tokens)
    print(tokenizer.decode(tokens) == text)


def _play_with_dataset_v1():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)

    enc_sample = enc_text[50:]
    context_length = 4
    x = enc_sample[:context_length]
    y = enc_sample[1 : context_length + 1]
    print(f"x: {x}")
    print(f"y:      {y}")
    print("--------------------------------")

    for i in range(1, context_length + 1):
        x = enc_sample[:i]
        y = enc_sample[i]
        print(f"{tokenizer.decode(x)} -> {tokenizer.decode([y])}")

    print("--------------------------------")

    print("Dataloader V1")
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, context_length=4, stride=1, shuffle=False
    )
    dataloader_iter = iter(dataloader)
    x, y = next(dataloader_iter)
    print(f"x: {x}")
    print(f"y: {y}")
    print()
    x, y = next(dataloader_iter)
    print(f"x: {x}")
    print(f"y: {y}")


class DatasetV1(Dataset):
    def __init__(self, txt, tokenizer, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        tokens = tokenizer.encode(txt, allowed_special="all")
        for i in range(0, len(tokens) - context_length, stride):
            self.input_ids.append(torch.tensor(tokens[i : i + context_length]))
            self.target_ids.append(torch.tensor(tokens[i + 1 : i + context_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, context_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = DatasetV1(txt, tokenizer, context_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader


def main():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")

    # parameters
    vocab_size = tokenizer.n_vocab
    embedding_dim = 512
    context_length = 256
    batch_size = 4

    dataloader = create_dataloader_v1(
        raw_text, batch_size=batch_size, context_length=context_length, stride=128, shuffle=False
    )
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    inputs, targets = batch
    print("Inputs:", inputs.shape)
    print("Targets:", targets.shape)

    token_embeddings_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    token_embeddings = token_embeddings_layer(inputs)
    print("Token embeddings:", token_embeddings.shape)

    position_embeddings_layer = torch.nn.Embedding(context_length, embedding_dim)
    position_embeddings = position_embeddings_layer(torch.arange(context_length))
    print("Position embeddings:", position_embeddings.shape)

    input_embeddings = token_embeddings + position_embeddings
    print("Input embeddings:", input_embeddings.shape)


if __name__ == "__main__":
    main()
