import os
import tiktoken
import urllib.request

import torch
import torch.nn as nn
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


def _play_with_attention():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )
    print("inputs.shape:", inputs.shape)

    query_2 = inputs[1]

    # dot product of query with input embedding vectors
    attention_scores = torch.zeros(inputs.shape[0])
    for i in range(inputs.shape[0]):
        attention_scores[i] = torch.dot(query_2, inputs[i])

    print(f"\nattention_scores:\n{attention_scores}")

    # normalize the attention scores with softmax
    attention_weights = torch.softmax(attention_scores, dim=0)
    print(f"\nattention_weights\n{attention_weights}")

    context_vector_2 = torch.zeros(query_2.shape)
    for i in range(len(attention_weights)):
        context_vector_2 += attention_weights[i] * inputs[i]
    print(f"\ncontext_vector_2\n{context_vector_2}")


def _play_with_simplifed_self_attention():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )

    attention_scores = inputs @ inputs.T

    attention_weights = torch.softmax(attention_scores, dim=1)

    context_vectors = attention_weights @ inputs
    print(f"\ncontext_vectors:\n{context_vectors}")


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        d_k = K.shape[-1]

        attention_scores = Q @ K.transpose(1, 2)
        attention_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attention_weights = torch.softmax(attention_scores / (d_k**0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ V

        return context_vectors


if __name__ == "__main__":
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )

    # inputs -> 6x3 -> ctx_len x d_in
    # W_q, W_k, W_v -> d_in x d_out
    # Q, K, V -> ctx_len x d_out
    # attention_scores -> ctx_len, ctx_len

    # inputs -> 2x6x3 -> batch x ctx_len x d_in
    # W_q, W_k, W_v -> d_in x d_out
    # Q, K, V -> batch x ctx_len x d_out
    # attention_scores -> batch x ctx_len x ctx_len

    d_in = inputs.shape[1]
    d_out = 2

    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]

    torch.manual_seed(123)
    sa = SelfAttention(d_in, d_out, context_length)
    print(sa(batch).shape)
