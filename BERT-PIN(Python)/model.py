#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  26 2023
BERT-PIN model
@author: Yi Hu
"""

import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)    # (N, value_len, embed_size)
        keys = self.keys(keys)          # (N, key_len, embed_size)
        queries = self.queries(query)   # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        alpha = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # alpha: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            alpha = alpha.masked_fill(mask == True, float("-1e20"))

        # Normalize alpha values
        # attention shape: (N, heads, query_len, key_len)
        attention = torch.softmax(alpha / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim),
        # then reshape and flatten the last two dimensions.

        out = self.fc_out(out)  # (N, query_len, embed_size)

        return out

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerLayer, self).__init__()
        self.attention = MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):

        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.temperature_embedding = nn.Embedding(src_vocab_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, embed_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, temperature, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions) + self.temperature_embedding(temperature))
        ).to(self.device)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        out = self.logsoftmax(out)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size=4000,
        embed_size=5000,
        num_layers=4,
        device="cup",
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=96,
    ):

        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device
        )

        self.device = device
        self.name = 'BERT'

    def make_src_mask(self, mask):
        src_mask = mask.unsqueeze(1).unsqueeze(2).to(self.device)
        # (N, 1, 1, src_len)
        return src_mask

    def forward(self, src, temperature, mask):
        # mask for patched load  False-True-False
        src_mask = self.make_src_mask(mask).to(self.device)
        enc_src = self.encoder(src, temperature, src_mask)

        return enc_src

