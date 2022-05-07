import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class TextCnn(nn.Module):

    def __init__(self, dropout=0.5):
        super(TextCnn, self).__init__()

        token_dim = 150
        class_num = 2
        kernel_num = 200
        kernel_sizes = [3, 4, 5, 6, 7]
        print(" kernel_num: {} kernel_sizes: {}".format(kernel_num, kernel_sizes))

        Ci = 1
        Co = kernel_num  # 200

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, token_dim), padding=(2, 0)) for f in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)

        # print("1: {} ".format(x.shape))  # [batch * 1 * max_seq_length * embedding_size]

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]

        # print("2: {} ".format([x_c.shape for x_c in x]))  # [[batch * out_channel * kernel_size]]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]

        # print("3: {} ".format([x_c.shape for x_c in x]))

        x = torch.cat(x, 1)  # (N, Co * len(kernel_sizes))
        # x = torch.stack(x, dim=1)

        # print("4: {} ".format(x.shape))

        x = self.dropout(x)  # (N, Co * len(kernel_sizes))

        # print("5: {} ".format(x.shape))

        logit = self.fc(x)  # (N, class_num)

        # print("6: {} ".format(logit))

        logit = torch.sigmoid(logit)
        # logit = F.softmax(logit, dim=0)  # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax

        # print("7: {} ".format(logit))

        return logit


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self,
                 ntoken: int,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
