import torch
import torch.nn as nn

from contrastive.models import LstmNet
from utils_data import aa2id_i, aa2id_o


class LstmNet(torch.nn.Module):

    def __init__(self, embedding_dim=64, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16):
        super(LstmNet, self).__init__()
        in_dim, out_dim = len(aa2id_i[True]), len(aa2id_o[True])
        self.word_embeddings = nn.Embedding(in_dim, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim1, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(2 * hidden_dim1, hidden_dim2, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm3 = torch.nn.LSTM(2 * hidden_dim2, hidden_dim3, dropout=0.2, batch_first=True, bidirectional=True)

    def forward(self, sequence):
        x = self.word_embeddings(sequence)
        x, _ = self.lstm(x)
        x = torch.nn.ReLU()(x)

        x, _ = self.lstm2(x)
        x = torch.nn.ReLU()(x)

        _, (h, _) = self.lstm3(x)
        return h[-1, :, :]


class ContrastiveSeq(nn.Module):

    def __init__(self):
        super(ContrastiveSeq, self).__init__()
        self.bi_lstm = LstmNet()

    def forward(self, sequence1, sequence2):
        output1 = self.bi_lstm(sequence1)
        output2 = self.bi_lstm(sequence2)

        # Manhatta distance calculation
        dist = torch.sum(torch.abs(output1 - output2), dim=1, keepdim=True)
        return dist
