import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from contrastive.models import LstmNet
from utils_data import aa2id_i, aa2id_o


class BiLSTM(nn.Module):
    def __init__(self, in_dim, embedding_dim, hidden_dim, out_dim, mapping, device):
        super(BiLSTM, self).__init__()
        self.device = device
        self.mapping = mapping
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(in_dim, embedding_dim)
        self.lstm_f = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, Xs):
        _aa2id = self.mapping
        batch_size = len(Xs)

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + seq[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + seq[::-1][:-1] for seq in Xs]

        # get sequence lengths
        Xs_len = [len(seq) for seq in Xs_f]
        lmax = max(Xs_len)

        # list to *.tensor
        Xs_f = [torch.tensor(seq, device='cpu') for seq in Xs_f]
        Xs_b = [torch.tensor(seq, device='cpu') for seq in Xs_b]

        # padding
        Xs_f = pad_sequence(Xs_f, batch_first=True).to(self.device)
        Xs_b = pad_sequence(Xs_b, batch_first=True).to(self.device)

        # embedding
        Xs_f = self.word_embeddings(Xs_f)
        Xs_b = self.word_embeddings(Xs_b)

        # packing the padded sequences
        Xs_f = pack_padded_sequence(Xs_f, Xs_len, batch_first=True, enforce_sorted=False)
        Xs_b = pack_padded_sequence(Xs_b, Xs_len, batch_first=True, enforce_sorted=False)

        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        # unpack outputs
        lstm_out_f, lstm_out_len = pad_packed_sequence(lstm_out_f, batch_first=True)
        lstm_out_b, _ = pad_packed_sequence(lstm_out_b, batch_first=True)

        lstm_out_valid_f = lstm_out_f.reshape(-1, self.hidden_dim)
        lstm_out_valid_b = lstm_out_b.reshape(-1, self.hidden_dim)

        idx_f = []
        [idx_f.extend([i * lmax + j for j in range(l)]) for i, l in enumerate(Xs_len)]
        idx_f = torch.tensor(idx_f, device=self.device)

        idx_b = []
        [idx_b.extend([i * lmax + j for j in range(l)][::-1]) for i, l in enumerate(Xs_len)]
        idx_b = torch.tensor(idx_b, device=self.device)

        lstm_out_valid_f = torch.index_select(lstm_out_valid_f, 0, idx_f)
        lstm_out_valid_b = torch.index_select(lstm_out_valid_b, 0, idx_b)

        lstm_out_valid = lstm_out_valid_f + lstm_out_valid_b

        # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # compute scores
        # TODO: checkme
        scores = F.log_softmax(out, dim=1)

        return scores

    def set_param(self, param_dict):
        try:
            for pn, _ in self.named_parameters():
                exec('self.%s.data = torch.tensor(param_dict[pn])' % pn)
            self.hidden_dim = param_dict['hidden_dim']
            self.fixed_len = param_dict['fixed_len']
            self.forward = self.forward_flen if self.fixed_len else self.forward_vlen
            self.to(self.device)
        except:
            print('Unmatched parameter names or shapes.')

    def get_param(self):
        param_dict = {}
        for pn, pv in self.named_parameters():
            param_dict[pn] = pv.data.cpu().numpy()
        param_dict['hidden_dim'] = self.hidden_dim
        param_dict['fixed_len'] = self.fixed_len
        return param_dict


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

    def __init__(self, embedding_dim=64, hidden_dim=64):
        super(ContrastiveSeq, self).__init__()
        '''
        in_dim=in_dim, embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim,
                              out_dim=out_dim, device=device,
                              mapping=aa2id_i[gapped]                    
                              
                              LstmNet
                              LSTM(input_size=embedding_dim,
                           hidden_size=hidden_dim,
                           dropout=0.2,
                           num_layers=1,
                           bidirectional=False,
                           batch_first=True)

        '''
        self.bi_lstm = LstmNet()

    def forward(self, sequence1, sequence2):
        output1 = self.bi_lstm(sequence1)
        output2 = self.bi_lstm(sequence2)

        # Manhatta distance calculation
        dist = torch.sum(torch.abs(output1 - output2), dim=1, keepdim=True)
        return dist
