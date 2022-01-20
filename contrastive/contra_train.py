from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from tqdm import tqdm

from contrastive.cont_model import ContrastiveSeq
from utils_data import aa2id_i


def to_id(seq: str, gapped: bool) -> List[int]:
    mapping = aa2id_i[gapped]
    eos = mapping['<EOS>']
    seq = [mapping['<SOS>']] + list(map(lambda ch: mapping[ch], seq)) + [eos]
    to_pad = 140 - len(seq) + 2
    return torch.IntTensor(seq + [eos] * to_pad)


class ContrastiveSeqDataset(Dataset):
    def __init__(self, file_path, gapped=True):
        self.gapped = gapped
        self.data = pd.read_csv(file_path)[['seq_one', 'seq_two', 'the_same']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        X = (to_id(row.seq_one, gapped=self.gapped), to_id(row.seq_two, gapped=self.gapped))
        return X, 1 if row.the_same else 0


def padding_collate(batch):
    X1 = []
    X2 = []
    y = []

    for i in range(len(batch)):
        # batch[i][0] is X, batch[i][1] is y
        x1 = batch[i][0][0]
        x2 = batch[i][0][1]

        X1.append(x1)
        X2.append(x2)

        y.append([batch[i][1]])

    X1 = torch.stack(X1)
    X2 = torch.stack(X2)

    return (X1, X2), torch.flatten(torch.IntTensor(y))


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss_contrastive = torch.mean(1 / 2 * (label) * torch.pow(dist, 2) +
                                      1 / 2 * (1 - label) * torch.pow(F.relu(self.margin - dist), 2)
                                      )

        return loss_contrastive


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")

    n_epoch = 2
    train_ds = ContrastiveSeqDataset(file_path="./data/contrastive_train_vlen.csv", gapped=True)
    test_ds = ContrastiveSeqDataset(file_path="./data/contrastive_test_vlen.csv", gapped=True)

    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=128, shuffle=True, collate_fn=padding_collate)
    test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=128, shuffle=False, collate_fn=padding_collate)

    model = ContrastiveSeq(embedding_dim=64)

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)
    # to track minimum validation loss
    min_loss = np.inf
    loss_fn = ContrastiveLoss()
    # model.cuda()
    for epoch in range(n_epoch):
        model.train()
        loss_avg, cnt = 0, 0
        with tqdm(total=len(train_ds), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
            for batch, labels in train_dl:
                seq_one, seq_two, labels = batch[0].to(device), batch[1].to(device), labels.to(device)
                model.zero_grad()
                output = model(seq_one, seq_two)
                loss = loss_fn(torch.flatten(output), labels)
                loss.backward()
                optimizer.step()

                L = len(labels)
                loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                cnt += L

                pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc': '0'})
                pbar.update(len(batch))
        scheduler.step()

        """
        # validation
        model.eval()
        loss_avg, acc_avg, cnt = 0, 0, 0
        with torch.set_grad_enabled(False):
            with tqdm(total=len(test_ds), desc='          (VLD)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for batch, batch_flatten in test_dl:
                    # targets
                    batch_flatten = torch.tensor(batch_flatten, device=model.device)

                    # forward routine
                    scores = model(batch, aa2id_i[self.gapped])
                    loss = loss_fn(scores, batch_flatten)

                    # compute statistics
                    L = len(batch_flatten)
                    predicted = torch.argmax(scores, 1)

                    corr = (predicted == batch_flatten).data.cpu().numpy()

                    loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                    acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
                    cnt += L

                    # update progress bar
                    pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc': '{:.6f}'.format(acc_avg)})
                    pbar.update(len(batch))
                wandb.log({'test_lost_avg': loss_avg, 'test_acc_avg': acc_avg})

            """

"""        # save model
        if loss_avg < min_loss and save_fp:
            min_loss = loss_avg
            self.save('{}/lstm_{:.6f}.npy'.format(save_fp, loss_avg))
"""
