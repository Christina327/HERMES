import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import random


def set_seed_all(rd_seed):
    random.seed(rd_seed)
    np.random.seed(rd_seed)

    torch.manual_seed(rd_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(rd_seed)
        torch.cuda.manual_seed_all(rd_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class TruncatedExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, min_lr=0, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


class LambdaLayer(torch.nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def get_metrics(yt, yp):
    auc = roc_auc_score(yt, yp)
    precision, recall, _ = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    return auc, aupr


def get_MACCS(smiles: str):
    # "smiles" string => Molecule object
    m = AllChem.MolFromSmiles(smiles)
    arr = np.zeros((1,), np.float32)
    # fingerprint
    fp = MACCSkeys.GenMACCSKeys(m)  # ExplicitBitVect
    DataStructs.ConvertToNumpyArray(fp, arr)  # 1-d nd_array
    return arr
