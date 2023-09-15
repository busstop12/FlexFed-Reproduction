import torch.backends.mps
import train

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# train.train('BASIC-COMMON', device)
train.train('CLUSTERED-COMMON', device)
# train.train('MAX-COMMON', device)

