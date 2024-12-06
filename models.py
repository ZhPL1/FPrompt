import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class PreTrainGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.gnns = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.gnns.append(GCNConv(in_channels, hidden_channels))
        self.activations.append(nn.LeakyReLU())
        for _ in range(num_layers - 2):
            self.gnns.append(GCNConv(hidden_channels, hidden_channels))
            self.activations.append(nn.LeakyReLU())
        self.gnns.append(GCNConv(hidden_channels, hidden_channels))
        self.activations.append(nn.LeakyReLU())
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.gnns[:-1]):
            x = conv(x, edge_index)
            x = self.activations[i](x)
        x = self.gnns[-1](x, edge_index)
        x = self.activations[-1](x)
        return x


class TuneAdapter(torch.nn.Module):
    def __init__(self, in_channels):
        super(TuneAdapter, self).__init__()
        self.adapter = nn.Sequential(nn.Linear(in_channels, in_channels // 2),
                                 nn.LeakyReLU(),
                                 nn.Linear(in_channels // 2, in_channels),
                                 nn.BatchNorm1d(in_channels, affine=False))

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.adapter(x)
        return x


class TuneClassifier(nn.Module):
    def __init__(self, in_channels):
        super(TuneClassifier, self).__init__()
        self.cla = nn.Linear(in_channels, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.cla(x)


class TunePrompt(nn.Module):
    def __init__(self, in_channels):
        super(TunePrompt, self).__init__()
        self.weight= nn.Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return x + self.weight