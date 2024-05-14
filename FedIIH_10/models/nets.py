import torch.nn as nn
import torch.nn.functional as F

from misc.utils import *
from torch_geometric.nn import GCNConv



class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


class DisenConv(nn.Module):
    def __init__(self, latent_factor_nums, niter, tau=1.0):
        super(DisenConv, self).__init__()
        self.k = latent_factor_nums
        self.niter = niter
        self.tau = tau

    def forward(self, x, edge_index):
        m, src, trg = edge_index.shape[1], edge_index[0], edge_index[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)
        u = x
        scatter_idx = trg.view(m, 1).expand(m, d)
        for clus_iter in range(self.niter):
            p = (z * u[trg].view(m, k, delta_d)).sum(dim=2)
            p = F.softmax(p / self.tau, dim=1)
            scatter_src = (z * p.view(m, k, 1)).view(m, d)
            u = torch.zeros(n, d, device=x.device)
            u.scatter_add_(0, scatter_idx, scatter_src)
            u += x

            u = F.normalize(u.view(n, k, delta_d), dim=2).view(n, d)
        return u


class DisentangledGNN(nn.Module):


    def __init__(self, nfeat, nclass, args):
        super(DisentangledGNN, self).__init__()
        self.latent_factor_nums = 10
        self.latent_factor_dims = args.n_latentdims
        self.routit = args.n_routit
        self.nlayer = args.n_layers

        if args.n_layers <= 2:
            self.nlayer = 1
            self.vae_nlayer = 1
        else:
            self.nlayer = args.n_layers - 2
            self.vae_nlayer = 2

        self.pca = SparseInputLinear(nfeat, self.latent_factor_nums * self.latent_factor_dims)

        self.base_gnn_ls = nn.ModuleList([DisenConv(self.latent_factor_nums, self.routit) for i in range(self.nlayer)])
        self.gnn_mean = nn.ModuleList([DisenConv(self.latent_factor_nums, self.routit) for i in range(self.vae_nlayer)])
        self.gnn_logstddev = nn.ModuleList(
            [DisenConv(self.latent_factor_nums, self.routit) for i in range(self.vae_nlayer)])

        self.clf = nn.Linear(self.latent_factor_dims * self.latent_factor_nums, nclass)
        self.dropout = args.dropout

    def _dropout(self, x):
        return F.dropout(x, self.dropout, training=self.training)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self._dropout(F.leaky_relu(self.pca(x)))
        for conv in self.base_gnn_ls:
            x = self._dropout(F.leaky_relu(conv(x, edge_index)))
        x = self.clf(x)
        return x

    def encode_for_HVAE(self, data):
        x, edge_index = data.x, data.edge_index

        x = self._dropout(F.leaky_relu(self.pca(x)))

        for conv in self.base_gnn_ls:
            x = self._dropout(F.leaky_relu(conv(x, edge_index)))

        mean = x
        for conv in self.gnn_mean:
            mean = self._dropout(F.leaky_relu(conv(mean, edge_index)))
        z_mu_n, z_mu_e, z_mu_n1, z_mu_e1, z_mu_n2, z_mu_e2, z_mu_n3, z_mu_e3, z_mu_n4, z_mu_e4 = torch.chunk(mean,
                                                                                                             chunks=self.latent_factor_nums,
                                                                                                             dim=1)

        logstd = x
        for conv in self.gnn_logstddev:
            logstd = self._dropout(F.leaky_relu(conv(logstd, edge_index)))
        z_logvar_n, z_logvar_e, z_logvar_n1, z_logvar_e1, z_logvar_n2, z_logvar_e2, z_logvar_n3, z_logvar_e3, z_logvar_n4, z_logvar_e4 = torch.chunk(
            logstd,
            chunks=self.latent_factor_nums,
            dim=1)

        return z_mu_n, z_logvar_n, z_mu_e, z_logvar_e, z_mu_n1, z_logvar_n1, z_mu_e1, z_logvar_e1, z_mu_n2, z_logvar_n2, z_mu_e2, z_logvar_e2, z_mu_n3, z_logvar_n3, z_mu_e3, z_logvar_e3, z_mu_n4, z_logvar_n4, z_mu_e4, z_logvar_e4
