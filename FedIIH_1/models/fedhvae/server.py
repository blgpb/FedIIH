import time
import numpy as np
import torch

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule
from torch import nn
from torch_geometric.nn import GCNConv


def kl_divergence(mu_A, logvar_A, mu_B, logvar_B):
    var_A = np.exp(logvar_A)
    var_B = np.exp(logvar_B)

    kl = np.sum(0.5 * (logvar_B - logvar_A + (var_A + (mu_A - mu_B) ** 2) / var_B - 1))

    return kl


def js_divergence(mu_A, logvar_A, mu_B, logvar_B):
    var_A = np.exp(logvar_A)
    var_B = np.exp(logvar_B)

    mu_M = 0.5 * (mu_A + mu_B)
    var_M = 0.5 * (var_A + var_B)

    kl_AM = kl_divergence(mu_A, logvar_A, mu_M, np.log(var_M))
    kl_BM = kl_divergence(mu_B, logvar_B, mu_M, np.log(var_M))

    js = 0.5 * (kl_AM + kl_BM)

    return js


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)

        self.model = DisentangledGNN(self.args.n_feat, self.args.n_clss, args).cuda(self.gpu_id)

        self.update_lists = []
        self.sim_matrices_n = []

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        if self.curr_rnd != 0:
            self.load_state()
        self.sd['global'] = self.get_weights()

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, 'server_state.pt')

        set_state_dict(self.model, loaded['model'], self.gpu_id)

        self.sim_matrices_n = loaded['sim_matrices_n']

    def get_weights(self):
        return {
            'model': get_state_dict(self.model)
        }

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()

        local_weights = []

        local_train_sizes = []

        clients_z_mu_n, clients_z_logvar_n = [], []

        for c_id in updated:
            clients_z_mu_n.append(self.sd[c_id]['z_mu_n'])
            clients_z_logvar_n.append(self.sd[c_id]['z_logvar_n'])

            local_weights.append(self.sd[c_id]['model'].copy())

            local_train_sizes.append(self.sd[c_id]['train_size'])

            del self.sd[c_id]

        n_connected = round(self.args.n_clients * self.args.frac)
        assert n_connected == len(clients_z_mu_n)

        sim_matrix_n = np.empty(shape=(n_connected, n_connected))

        self.sd[f'Beta_mu'] = np.sum(clients_z_mu_n, axis=0) / (n_connected + 0.25)

        for i in range(n_connected):
            for j in range(n_connected):
                mu_A = clients_z_mu_n[i]  
                logvar_A = clients_z_logvar_n[i]  

                mu_B = clients_z_mu_n[j]  
                logvar_B = clients_z_logvar_n[j]  

                js = js_divergence(mu_A, logvar_A, mu_B, logvar_B)
                sim_matrix_n[i, j] = 1 - js / np.log(2)

        if self.args.agg_norm == 'exp':
            sim_matrix_n = np.exp(self.args.norm_scale * sim_matrix_n)

        row_sums = sim_matrix_n.sum(axis=1)
        sim_matrix_n = sim_matrix_n / row_sums[:, np.newaxis]

        st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()

        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')

        print(sim_matrix_n)

        st = time.time()
        for i, c_id in enumerate(updated):
            aggr_local_model_weights = self.aggregate(local_weights, sim_matrix_n[i, :])

            if f'personalized_{c_id}' in self.sd: del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = {'model': aggr_local_model_weights}

        self.update_lists.append(updated)
        self.sim_matrices_n.append(sim_matrix_n)

        self.logger.print(f'local model has been updated ({time.time() - st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),

            'sim_matrices_n': self.sim_matrices_n,

            'update_lists': self.update_lists
        })